#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/environ_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#if defined(TI_WITH_VULKAN)
#include "taichi/program/vulkan_command_replay.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {

constexpr uint32_t kRadixBits = 4;
constexpr uint32_t kRadixBins = 1u << kRadixBits;
constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kSingleChunkPrefixMaxBlocks = 32;
constexpr uint32_t kInlineChunkPrefixMaxChunks = 4;
constexpr uint32_t kRadix8Bins = 256;
constexpr uint32_t kRadix8PartitionSize = 2048;
constexpr uint32_t kHistogramPrivateChunkSize = 2048;
constexpr uint32_t kReducePrivateChunkSize = 2048;

size_t vulkan_resource_replay_ring_capacity(size_t requested = 1) {
  const int configured =
      get_environ_config("TI_VULKAN_RESOURCE_REPLAY_RING_SIZE", 32);
  const size_t base =
      static_cast<size_t>(std::max(1, configured));
  return std::max(base, requested);
}

struct VulkanResourceSetRing {
  Device *device{nullptr};
  std::vector<std::unique_ptr<ShaderResourceSet>> sets;
  size_t cursor{0};
  size_t capacity{0};

  void reset() {
    sets.clear();
    cursor = 0;
    capacity = 0;
    device = nullptr;
  }

  void ensure_device(Device *dev, size_t requested = 1) {
    if (device == dev) {
      capacity = std::max(capacity, vulkan_resource_replay_ring_capacity(requested));
      return;
    }
    reset();
    device = dev;
    capacity = vulkan_resource_replay_ring_capacity(requested);
  }

  void rewind_for(size_t requested) {
    TI_ASSERT(requested <= capacity);
    if (cursor > capacity - requested) {
      cursor = 0;
    }
  }

  std::pair<ShaderResourceSet *, size_t> next_slot(Device *dev) {
    ensure_device(dev);
    if (cursor >= capacity) {
      cursor = 0;
    }
    if (cursor == sets.size()) {
      sets.emplace_back(dev->create_resource_set());
    }
    const size_t slot = cursor;
    ShaderResourceSet *result = sets[slot].get();
    ++cursor;
    return {result, slot};
  }

  ShaderResourceSet *next(Device *dev) {
    return next_slot(dev).first;
  }

  void ensure_preallocated(Device *dev, size_t requested) {
    ensure_device(dev, requested);
    while (sets.size() < capacity) {
      sets.emplace_back(dev->create_resource_set());
    }
  }
};

struct VulkanRwBufferBindingKey {
  DeviceAllocation alloc{kDeviceNullAllocation};
  uint64_t generation{0};
  uint64_t offset{0};
  size_t bytes{0};
  bool valid{false};

  bool matches(DeviceAllocation requested_alloc,
               uint64_t requested_generation,
               uint64_t requested_offset,
               size_t requested_bytes) const {
    return valid && alloc == requested_alloc &&
           generation == requested_generation && offset == requested_offset &&
           bytes == requested_bytes;
  }

  void set(DeviceAllocation requested_alloc,
           uint64_t requested_generation,
           uint64_t requested_offset,
           size_t requested_bytes) {
    alloc = requested_alloc;
    generation = requested_generation;
    offset = requested_offset;
    bytes = requested_bytes;
    valid = true;
  }

  void reset() {
    alloc = kDeviceNullAllocation;
    generation = 0;
    offset = 0;
    bytes = 0;
    valid = false;
  }
};

struct VulkanRwBufferBindingRequest {
  DeviceAllocation alloc{kDeviceNullAllocation};
  uint64_t offset{0};
  size_t bytes{0};
};

VulkanRwBufferBindingRequest rw_buffer_request(DeviceAllocation alloc,
                                               uint64_t offset,
                                               size_t bytes) {
  return {alloc, offset, bytes};
}

uint64_t vulkan_allocation_generation(DeviceAllocation alloc) {
  if (alloc == kDeviceNullAllocation) {
    return 0;
  }
  return static_cast<vulkan::VulkanDevice *>(alloc.device)
      ->allocation_generation(alloc);
}

template <size_t N>
struct VulkanRwBufferReplay {
  std::array<VulkanRwBufferBindingKey, N> bindings;

  void reset() {
    for (auto &binding : bindings) {
      binding.reset();
    }
  }

  bool rw_buffer(ShaderResourceSet *resource_set,
                 uint32_t binding,
                 DeviceAllocation alloc,
                 uint64_t offset,
                 size_t bytes) {
    auto &cached = bindings[binding];
    const uint64_t generation = vulkan_allocation_generation(alloc);
    if (cached.matches(alloc, generation, offset, bytes)) {
      return false;
    }
    resource_set->rw_buffer(binding, alloc.get_ptr(offset), bytes);
    cached.set(alloc, generation, offset, bytes);
    return true;
  }

  bool matches(uint32_t binding,
               DeviceAllocation alloc,
               uint64_t offset,
               size_t bytes) const {
    const auto &cached = bindings[binding];
    if (!cached.valid) {
      return false;
    }
    const uint64_t generation = vulkan_allocation_generation(alloc);
    return cached.matches(alloc, generation, offset, bytes);
  }
};

void prepare_resource_set_replay(Program *program,
                                 Device *device,
                                 VulkanResourceSetRing &ring,
                                 size_t requested);

template <size_t N>
struct VulkanReplayResourceSet {
  ShaderResourceSet *bindings{nullptr};
  VulkanRwBufferReplay<N> *replay{nullptr};
};

void push_vulkan_command_key_range(VulkanCommandReplayKey &key,
                                   DeviceAllocation alloc,
                                   uint64_t offset,
                                   size_t bytes) {
  key.push(alloc.alloc_id);
  key.push(vulkan_allocation_generation(alloc));
  key.push(offset);
  key.push(static_cast<uint64_t>(bytes));
}

template <size_t N>
struct VulkanResourceSetReplayRing {
  Device *device{nullptr};
  VulkanResourceSetRing ring;
  std::vector<VulkanRwBufferReplay<N>> replays;
  bool hot_valid{false};
  size_t hot_slot{0};

  void reset() {
    ring.reset();
    replays.clear();
    device = nullptr;
    hot_valid = false;
    hot_slot = 0;
  }

  void prepare(Program *program, Device *dev, size_t requested) {
    if (requested == 0) {
      return;
    }
    if (device != dev) {
      replays.clear();
      device = dev;
      hot_valid = false;
    }
    ring.ensure_device(dev, requested);
    // VulkanResourceSet::finalize() replaces a descriptor set while its old
    // instance is pinned by a recorded or submitted command buffer. Reusing a
    // wrapper slot therefore does not require a device wait: the command
    // buffer owns the exact descriptor set it recorded until fence retirement.
    ring.rewind_for(requested);
    (void)program;
    if (replays.size() < ring.capacity) {
      replays.resize(ring.capacity);
    }
  }

  VulkanReplayResourceSet<N> next(Device *dev) {
    auto [bindings, slot] = ring.next_slot(dev);
    if (device != dev) {
      replays.clear();
      device = dev;
      hot_valid = false;
    }
    if (replays.size() <= slot) {
      replays.resize(slot + 1);
    }
    hot_valid = true;
    hot_slot = slot;
    return {bindings, &replays[slot]};
  }

  bool hot_matches(
      const std::array<VulkanRwBufferBindingRequest, N> &requests) const {
    if (!hot_valid || hot_slot >= ring.sets.size() ||
        hot_slot >= replays.size()) {
      return false;
    }
    const auto &replay = replays[hot_slot];
    for (uint32_t i = 0; i < N; ++i) {
      if (!replay.matches(i, requests[i].alloc, requests[i].offset,
                          requests[i].bytes)) {
        return false;
      }
    }
    return true;
  }

  VulkanReplayResourceSet<N> bind(
      Program *program,
      Device *dev,
      const std::array<VulkanRwBufferBindingRequest, N> &requests) {
    if (device != dev) {
      replays.clear();
      device = dev;
      hot_valid = false;
    }
    if (hot_matches(requests)) {
      return {ring.sets[hot_slot].get(), &replays[hot_slot]};
    }
    prepare(program, dev, 1);
    auto resource = next(dev);
    for (uint32_t i = 0; i < N; ++i) {
      resource.replay->rw_buffer(resource.bindings, i, requests[i].alloc,
                                 requests[i].offset, requests[i].bytes);
    }
    return resource;
  }
};

void prepare_resource_set_replay(Program *program,
                                 Device *device,
                                 VulkanResourceSetRing &ring,
                                 size_t requested) {
  if (requested == 0) {
    return;
  }
  ring.ensure_device(device, requested);
  ring.rewind_for(requested);
  (void)program;
  ring.ensure_preallocated(device, requested);
}

struct VulkanSortCpuProfileSample {
  uint64_t sort_calls{0};
  uint64_t lambda_calls{0};
  uint64_t workspace_reallocs{0};
  uint64_t realloc_sync_calls{0};
  uint64_t internal_sync_calls{0};
  uint64_t dispatch_calls{0};
  uint64_t bind_pipeline_calls{0};
  uint64_t bind_shader_resources_calls{0};
  uint64_t resource_set_calls{0};
  uint64_t resource_set_create_calls{0};
  uint64_t rw_buffer_calls{0};
  uint64_t buffer_fill_calls{0};
  uint64_t buffer_barrier_calls{0};
  uint64_t buffer_copy_calls{0};
  double total_call_us{0.0};
  double get_cache_us{0.0};
  double ensure_workspace_us{0.0};
  double enqueue_us{0.0};
  double realloc_sync_us{0.0};
  double internal_sync_us{0.0};
  double lambda_total_us{0.0};
  double radix8_body_us{0.0};
  double bind_pipeline_us{0.0};
  double bind_shader_resources_us{0.0};
  double profiler_scope_us{0.0};
  double dispatch_us{0.0};
  double resource_set_us{0.0};
  double rw_buffer_us{0.0};
  double buffer_fill_us{0.0};
  double buffer_barrier_us{0.0};
  double buffer_copy_us{0.0};

  void add(const VulkanSortCpuProfileSample &other) {
#define TI_VULKAN_SORT_PROFILE_ADD_FIELD(name) name += other.name
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(sort_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(lambda_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(workspace_reallocs);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(realloc_sync_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(internal_sync_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(dispatch_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_pipeline_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_shader_resources_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_create_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(rw_buffer_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_fill_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_barrier_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_copy_calls);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(total_call_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(get_cache_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(ensure_workspace_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(enqueue_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(realloc_sync_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(internal_sync_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(lambda_total_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(radix8_body_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_pipeline_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(bind_shader_resources_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(profiler_scope_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(dispatch_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(resource_set_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(rw_buffer_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_fill_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_barrier_us);
    TI_VULKAN_SORT_PROFILE_ADD_FIELD(buffer_copy_us);
#undef TI_VULKAN_SORT_PROFILE_ADD_FIELD
  }
};

struct VulkanSortCpuProfileTotals {
  std::mutex mutex;
  VulkanSortCpuProfileSample totals;

  void clear() {
    std::lock_guard<std::mutex> guard(mutex);
    totals = {};
  }

  void merge(const VulkanSortCpuProfileSample &sample) {
    std::lock_guard<std::mutex> guard(mutex);
    totals.add(sample);
  }

  std::string report_json() {
    std::lock_guard<std::mutex> guard(mutex);
    const auto &t = totals;
    const double denom = t.sort_calls ? static_cast<double>(t.sort_calls) : 1.0;
    auto per_sort = [&](double value) { return value / denom; };
    auto count_per_sort = [&](uint64_t value) {
      return static_cast<double>(value) / denom;
    };
    std::ostringstream os;
    os << std::fixed << std::setprecision(6);
    os << "{";
    os << "\"sort_calls\":" << t.sort_calls << ",";
    os << "\"lambda_calls\":" << t.lambda_calls << ",";
    os << "\"counts_per_sort\":{";
    os << "\"workspace_reallocs\":" << count_per_sort(t.workspace_reallocs)
       << ",";
    os << "\"realloc_sync_calls\":" << count_per_sort(t.realloc_sync_calls)
       << ",";
    os << "\"internal_sync_calls\":" << count_per_sort(t.internal_sync_calls)
       << ",";
    os << "\"dispatch_calls\":" << count_per_sort(t.dispatch_calls) << ",";
    os << "\"bind_pipeline_calls\":" << count_per_sort(t.bind_pipeline_calls)
       << ",";
    os << "\"bind_shader_resources_calls\":"
       << count_per_sort(t.bind_shader_resources_calls) << ",";
    os << "\"resource_set_calls\":" << count_per_sort(t.resource_set_calls)
       << ",";
    os << "\"resource_set_create_calls\":"
       << count_per_sort(t.resource_set_create_calls) << ",";
    os << "\"rw_buffer_calls\":" << count_per_sort(t.rw_buffer_calls) << ",";
    os << "\"buffer_fill_calls\":" << count_per_sort(t.buffer_fill_calls)
       << ",";
    os << "\"buffer_barrier_calls\":"
       << count_per_sort(t.buffer_barrier_calls) << ",";
    os << "\"buffer_copy_calls\":" << count_per_sort(t.buffer_copy_calls);
    os << "},";
    os << "\"per_sort_us\":{";
    os << "\"total_call_us\":" << per_sort(t.total_call_us) << ",";
    os << "\"get_cache_us\":" << per_sort(t.get_cache_us) << ",";
    os << "\"ensure_workspace_us\":" << per_sort(t.ensure_workspace_us) << ",";
    os << "\"enqueue_us\":" << per_sort(t.enqueue_us) << ",";
    os << "\"realloc_sync_us\":" << per_sort(t.realloc_sync_us) << ",";
    os << "\"internal_sync_us\":" << per_sort(t.internal_sync_us) << ",";
    os << "\"lambda_total_us\":" << per_sort(t.lambda_total_us) << ",";
    os << "\"radix8_body_us\":" << per_sort(t.radix8_body_us) << ",";
    os << "\"bind_pipeline_us\":" << per_sort(t.bind_pipeline_us) << ",";
    os << "\"bind_shader_resources_us\":"
       << per_sort(t.bind_shader_resources_us) << ",";
    os << "\"profiler_scope_us\":" << per_sort(t.profiler_scope_us) << ",";
    os << "\"dispatch_us\":" << per_sort(t.dispatch_us) << ",";
    os << "\"resource_set_us\":" << per_sort(t.resource_set_us) << ",";
    os << "\"rw_buffer_us\":" << per_sort(t.rw_buffer_us) << ",";
    os << "\"buffer_fill_us\":" << per_sort(t.buffer_fill_us) << ",";
    os << "\"buffer_barrier_us\":" << per_sort(t.buffer_barrier_us) << ",";
    os << "\"buffer_copy_us\":" << per_sort(t.buffer_copy_us);
    os << "}}";
    return os.str();
  }
};

VulkanSortCpuProfileTotals g_vulkan_sort_cpu_profile;

bool vulkan_sort_cpu_profile_enabled() {
  return get_environ_config("TI_VULKAN_SORT_CPU_PROFILE", 0) != 0;
}

double profile_time_us() {
  return Time::get_time() * 1000000.0;
}

static const uint32_t kInitI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/init_i32.comp.spv.h"
    ;
static const uint32_t kCopyI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/copy_i32.comp.spv.h"
    ;
static const uint32_t kSortInitU32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_u32_index.comp.spv.h"
    ;
static const uint32_t kSortInitI32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_i32_index.comp.spv.h"
    ;
static const uint32_t kSortInitF32IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_f32_index.comp.spv.h"
    ;
static const uint32_t kSortInitU64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_u64_index.comp.spv.h"
    ;
static const uint32_t kSortInitI64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_i64_index.comp.spv.h"
    ;
static const uint32_t kSortInitF64IndexSpv[] =
#include "taichi/program/vulkan_sort_shaders/sort_init_f64_index.comp.spv.h"
    ;
static const uint32_t kGatherU32ByU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_u32_by_u32.comp.spv.h"
    ;
static const uint32_t kPrefixBlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_block.comp.spv.h"
    ;
static const uint32_t kPrefixChunksSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_chunks.comp.spv.h"
    ;
static const uint32_t kPrefixSingleChunkSpv[] =
#include "taichi/program/vulkan_sort_shaders/prefix_single_chunk.comp.spv.h"
    ;
static const uint32_t kScanI32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block.comp.spv.h"
    ;
static const uint32_t kScanI32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanI32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanI32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add.comp.spv.h"
    ;
static const uint32_t kScanI32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanF32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block.comp.spv.h"
    ;
static const uint32_t kScanF32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanF32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanF32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add.comp.spv.h"
    ;
static const uint32_t kScanF32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanU32BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block.comp.spv.h"
    ;
static const uint32_t kScanU32BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_strided.comp.spv.h"
    ;
static const uint32_t kScanU32BlockSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_subgroup.comp.spv.h"
    ;
static const uint32_t kScanU32AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add.comp.spv.h"
    ;
static const uint32_t kScanU32AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add_strided.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup_strided.comp.spv.h"
    ;
static const uint32_t kScanU64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block.comp.spv.h"
    ;
static const uint32_t kScanU64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanU64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add.comp.spv.h"
    ;
static const uint32_t kScanU64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add_strided.comp.spv.h"
    ;
static const uint32_t kScanI64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block.comp.spv.h"
    ;
static const uint32_t kScanI64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanI64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add.comp.spv.h"
    ;
static const uint32_t kScanI64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add_strided.comp.spv.h"
    ;
static const uint32_t kScanF64BlockSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block.comp.spv.h"
    ;
static const uint32_t kScanF64BlockStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block_strided.comp.spv.h"
    ;
static const uint32_t kScanF64AddSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add.comp.spv.h"
    ;
static const uint32_t kScanF64AddStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add_strided.comp.spv.h"
    ;
static const uint32_t kScanI32BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanI32BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanI32AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanI32AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup_reverse.comp.spv.h"
    ;
static const uint32_t kScanI32SmallSubgroupStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i32_small_subgroup_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup_reverse.comp.spv.h"
    ;
static const uint32_t kScanF32SmallSubgroupStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f32_small_subgroup_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup_reverse.comp.spv.h"
    ;
static const uint32_t kScanU32SmallSubgroupStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u32_small_subgroup_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanU64BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanU64BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanU64AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanU64AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_u64_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanI64BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanI64BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanI64AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanI64AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_i64_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanF64BlockReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block_reverse.comp.spv.h"
    ;
static const uint32_t kScanF64BlockStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_block_strided_reverse.comp.spv.h"
    ;
static const uint32_t kScanF64AddReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add_reverse.comp.spv.h"
    ;
static const uint32_t kScanF64AddStridedReverseSpv[] =
#include "taichi/program/vulkan_sort_shaders/scan_f64_add_strided_reverse.comp.spv.h"
    ;
static const uint32_t kCompactI32FlagsSpv[] =
#include "taichi/program/vulkan_sort_shaders/compact_i32_flags.comp.spv.h"
    ;
static const uint32_t kCompactI32ScatterSpv[] =
#include "taichi/program/vulkan_sort_shaders/compact_i32_scatter.comp.spv.h"
    ;
static const uint32_t kCompactI32ScatterPacketSpv[] =
#include "taichi/program/vulkan_sort_shaders/compact_i32_scatter_packet.comp.spv.h"
    ;
static const uint32_t kHistogramI32ClearSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_clear.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI32ReducePrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_reduce_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32SingleSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_single_shared.comp.spv.h"
    ;
static const uint32_t kHistogramU32CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramU32CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramU32SingleSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_single_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI64ClearSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i64_clear.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramI32I64CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i32_i64_count_private_shared.comp.spv.h"
    ;
static const uint32_t kHistogramI64ReducePrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_i64_reduce_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountDirectSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_direct.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_private.comp.spv.h"
    ;
static const uint32_t kHistogramU32I64CountPrivateSharedSpv[] =
#include "taichi/program/vulkan_sort_shaders/histogram_u32_i64_count_private_shared.comp.spv.h"
    ;
static const uint32_t kReduceI32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceI32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceI32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceI32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceI32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceI32SumAtomicSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_atomic.comp.spv.h"
    ;
static const uint32_t kCheckCountI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_i32.comp.spv.h"
    ;
static const uint32_t kCheckCountF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_f32.comp.spv.h"
    ;
static const uint32_t kCheckCountU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_u32.comp.spv.h"
    ;
static const uint32_t kCheckCountU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_u64.comp.spv.h"
    ;
static const uint32_t kCheckCountI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_i64.comp.spv.h"
    ;
static const uint32_t kCheckCountF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/check_count_f64.comp.spv.h"
    ;
static const uint32_t kMetricReduceF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/metric_reduce_f32.comp.spv.h"
    ;
static const uint32_t kSparseAxpyF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_axpy_f32.comp.spv.h"
    ;
static const uint32_t kSparseDiagonalApplyF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_diagonal_apply_f32.comp.spv.h"
    ;
static const uint32_t kSparseDiagonalRefreshF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_diagonal_refresh_f32.comp.spv.h"
    ;
static const uint32_t kSparseBlockCholeskyRefreshF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_block_cholesky_refresh_f32.comp.spv.h"
    ;
static const uint32_t kSparseBlockCholeskyApplyF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_block_cholesky_apply_f32.comp.spv.h"
    ;
static const uint32_t kSparseDotPartialF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_dot_partial_f32.comp.spv.h"
    ;
static const uint32_t kSparseSqrtScalarF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_sqrt_scalar_f32.comp.spv.h"
    ;
static const uint32_t kSparseScalarDivideF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_scalar_divide_f32.comp.spv.h"
    ;
static const uint32_t kSparseCgUpdateF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_cg_update_f32.comp.spv.h"
    ;
static const uint32_t kSparseCgDirectionF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_cg_direction_f32.comp.spv.h"
    ;
static const uint32_t kSparseConvergenceF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_convergence_f32.comp.spv.h"
    ;
static const uint32_t kSparseMinresScalarF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_minres_scalar_f32.comp.spv.h"
    ;
static const uint32_t kSparseMinresVectorStateF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_minres_vector_state_f32.comp.spv.h"
    ;
static const uint32_t kSparseMinresCommitF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_minres_commit_f32.comp.spv.h"
    ;
static const uint32_t kSparseBicgstabScalarF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_bicgstab_scalar_f32.comp.spv.h"
    ;
static const uint32_t kSparseBicgstabDirectionF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_bicgstab_direction_f32.comp.spv.h"
    ;
static const uint32_t kSparseBicgstabIntermediateF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_bicgstab_intermediate_f32.comp.spv.h"
    ;
static const uint32_t kSparseBicgstabCommitF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_bicgstab_commit_f32.comp.spv.h"
    ;
static const uint32_t kSparseBicgstabReconcileF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_bicgstab_reconcile_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresMultiDotPartialF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_multi_dot_partial_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresMultiDotFinalF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_multi_dot_final_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresProjectionF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_projection_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresBasisF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_basis_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresCombineF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_combine_f32.comp.spv.h"
    ;
static const uint32_t kSparseGmresScalarF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_gmres_scalar_f32.comp.spv.h"
    ;
static const uint32_t kSparseTripletPackF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_triplet_pack_f32.comp.spv.h"
    ;
static const uint32_t kSparseTripletPackPackedF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_triplet_pack_packed_f32.comp.spv.h"
    ;
static const uint32_t kSparseSegmentFlagsU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_segment_flags_u64.comp.spv.h"
    ;
static const uint32_t kSparseSegmentScatterU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_segment_scatter_u64.comp.spv.h"
    ;
static const uint32_t kSparseSegmentReduceF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_segment_reduce_f32.comp.spv.h"
    ;
static const uint32_t kSparseCsrFinalizeU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_csr_finalize_u64.comp.spv.h"
    ;
static const uint32_t kSparseAssemblyFinalizeControlSpv[] =
#include "taichi/program/vulkan_sort_shaders/sparse_assembly_finalize_control.comp.spv.h"
    ;
static const uint32_t kCsrSpmvF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/csr_spmv_f32.comp.spv.h"
    ;
static const uint32_t kBsrSpmvF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bsr_spmv_f32.comp.spv.h"
    ;
static const uint32_t kReduceI32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_single.comp.spv.h"
    ;
static const uint32_t kReduceF32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceF32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceF32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceF32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceF32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceF32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_single.comp.spv.h"
    ;
static const uint32_t kReduceU32SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceU32MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_private.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_private.comp.spv.h"
    ;
static const uint32_t kReduceU32SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceU32MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_final.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_final.comp.spv.h"
    ;
static const uint32_t kReduceU32SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceU32MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_single.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_single.comp.spv.h"
    ;
static const uint32_t kReduceU64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceU64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceU64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceU64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceU64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceU64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceI64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceI64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceI64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceI64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceI64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceI64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceF64SumPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_private.comp.spv.h"
    ;
static const uint32_t kReduceF64MinPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_private.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxPrivateSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_private.comp.spv.h"
    ;
static const uint32_t kReduceF64SumFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_final.comp.spv.h"
    ;
static const uint32_t kReduceF64MinFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_final.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxFinalSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_final.comp.spv.h"
    ;
static const uint32_t kReduceF64SumSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_single.comp.spv.h"
    ;
static const uint32_t kReduceF64MinSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_single.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxSingleSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_single.comp.spv.h"
    ;
static const uint32_t kReduceI32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU32MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u32_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceU64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_u64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceI64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_i64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64SumPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MinPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxPrivateStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_private_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64SumSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_sum_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MinSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_min_single_strided.comp.spv.h"
    ;
static const uint32_t kReduceF64MaxSingleStridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/reduce_f64_max_single_strided.comp.spv.h"
    ;
static const uint32_t kTransformI32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_i32_affine.comp.spv.h"
    ;
static const uint32_t kTransformI32AffineDenseSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_i32_affine_dense.comp.spv.h"
    ;
static const uint32_t kTransformF32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_f32_affine.comp.spv.h"
    ;
static const uint32_t kTransformIndexedI32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_indexed_i32_affine.comp.spv.h"
    ;
static const uint32_t kTransformIndexedF32AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_indexed_f32_affine.comp.spv.h"
    ;
static const uint32_t kTransformU64AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_u64_affine.comp.spv.h"
    ;
static const uint32_t kTransformF64AffineSpv[] =
#include "taichi/program/vulkan_sort_shaders/transform_f64_affine.comp.spv.h"
    ;
static const uint32_t kAddMergeI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_i32.comp.spv.h"
    ;
static const uint32_t kAddMergeF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_f32.comp.spv.h"
    ;
static const uint32_t kAddMergeU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_u32.comp.spv.h"
    ;
static const uint32_t kAddMergeU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_u64.comp.spv.h"
    ;
static const uint32_t kAddMergeI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_i64.comp.spv.h"
    ;
static const uint32_t kAddMergeF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/add_merge_f64.comp.spv.h"
    ;
static const uint32_t kGatherU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterDenseU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_dense_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kGatherStridedU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/gather_strided_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterStridedU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_strided_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddF32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddU32ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u32_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddU64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddF64ByI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f64_by_i32.comp.spv.h"
    ;
static const uint32_t kScatterAddI32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddF32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddU32ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u32_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddU64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddI64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddF64ByI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f64_by_i32_strided.comp.spv.h"
    ;
static const uint32_t kScatterAddI32ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i32_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kScatterAddF32ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f32_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kScatterAddU32ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u32_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kScatterAddU64ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_u64_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kScatterAddI64ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_i64_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kScatterAddF64ByI32PackedSpv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_add_f64_by_i32_packed.comp.spv.h"
    ;
static const uint32_t kBucketClearI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_clear_i32.comp.spv.h"
    ;
static const uint32_t kBucketCountI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_count_i32.comp.spv.h"
    ;
static const uint32_t kBucketCountPrivateSharedI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_count_private_shared_i32.comp.spv.h"
    ;
static const uint32_t kBucketPrefixI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_prefix_i32.comp.spv.h"
    ;
static const uint32_t kBucketPrefixChunksI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_prefix_chunks_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_f32.comp.spv.h"
    ;
static const uint32_t kBucketScatterU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_u32.comp.spv.h"
    ;
static const uint32_t kBucketScatterRaw64Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_raw64.comp.spv.h"
    ;
static const uint32_t kBucketScatterRawWordsSpv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_raw_words.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_i32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_f32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_u32.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedRaw64Spv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_raw64.comp.spv.h"
    ;
static const uint32_t kBucketScatterPrivateSharedRawWordsSpv[] =
#include "taichi/program/vulkan_sort_shaders/bucket_scatter_private_shared_raw_words.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroU64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_u64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroI64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_i64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceZeroF64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_zero_f64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU32StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u32_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumU64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_u64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumI64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_i64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceAtomicSumF64StridedSpv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_atomic_sum_f64_strided.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumI32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_i32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumF32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_f32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumU32Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_u32.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumU64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_u64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumI64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_i64.comp.spv.h"
    ;
static const uint32_t kGroupedReduceSumF64Spv[] =
#include "taichi/program/vulkan_sort_shaders/grouped_reduce_sum_f64.comp.spv.h"
    ;

static const uint32_t kRankHistShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift0.comp.spv.h"
    ;
static const uint32_t kRankHistShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift4.comp.spv.h"
    ;
static const uint32_t kRankHistShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift8.comp.spv.h"
    ;
static const uint32_t kRankHistShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift12.comp.spv.h"
    ;
static const uint32_t kRankHistShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift16.comp.spv.h"
    ;
static const uint32_t kRankHistShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift20.comp.spv.h"
    ;
static const uint32_t kRankHistShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift24.comp.spv.h"
    ;
static const uint32_t kRankHistShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_shift28.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift0.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift4.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift8.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift12.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift16.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift20.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift24.comp.spv.h"
    ;
static const uint32_t kRankHistSubgroupShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/rank_hist_subgroup_shift28.comp.spv.h"
    ;

static const uint32_t kScatterKeysShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift0.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift4.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift8.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift12.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift16.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift20.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift24.comp.spv.h"
    ;
static const uint32_t kScatterKeysShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_shift28.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift0.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift4.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift8.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift12.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift16.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift20.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift24.comp.spv.h"
    ;
static const uint32_t kScatterKeysInlineChunksShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_keys_inline_chunks_shift28.comp.spv.h"
    ;

static const uint32_t kScatterPairsShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsRaw64Shift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_raw64_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift0Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift0.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift4Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift4.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift8Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift8.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift12Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift12.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift16Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift16.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift20Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift20.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift24Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift24.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksShift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_shift28.comp.spv.h"
    ;
static const uint32_t kScatterPairsInlineChunksRaw64Shift28Spv[] =
#include "taichi/program/vulkan_sort_shaders/scatter_pairs_inline_chunks_raw64_shift28.comp.spv.h"
    ;

static const uint32_t kRadix8UpsweepSpv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_upsweep.comp.spv.h"
    ;
static const uint32_t kRadix8SpineSpv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_spine.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepKeysSpv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_keys.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsSpv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs.comp.spv.h"
    ;
static const uint32_t kRadix8DownsweepPairsRaw64Spv[] =
#include "taichi/program/vulkan_sort_shaders/radix8_downsweep_pairs_raw64.comp.spv.h"
    ;

std::unique_ptr<Pipeline> create_pipeline_from_spv(Device *device,
                                                   const uint32_t *spv,
                                                   size_t spv_bytes,
                                                   const std::string &name) {
  PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                          spv,
                          spv_bytes,
                          PipelineStageType::compute};
  auto [pipeline, res] = device->create_pipeline_unique(desc, name);
  TI_ERROR_IF(res != RhiResult::success,
              "Failed to create Vulkan sort pipeline '{}': RhiResult({})",
              name, res);
  return std::move(pipeline);
}

template <size_t N>
std::unique_ptr<Pipeline> create_pipeline(Device *device,
                                          const uint32_t (&spv)[N],
                                          const std::string &name) {
  return create_pipeline_from_spv(device, spv, sizeof(spv), name);
}

struct VulkanScanStridedSpvSet {
  const uint32_t *block_spv;
  size_t block_bytes;
  const uint32_t *add_spv;
  size_t add_bytes;
  const uint32_t *small_spv;
  size_t small_bytes;
  const char *dtype_name;
};

struct VulkanScanReverseSpvSet {
  const uint32_t *block_spv;
  size_t block_bytes;
  const uint32_t *block_strided_spv;
  size_t block_strided_bytes;
  const uint32_t *add_spv;
  size_t add_bytes;
  const uint32_t *add_strided_spv;
  size_t add_strided_bytes;
  const uint32_t *small_spv;
  size_t small_bytes;
  const uint32_t *small_strided_spv;
  size_t small_strided_bytes;
  const char *dtype_name;
};

const VulkanScanStridedSpvSet &vulkan_scan_strided_spv_set(int value_type) {
  static const VulkanScanStridedSpvSet sets[] = {
      {kScanI32BlockStridedSpv, sizeof(kScanI32BlockStridedSpv),
       kScanI32AddStridedSpv, sizeof(kScanI32AddStridedSpv),
       kScanI32SmallSubgroupStridedSpv,
       sizeof(kScanI32SmallSubgroupStridedSpv), "i32"},
      {kScanF32BlockStridedSpv, sizeof(kScanF32BlockStridedSpv),
       kScanF32AddStridedSpv, sizeof(kScanF32AddStridedSpv),
       kScanF32SmallSubgroupStridedSpv,
       sizeof(kScanF32SmallSubgroupStridedSpv), "f32"},
      {kScanU32BlockStridedSpv, sizeof(kScanU32BlockStridedSpv),
       kScanU32AddStridedSpv, sizeof(kScanU32AddStridedSpv),
       kScanU32SmallSubgroupStridedSpv,
       sizeof(kScanU32SmallSubgroupStridedSpv), "u32"},
      {kScanU64BlockStridedSpv, sizeof(kScanU64BlockStridedSpv),
       kScanU64AddStridedSpv, sizeof(kScanU64AddStridedSpv), nullptr, 0,
       "u64"},
      {kScanI64BlockStridedSpv, sizeof(kScanI64BlockStridedSpv),
       kScanI64AddStridedSpv, sizeof(kScanI64AddStridedSpv), nullptr, 0,
       "i64"},
      {kScanF64BlockStridedSpv, sizeof(kScanF64BlockStridedSpv),
       kScanF64AddStridedSpv, sizeof(kScanF64AddStridedSpv), nullptr, 0,
       "f64"},
  };
  TI_ERROR_IF(value_type < 0 || value_type >= 6,
              "Unsupported Vulkan strided scan value type.");
  return sets[value_type];
}

const VulkanScanReverseSpvSet &vulkan_scan_reverse_spv_set(int value_type) {
  static const VulkanScanReverseSpvSet sets[] = {
      {kScanI32BlockReverseSpv, sizeof(kScanI32BlockReverseSpv),
       kScanI32BlockStridedReverseSpv,
       sizeof(kScanI32BlockStridedReverseSpv), kScanI32AddReverseSpv,
       sizeof(kScanI32AddReverseSpv), kScanI32AddStridedReverseSpv,
       sizeof(kScanI32AddStridedReverseSpv), kScanI32SmallSubgroupReverseSpv,
       sizeof(kScanI32SmallSubgroupReverseSpv),
       kScanI32SmallSubgroupStridedReverseSpv,
       sizeof(kScanI32SmallSubgroupStridedReverseSpv), "i32"},
      {kScanF32BlockReverseSpv, sizeof(kScanF32BlockReverseSpv),
       kScanF32BlockStridedReverseSpv,
       sizeof(kScanF32BlockStridedReverseSpv), kScanF32AddReverseSpv,
       sizeof(kScanF32AddReverseSpv), kScanF32AddStridedReverseSpv,
       sizeof(kScanF32AddStridedReverseSpv), kScanF32SmallSubgroupReverseSpv,
       sizeof(kScanF32SmallSubgroupReverseSpv),
       kScanF32SmallSubgroupStridedReverseSpv,
       sizeof(kScanF32SmallSubgroupStridedReverseSpv), "f32"},
      {kScanU32BlockReverseSpv, sizeof(kScanU32BlockReverseSpv),
       kScanU32BlockStridedReverseSpv,
       sizeof(kScanU32BlockStridedReverseSpv), kScanU32AddReverseSpv,
       sizeof(kScanU32AddReverseSpv), kScanU32AddStridedReverseSpv,
       sizeof(kScanU32AddStridedReverseSpv), kScanU32SmallSubgroupReverseSpv,
       sizeof(kScanU32SmallSubgroupReverseSpv),
       kScanU32SmallSubgroupStridedReverseSpv,
       sizeof(kScanU32SmallSubgroupStridedReverseSpv), "u32"},
      {kScanU64BlockReverseSpv, sizeof(kScanU64BlockReverseSpv),
       kScanU64BlockStridedReverseSpv,
       sizeof(kScanU64BlockStridedReverseSpv), kScanU64AddReverseSpv,
       sizeof(kScanU64AddReverseSpv), kScanU64AddStridedReverseSpv,
       sizeof(kScanU64AddStridedReverseSpv), nullptr, 0, nullptr, 0, "u64"},
      {kScanI64BlockReverseSpv, sizeof(kScanI64BlockReverseSpv),
       kScanI64BlockStridedReverseSpv,
       sizeof(kScanI64BlockStridedReverseSpv), kScanI64AddReverseSpv,
       sizeof(kScanI64AddReverseSpv), kScanI64AddStridedReverseSpv,
       sizeof(kScanI64AddStridedReverseSpv), nullptr, 0, nullptr, 0, "i64"},
      {kScanF64BlockReverseSpv, sizeof(kScanF64BlockReverseSpv),
       kScanF64BlockStridedReverseSpv,
       sizeof(kScanF64BlockStridedReverseSpv), kScanF64AddReverseSpv,
       sizeof(kScanF64AddReverseSpv), kScanF64AddStridedReverseSpv,
       sizeof(kScanF64AddStridedReverseSpv), nullptr, 0, nullptr, 0, "f64"},
  };
  TI_ERROR_IF(value_type < 0 || value_type >= 6,
              "Unsupported Vulkan reverse scan value type.");
  return sets[value_type];
}

struct VulkanRadixSortCache {
  Device *device{nullptr};
  size_t capacity{0};
  size_t num_blocks{0};
  size_t cached_bytes{0};
  DeviceAllocation key_in{kDeviceNullAllocation};
  DeviceAllocation key_out{kDeviceNullAllocation};
  DeviceAllocation rank{kDeviceNullAllocation};
  DeviceAllocation hist{kDeviceNullAllocation};
  DeviceAllocation offsets{kDeviceNullAllocation};
  DeviceAllocation chunk_sums{kDeviceNullAllocation};
  DeviceAllocation chunk_offsets{kDeviceNullAllocation};
  DeviceAllocation radix8_global_hist{kDeviceNullAllocation};
  DeviceAllocation radix8_partition_hist{kDeviceNullAllocation};
  DeviceAllocation value_in{kDeviceNullAllocation};
  DeviceAllocation value_out{kDeviceNullAllocation};
  DeviceAllocation key_high{kDeviceNullAllocation};
  bool has_value_buffers{false};
  bool has_high_key_buffer{false};
  size_t key_bytes_per_item{sizeof(uint32_t)};
  size_t value_bytes_per_item{0};
  bool workspace_uses_radix8{false};
  bool subgroup_rank_enabled{false};
  bool radix8_enabled{false};
  bool inline_chunk_prefix_allowed{false};

  std::unique_ptr<Pipeline> init_i32;
  std::unique_ptr<Pipeline> copy_i32;
  std::unique_ptr<Pipeline> sort_init_u32_index;
  std::unique_ptr<Pipeline> sort_init_i32_index;
  std::unique_ptr<Pipeline> sort_init_f32_index;
  std::unique_ptr<Pipeline> sort_init_u64_index;
  std::unique_ptr<Pipeline> sort_init_i64_index;
  std::unique_ptr<Pipeline> sort_init_f64_index;
  std::unique_ptr<Pipeline> gather_u32_by_u32;
  std::unique_ptr<Pipeline> prefix_block;
  std::unique_ptr<Pipeline> prefix_chunks;
  std::unique_ptr<Pipeline> prefix_single_chunk;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist;
  std::array<std::unique_ptr<Pipeline>, 8> rank_hist_subgroup;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_keys_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_raw64;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_inline_chunks;
  std::array<std::unique_ptr<Pipeline>, 8> scatter_pairs_inline_chunks_raw64;
  std::unique_ptr<Pipeline> radix8_upsweep;
  std::unique_ptr<Pipeline> radix8_spine;
  std::unique_ptr<Pipeline> radix8_downsweep_keys;
  std::unique_ptr<Pipeline> radix8_downsweep_pairs;
  std::unique_ptr<Pipeline> radix8_downsweep_pairs_raw64;
  std::unique_ptr<ShaderResourceSet> radix8_spine_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4> radix8_upsweep_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4>
      radix8_downsweep_keys_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 4>
      radix8_downsweep_pairs_bindings;
  std::unique_ptr<ShaderResourceSet> init_i32_bindings;
  std::unique_ptr<ShaderResourceSet> copy_i32_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6> sort_init_index_bindings;
  std::unique_ptr<ShaderResourceSet> gather_high32_bindings;
  std::unique_ptr<ShaderResourceSet> gather_keys_bindings;
  std::unique_ptr<ShaderResourceSet> gather_values_bindings;
  std::array<VulkanRwBufferReplay<3>, 4> radix8_upsweep_replay;
  std::array<VulkanRwBufferReplay<4>, 4> radix8_downsweep_keys_replay;
  std::array<VulkanRwBufferReplay<6>, 4> radix8_downsweep_pairs_replay;
  VulkanRwBufferReplay<2> init_i32_replay;
  VulkanRwBufferReplay<2> copy_i32_replay;
  std::array<VulkanRwBufferReplay<4>, 6> sort_init_index_replay;
  VulkanRwBufferReplay<3> gather_high32_replay;
  VulkanRwBufferReplay<3> gather_keys_replay;
  VulkanRwBufferReplay<3> gather_values_replay;
  VulkanCommandReplayCache command_replay;

  void clear_allocs() {
    clear_resource_sets();
    if (!device) {
      return;
    }
    for (DeviceAllocation *alloc :
         {&key_in, &key_out, &rank, &hist, &offsets, &chunk_sums,
          &chunk_offsets, &radix8_global_hist, &radix8_partition_hist,
          &value_in, &value_out, &key_high}) {
      if (*alloc != kDeviceNullAllocation) {
        device->dealloc_memory(*alloc);
        *alloc = kDeviceNullAllocation;
      }
    }
    capacity = 0;
    num_blocks = 0;
    cached_bytes = 0;
    has_value_buffers = false;
    has_high_key_buffer = false;
    key_bytes_per_item = sizeof(uint32_t);
    value_bytes_per_item = 0;
    workspace_uses_radix8 = false;
  }

  void clear_resource_sets() {
    command_replay.reset();
    radix8_spine_bindings.reset();
    for (auto &bindings : radix8_upsweep_bindings) {
      bindings.reset();
    }
    for (auto &bindings : radix8_downsweep_keys_bindings) {
      bindings.reset();
    }
    for (auto &bindings : radix8_downsweep_pairs_bindings) {
      bindings.reset();
    }
    init_i32_bindings.reset();
    copy_i32_bindings.reset();
    for (auto &bindings : sort_init_index_bindings) {
      bindings.reset();
    }
    gather_high32_bindings.reset();
    gather_keys_bindings.reset();
    gather_values_bindings.reset();
    for (auto &replay : radix8_upsweep_replay) {
      replay.reset();
    }
    for (auto &replay : radix8_downsweep_keys_replay) {
      replay.reset();
    }
    for (auto &replay : radix8_downsweep_pairs_replay) {
      replay.reset();
    }
    init_i32_replay.reset();
    copy_i32_replay.reset();
    for (auto &replay : sort_init_index_replay) {
      replay.reset();
    }
    gather_high32_replay.reset();
    gather_keys_replay.reset();
    gather_values_replay.reset();
  }

  void reset_pipelines() {
    init_i32.reset();
    copy_i32.reset();
    sort_init_u32_index.reset();
    sort_init_i32_index.reset();
    sort_init_f32_index.reset();
    sort_init_u64_index.reset();
    sort_init_i64_index.reset();
    sort_init_f64_index.reset();
    gather_u32_by_u32.reset();
    prefix_block.reset();
    prefix_chunks.reset();
    prefix_single_chunk.reset();
    for (auto &pipeline : rank_hist) {
      pipeline.reset();
    }
    for (auto &pipeline : rank_hist_subgroup) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_keys) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_keys_inline_chunks) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_pairs) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_pairs_raw64) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_pairs_inline_chunks) {
      pipeline.reset();
    }
    for (auto &pipeline : scatter_pairs_inline_chunks_raw64) {
      pipeline.reset();
    }
    radix8_upsweep.reset();
    radix8_spine.reset();
    radix8_downsweep_keys.reset();
    radix8_downsweep_pairs.reset();
    radix8_downsweep_pairs_raw64.reset();
    clear_resource_sets();
  }

  ~VulkanRadixSortCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void ensure_device_only(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      reset_pipelines();
    }
    device = dev;
    const bool subgroup_rank_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_ballot) != 0;
    const bool subgroup_arithmetic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_arithmetic) !=
        0;
    const bool subgroup_rank_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_SUBGROUP_RANK", 1) != 0;
    const bool radix8_requested =
        get_environ_config("TI_VULKAN_SORT_ENABLE_RADIX8", 1) != 0;
    subgroup_rank_enabled =
        subgroup_rank_supported && subgroup_rank_allowed;
    radix8_enabled =
        radix8_requested && subgroup_rank_supported &&
        subgroup_arithmetic_supported;
    inline_chunk_prefix_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_INLINE_CHUNK_PREFIX", 1) != 0;
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev && init_i32) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      init_i32.reset();
      copy_i32.reset();
      sort_init_u32_index.reset();
      sort_init_i32_index.reset();
      sort_init_f32_index.reset();
      sort_init_u64_index.reset();
      sort_init_i64_index.reset();
      sort_init_f64_index.reset();
      gather_u32_by_u32.reset();
      prefix_block.reset();
      prefix_chunks.reset();
      prefix_single_chunk.reset();
      for (auto &pipeline : rank_hist) {
        pipeline.reset();
      }
      for (auto &pipeline : rank_hist_subgroup) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_keys) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_keys_inline_chunks) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_raw64) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_inline_chunks) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_pairs_inline_chunks_raw64) {
        pipeline.reset();
      }
      radix8_upsweep.reset();
      radix8_spine.reset();
      radix8_downsweep_keys.reset();
      radix8_downsweep_pairs.reset();
      radix8_downsweep_pairs_raw64.reset();
      clear_resource_sets();
    }
    device = dev;
    const bool subgroup_rank_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_ballot) != 0;
    const bool subgroup_arithmetic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_arithmetic) !=
        0;
    const bool subgroup_rank_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_SUBGROUP_RANK", 1) != 0;
    const bool subgroup_rank_requested =
        subgroup_rank_supported && subgroup_rank_allowed;
    const bool radix8_requested =
        get_environ_config("TI_VULKAN_SORT_ENABLE_RADIX8", 1) != 0;
    const bool radix8_supported =
        subgroup_rank_supported && subgroup_arithmetic_supported;
    inline_chunk_prefix_allowed =
        get_environ_config("TI_VULKAN_SORT_ENABLE_INLINE_CHUNK_PREFIX", 1) != 0;
    if (init_i32) {
      subgroup_rank_enabled =
          subgroup_rank_requested && rank_hist_subgroup[0] != nullptr;
      radix8_enabled = radix8_requested && radix8_spine != nullptr;
      return;
    }
    subgroup_rank_enabled = subgroup_rank_requested;
    radix8_enabled = radix8_requested && radix8_supported;
    init_i32 = create_pipeline(dev, kInitI32Spv, "vulkan_sort_init_i32");
    copy_i32 = create_pipeline(dev, kCopyI32Spv, "vulkan_sort_copy_i32");
    sort_init_u32_index = create_pipeline(
        dev, kSortInitU32IndexSpv, "vulkan_sort_init_u32_index");
    sort_init_i32_index = create_pipeline(
        dev, kSortInitI32IndexSpv, "vulkan_sort_init_i32_index");
    sort_init_f32_index = create_pipeline(
        dev, kSortInitF32IndexSpv, "vulkan_sort_init_f32_index");
    sort_init_u64_index = create_pipeline(
        dev, kSortInitU64IndexSpv, "vulkan_sort_init_u64_index");
    sort_init_i64_index = create_pipeline(
        dev, kSortInitI64IndexSpv, "vulkan_sort_init_i64_index");
    sort_init_f64_index = create_pipeline(
        dev, kSortInitF64IndexSpv, "vulkan_sort_init_f64_index");
    gather_u32_by_u32 =
        create_pipeline(dev, kGatherU32ByU32Spv, "vulkan_sort_gather_u32_by_u32");
    prefix_block =
        create_pipeline(dev, kPrefixBlockSpv, "vulkan_sort_prefix_block");
    prefix_chunks =
        create_pipeline(dev, kPrefixChunksSpv, "vulkan_sort_prefix_chunks");
    prefix_single_chunk = create_pipeline(dev, kPrefixSingleChunkSpv,
                                          "vulkan_sort_prefix_single_chunk");

    const std::array<const uint32_t *, 8> rank_data = {
        kRankHistShift0Spv,  kRankHistShift4Spv,  kRankHistShift8Spv,
        kRankHistShift12Spv, kRankHistShift16Spv, kRankHistShift20Spv,
        kRankHistShift24Spv, kRankHistShift28Spv};
    const std::array<size_t, 8> rank_sizes = {
        sizeof(kRankHistShift0Spv),  sizeof(kRankHistShift4Spv),
        sizeof(kRankHistShift8Spv),  sizeof(kRankHistShift12Spv),
        sizeof(kRankHistShift16Spv), sizeof(kRankHistShift20Spv),
        sizeof(kRankHistShift24Spv), sizeof(kRankHistShift28Spv)};
    const std::array<const uint32_t *, 8> rank_subgroup_data = {
        kRankHistSubgroupShift0Spv,  kRankHistSubgroupShift4Spv,
        kRankHistSubgroupShift8Spv,  kRankHistSubgroupShift12Spv,
        kRankHistSubgroupShift16Spv, kRankHistSubgroupShift20Spv,
        kRankHistSubgroupShift24Spv, kRankHistSubgroupShift28Spv};
    const std::array<size_t, 8> rank_subgroup_sizes = {
        sizeof(kRankHistSubgroupShift0Spv),
        sizeof(kRankHistSubgroupShift4Spv),
        sizeof(kRankHistSubgroupShift8Spv),
        sizeof(kRankHistSubgroupShift12Spv),
        sizeof(kRankHistSubgroupShift16Spv),
        sizeof(kRankHistSubgroupShift20Spv),
        sizeof(kRankHistSubgroupShift24Spv),
        sizeof(kRankHistSubgroupShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_data = {
        kScatterKeysShift0Spv,  kScatterKeysShift4Spv,
        kScatterKeysShift8Spv,  kScatterKeysShift12Spv,
        kScatterKeysShift16Spv, kScatterKeysShift20Spv,
        kScatterKeysShift24Spv, kScatterKeysShift28Spv};
    const std::array<size_t, 8> scatter_key_sizes = {
        sizeof(kScatterKeysShift0Spv),  sizeof(kScatterKeysShift4Spv),
        sizeof(kScatterKeysShift8Spv),  sizeof(kScatterKeysShift12Spv),
        sizeof(kScatterKeysShift16Spv), sizeof(kScatterKeysShift20Spv),
        sizeof(kScatterKeysShift24Spv), sizeof(kScatterKeysShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_inline_data = {
        kScatterKeysInlineChunksShift0Spv,
        kScatterKeysInlineChunksShift4Spv,
        kScatterKeysInlineChunksShift8Spv,
        kScatterKeysInlineChunksShift12Spv,
        kScatterKeysInlineChunksShift16Spv,
        kScatterKeysInlineChunksShift20Spv,
        kScatterKeysInlineChunksShift24Spv,
        kScatterKeysInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_key_inline_sizes = {
        sizeof(kScatterKeysInlineChunksShift0Spv),
        sizeof(kScatterKeysInlineChunksShift4Spv),
        sizeof(kScatterKeysInlineChunksShift8Spv),
        sizeof(kScatterKeysInlineChunksShift12Spv),
        sizeof(kScatterKeysInlineChunksShift16Spv),
        sizeof(kScatterKeysInlineChunksShift20Spv),
        sizeof(kScatterKeysInlineChunksShift24Spv),
        sizeof(kScatterKeysInlineChunksShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_data = {
        kScatterPairsShift0Spv,  kScatterPairsShift4Spv,
        kScatterPairsShift8Spv,  kScatterPairsShift12Spv,
        kScatterPairsShift16Spv, kScatterPairsShift20Spv,
        kScatterPairsShift24Spv, kScatterPairsShift28Spv};
    const std::array<size_t, 8> scatter_pair_sizes = {
        sizeof(kScatterPairsShift0Spv),  sizeof(kScatterPairsShift4Spv),
        sizeof(kScatterPairsShift8Spv),  sizeof(kScatterPairsShift12Spv),
        sizeof(kScatterPairsShift16Spv), sizeof(kScatterPairsShift20Spv),
        sizeof(kScatterPairsShift24Spv), sizeof(kScatterPairsShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_raw64_data = {
        kScatterPairsRaw64Shift0Spv,  kScatterPairsRaw64Shift4Spv,
        kScatterPairsRaw64Shift8Spv,  kScatterPairsRaw64Shift12Spv,
        kScatterPairsRaw64Shift16Spv, kScatterPairsRaw64Shift20Spv,
        kScatterPairsRaw64Shift24Spv, kScatterPairsRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_raw64_sizes = {
        sizeof(kScatterPairsRaw64Shift0Spv),
        sizeof(kScatterPairsRaw64Shift4Spv),
        sizeof(kScatterPairsRaw64Shift8Spv),
        sizeof(kScatterPairsRaw64Shift12Spv),
        sizeof(kScatterPairsRaw64Shift16Spv),
        sizeof(kScatterPairsRaw64Shift20Spv),
        sizeof(kScatterPairsRaw64Shift24Spv),
        sizeof(kScatterPairsRaw64Shift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_inline_data = {
        kScatterPairsInlineChunksShift0Spv,
        kScatterPairsInlineChunksShift4Spv,
        kScatterPairsInlineChunksShift8Spv,
        kScatterPairsInlineChunksShift12Spv,
        kScatterPairsInlineChunksShift16Spv,
        kScatterPairsInlineChunksShift20Spv,
        kScatterPairsInlineChunksShift24Spv,
        kScatterPairsInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_sizes = {
        sizeof(kScatterPairsInlineChunksShift0Spv),
        sizeof(kScatterPairsInlineChunksShift4Spv),
        sizeof(kScatterPairsInlineChunksShift8Spv),
        sizeof(kScatterPairsInlineChunksShift12Spv),
        sizeof(kScatterPairsInlineChunksShift16Spv),
        sizeof(kScatterPairsInlineChunksShift20Spv),
        sizeof(kScatterPairsInlineChunksShift24Spv),
        sizeof(kScatterPairsInlineChunksShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_inline_raw64_data = {
        kScatterPairsInlineChunksRaw64Shift0Spv,
        kScatterPairsInlineChunksRaw64Shift4Spv,
        kScatterPairsInlineChunksRaw64Shift8Spv,
        kScatterPairsInlineChunksRaw64Shift12Spv,
        kScatterPairsInlineChunksRaw64Shift16Spv,
        kScatterPairsInlineChunksRaw64Shift20Spv,
        kScatterPairsInlineChunksRaw64Shift24Spv,
        kScatterPairsInlineChunksRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_raw64_sizes = {
        sizeof(kScatterPairsInlineChunksRaw64Shift0Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift4Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift8Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift12Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift16Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift20Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift24Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift28Spv)};
    if (!radix8_enabled) {
      for (int pass = 0; pass < 8; ++pass) {
        PipelineSourceDesc rank_desc{PipelineSourceType::spirv_binary,
                                     rank_data[pass],
                                     rank_sizes[pass],
                                     PipelineStageType::compute};
        auto [rank_pipeline, rank_res] = dev->create_pipeline_unique(
            rank_desc, fmt::format("vulkan_sort_rank_hist_{}", pass));
        TI_ERROR_IF(rank_res != RhiResult::success,
                    "Failed to create Vulkan sort rank pipeline {}: "
                    "RhiResult({})",
                    pass, rank_res);
        rank_hist[pass] = std::move(rank_pipeline);

        if (subgroup_rank_enabled) {
          PipelineSourceDesc rank_subgroup_desc{
              PipelineSourceType::spirv_binary,
              rank_subgroup_data[pass],
              rank_subgroup_sizes[pass],
              PipelineStageType::compute};
          auto [rank_subgroup_pipeline, rank_subgroup_res] =
              dev->create_pipeline_unique(
                  rank_subgroup_desc,
                  fmt::format("vulkan_sort_rank_hist_subgroup_{}", pass));
          TI_ERROR_IF(rank_subgroup_res != RhiResult::success,
                      "Failed to create Vulkan sort subgroup rank pipeline {}: "
                      "RhiResult({})",
                      pass, rank_subgroup_res);
          rank_hist_subgroup[pass] = std::move(rank_subgroup_pipeline);
        }

        PipelineSourceDesc scatter_key_desc{PipelineSourceType::spirv_binary,
                                            scatter_key_data[pass],
                                            scatter_key_sizes[pass],
                                            PipelineStageType::compute};
        auto [scatter_key_pipeline, scatter_key_res] =
            dev->create_pipeline_unique(
                scatter_key_desc,
                fmt::format("vulkan_sort_scatter_keys_{}", pass));
        TI_ERROR_IF(scatter_key_res != RhiResult::success,
                    "Failed to create Vulkan sort key scatter pipeline {}: "
                    "RhiResult({})",
                    pass, scatter_key_res);
        scatter_keys[pass] = std::move(scatter_key_pipeline);

        PipelineSourceDesc scatter_key_inline_desc{
            PipelineSourceType::spirv_binary,
            scatter_key_inline_data[pass],
            scatter_key_inline_sizes[pass],
            PipelineStageType::compute};
        auto [scatter_key_inline_pipeline, scatter_key_inline_res] =
            dev->create_pipeline_unique(
                scatter_key_inline_desc,
                fmt::format("vulkan_sort_scatter_keys_inline_chunks_{}",
                            pass));
        TI_ERROR_IF(scatter_key_inline_res != RhiResult::success,
                    "Failed to create Vulkan sort inline key scatter pipeline "
                    "{}: RhiResult({})",
                    pass, scatter_key_inline_res);
        scatter_keys_inline_chunks[pass] =
            std::move(scatter_key_inline_pipeline);

        PipelineSourceDesc scatter_pair_desc{PipelineSourceType::spirv_binary,
                                             scatter_pair_data[pass],
                                             scatter_pair_sizes[pass],
                                             PipelineStageType::compute};
        auto [scatter_pair_pipeline, scatter_pair_res] =
            dev->create_pipeline_unique(
                scatter_pair_desc,
                fmt::format("vulkan_sort_scatter_pairs_{}", pass));
        TI_ERROR_IF(scatter_pair_res != RhiResult::success,
                    "Failed to create Vulkan sort pair scatter pipeline {}: "
                    "RhiResult({})",
                    pass, scatter_pair_res);
        scatter_pairs[pass] = std::move(scatter_pair_pipeline);

        PipelineSourceDesc scatter_pair_raw64_desc{
            PipelineSourceType::spirv_binary,
            scatter_pair_raw64_data[pass],
            scatter_pair_raw64_sizes[pass],
            PipelineStageType::compute};
        auto [scatter_pair_raw64_pipeline, scatter_pair_raw64_res] =
            dev->create_pipeline_unique(
                scatter_pair_raw64_desc,
                fmt::format("vulkan_sort_scatter_pairs_raw64_{}", pass));
        TI_ERROR_IF(scatter_pair_raw64_res != RhiResult::success,
                    "Failed to create Vulkan sort raw64 pair scatter pipeline "
                    "{}: RhiResult({})",
                    pass, scatter_pair_raw64_res);
        scatter_pairs_raw64[pass] = std::move(scatter_pair_raw64_pipeline);

        PipelineSourceDesc scatter_pair_inline_desc{
            PipelineSourceType::spirv_binary,
            scatter_pair_inline_data[pass],
            scatter_pair_inline_sizes[pass],
            PipelineStageType::compute};
        auto [scatter_pair_inline_pipeline, scatter_pair_inline_res] =
            dev->create_pipeline_unique(
                scatter_pair_inline_desc,
                fmt::format("vulkan_sort_scatter_pairs_inline_chunks_{}",
                            pass));
        TI_ERROR_IF(scatter_pair_inline_res != RhiResult::success,
                    "Failed to create Vulkan sort inline pair scatter pipeline "
                    "{}: RhiResult({})",
                    pass, scatter_pair_inline_res);
        scatter_pairs_inline_chunks[pass] =
            std::move(scatter_pair_inline_pipeline);

        PipelineSourceDesc scatter_pair_inline_raw64_desc{
            PipelineSourceType::spirv_binary,
            scatter_pair_inline_raw64_data[pass],
            scatter_pair_inline_raw64_sizes[pass],
            PipelineStageType::compute};
        auto [scatter_pair_inline_raw64_pipeline,
              scatter_pair_inline_raw64_res] =
            dev->create_pipeline_unique(
                scatter_pair_inline_raw64_desc,
                fmt::format("vulkan_sort_scatter_pairs_inline_chunks_raw64_{}",
                            pass));
        TI_ERROR_IF(
            scatter_pair_inline_raw64_res != RhiResult::success,
            "Failed to create Vulkan sort inline raw64 pair scatter pipeline "
            "{}: RhiResult({})",
            pass, scatter_pair_inline_raw64_res);
        scatter_pairs_inline_chunks_raw64[pass] =
            std::move(scatter_pair_inline_raw64_pipeline);
      }
    }

    if (radix8_enabled) {
      radix8_spine =
          create_pipeline(dev, kRadix8SpineSpv, "vulkan_sort_radix8_spine");
      radix8_upsweep =
          create_pipeline(dev, kRadix8UpsweepSpv,
                          "vulkan_sort_radix8_upsweep");
      radix8_downsweep_keys =
          create_pipeline(dev, kRadix8DownsweepKeysSpv,
                          "vulkan_sort_radix8_downsweep_keys");
      radix8_downsweep_pairs =
          create_pipeline(dev, kRadix8DownsweepPairsSpv,
                          "vulkan_sort_radix8_downsweep_pairs");
      radix8_downsweep_pairs_raw64 =
          create_pipeline(dev, kRadix8DownsweepPairsRaw64Spv,
                          "vulkan_sort_radix8_downsweep_pairs_raw64");
    }
  }

  Pipeline *sort_init_index_pipeline(int key_type) const {
    switch (key_type) {
      case 0:
        return sort_init_u32_index.get();
      case 1:
        return sort_init_i32_index.get();
      case 2:
        return sort_init_f32_index.get();
      case 3:
        return sort_init_u64_index.get();
      case 4:
        return sort_init_i64_index.get();
      case 5:
        return sort_init_f64_index.get();
      default:
        TI_ERROR("Unsupported Vulkan sort key type.");
    }
    return nullptr;
  }

  void ensure_sort_init_index_pipeline(Device *dev, int key_type) {
    ensure_device_only(dev);
    switch (key_type) {
      case 0:
        if (!sort_init_u32_index) {
          sort_init_u32_index = create_pipeline(
              dev, kSortInitU32IndexSpv, "vulkan_sort_init_u32_index");
        }
        return;
      case 1:
        if (!sort_init_i32_index) {
          sort_init_i32_index = create_pipeline(
              dev, kSortInitI32IndexSpv, "vulkan_sort_init_i32_index");
        }
        return;
      case 2:
        if (!sort_init_f32_index) {
          sort_init_f32_index = create_pipeline(
              dev, kSortInitF32IndexSpv, "vulkan_sort_init_f32_index");
        }
        return;
      case 3:
        if (!sort_init_u64_index) {
          sort_init_u64_index = create_pipeline(
              dev, kSortInitU64IndexSpv, "vulkan_sort_init_u64_index");
        }
        return;
      case 4:
        if (!sort_init_i64_index) {
          sort_init_i64_index = create_pipeline(
              dev, kSortInitI64IndexSpv, "vulkan_sort_init_i64_index");
        }
        return;
      case 5:
        if (!sort_init_f64_index) {
          sort_init_f64_index = create_pipeline(
              dev, kSortInitF64IndexSpv, "vulkan_sort_init_f64_index");
        }
        return;
      default:
        TI_ERROR("Unsupported Vulkan sort key type.");
    }
  }

  void ensure_signed_key_pipelines(Device *dev) {
    ensure_device_only(dev);
    if (!init_i32) {
      init_i32 = create_pipeline(dev, kInitI32Spv, "vulkan_sort_init_i32");
    }
    if (!copy_i32) {
      copy_i32 = create_pipeline(dev, kCopyI32Spv, "vulkan_sort_copy_i32");
    }
  }

  void ensure_gather_pipeline(Device *dev) {
    ensure_device_only(dev);
    if (!gather_u32_by_u32) {
      gather_u32_by_u32 = create_pipeline(
          dev, kGatherU32ByU32Spv, "vulkan_sort_gather_u32_by_u32");
    }
  }

  void ensure_radix4_prefix_pipelines(Device *dev) {
    ensure_device_only(dev);
    if (!prefix_block) {
      prefix_block =
          create_pipeline(dev, kPrefixBlockSpv, "vulkan_sort_prefix_block");
    }
    if (!prefix_chunks) {
      prefix_chunks =
          create_pipeline(dev, kPrefixChunksSpv, "vulkan_sort_prefix_chunks");
    }
    if (!prefix_single_chunk) {
      prefix_single_chunk = create_pipeline(dev, kPrefixSingleChunkSpv,
                                            "vulkan_sort_prefix_single_chunk");
    }
  }

  void ensure_radix4_pipelines(Device *dev,
                               bool use_payload,
                               bool raw64_values,
                               bool inline_chunk_offsets) {
    ensure_device_only(dev);
    ensure_radix4_prefix_pipelines(dev);
    const std::array<const uint32_t *, 8> rank_data = {
        kRankHistShift0Spv,  kRankHistShift4Spv,  kRankHistShift8Spv,
        kRankHistShift12Spv, kRankHistShift16Spv, kRankHistShift20Spv,
        kRankHistShift24Spv, kRankHistShift28Spv};
    const std::array<size_t, 8> rank_sizes = {
        sizeof(kRankHistShift0Spv),  sizeof(kRankHistShift4Spv),
        sizeof(kRankHistShift8Spv),  sizeof(kRankHistShift12Spv),
        sizeof(kRankHistShift16Spv), sizeof(kRankHistShift20Spv),
        sizeof(kRankHistShift24Spv), sizeof(kRankHistShift28Spv)};
    const std::array<const uint32_t *, 8> rank_subgroup_data = {
        kRankHistSubgroupShift0Spv,  kRankHistSubgroupShift4Spv,
        kRankHistSubgroupShift8Spv,  kRankHistSubgroupShift12Spv,
        kRankHistSubgroupShift16Spv, kRankHistSubgroupShift20Spv,
        kRankHistSubgroupShift24Spv, kRankHistSubgroupShift28Spv};
    const std::array<size_t, 8> rank_subgroup_sizes = {
        sizeof(kRankHistSubgroupShift0Spv),
        sizeof(kRankHistSubgroupShift4Spv),
        sizeof(kRankHistSubgroupShift8Spv),
        sizeof(kRankHistSubgroupShift12Spv),
        sizeof(kRankHistSubgroupShift16Spv),
        sizeof(kRankHistSubgroupShift20Spv),
        sizeof(kRankHistSubgroupShift24Spv),
        sizeof(kRankHistSubgroupShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_data = {
        kScatterKeysShift0Spv,  kScatterKeysShift4Spv,
        kScatterKeysShift8Spv,  kScatterKeysShift12Spv,
        kScatterKeysShift16Spv, kScatterKeysShift20Spv,
        kScatterKeysShift24Spv, kScatterKeysShift28Spv};
    const std::array<size_t, 8> scatter_key_sizes = {
        sizeof(kScatterKeysShift0Spv),  sizeof(kScatterKeysShift4Spv),
        sizeof(kScatterKeysShift8Spv),  sizeof(kScatterKeysShift12Spv),
        sizeof(kScatterKeysShift16Spv), sizeof(kScatterKeysShift20Spv),
        sizeof(kScatterKeysShift24Spv), sizeof(kScatterKeysShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_key_inline_data = {
        kScatterKeysInlineChunksShift0Spv,
        kScatterKeysInlineChunksShift4Spv,
        kScatterKeysInlineChunksShift8Spv,
        kScatterKeysInlineChunksShift12Spv,
        kScatterKeysInlineChunksShift16Spv,
        kScatterKeysInlineChunksShift20Spv,
        kScatterKeysInlineChunksShift24Spv,
        kScatterKeysInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_key_inline_sizes = {
        sizeof(kScatterKeysInlineChunksShift0Spv),
        sizeof(kScatterKeysInlineChunksShift4Spv),
        sizeof(kScatterKeysInlineChunksShift8Spv),
        sizeof(kScatterKeysInlineChunksShift12Spv),
        sizeof(kScatterKeysInlineChunksShift16Spv),
        sizeof(kScatterKeysInlineChunksShift20Spv),
        sizeof(kScatterKeysInlineChunksShift24Spv),
        sizeof(kScatterKeysInlineChunksShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_data = {
        kScatterPairsShift0Spv,  kScatterPairsShift4Spv,
        kScatterPairsShift8Spv,  kScatterPairsShift12Spv,
        kScatterPairsShift16Spv, kScatterPairsShift20Spv,
        kScatterPairsShift24Spv, kScatterPairsShift28Spv};
    const std::array<size_t, 8> scatter_pair_sizes = {
        sizeof(kScatterPairsShift0Spv),  sizeof(kScatterPairsShift4Spv),
        sizeof(kScatterPairsShift8Spv),  sizeof(kScatterPairsShift12Spv),
        sizeof(kScatterPairsShift16Spv), sizeof(kScatterPairsShift20Spv),
        sizeof(kScatterPairsShift24Spv), sizeof(kScatterPairsShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_raw64_data = {
        kScatterPairsRaw64Shift0Spv,  kScatterPairsRaw64Shift4Spv,
        kScatterPairsRaw64Shift8Spv,  kScatterPairsRaw64Shift12Spv,
        kScatterPairsRaw64Shift16Spv, kScatterPairsRaw64Shift20Spv,
        kScatterPairsRaw64Shift24Spv, kScatterPairsRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_raw64_sizes = {
        sizeof(kScatterPairsRaw64Shift0Spv),
        sizeof(kScatterPairsRaw64Shift4Spv),
        sizeof(kScatterPairsRaw64Shift8Spv),
        sizeof(kScatterPairsRaw64Shift12Spv),
        sizeof(kScatterPairsRaw64Shift16Spv),
        sizeof(kScatterPairsRaw64Shift20Spv),
        sizeof(kScatterPairsRaw64Shift24Spv),
        sizeof(kScatterPairsRaw64Shift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_inline_data = {
        kScatterPairsInlineChunksShift0Spv,
        kScatterPairsInlineChunksShift4Spv,
        kScatterPairsInlineChunksShift8Spv,
        kScatterPairsInlineChunksShift12Spv,
        kScatterPairsInlineChunksShift16Spv,
        kScatterPairsInlineChunksShift20Spv,
        kScatterPairsInlineChunksShift24Spv,
        kScatterPairsInlineChunksShift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_sizes = {
        sizeof(kScatterPairsInlineChunksShift0Spv),
        sizeof(kScatterPairsInlineChunksShift4Spv),
        sizeof(kScatterPairsInlineChunksShift8Spv),
        sizeof(kScatterPairsInlineChunksShift12Spv),
        sizeof(kScatterPairsInlineChunksShift16Spv),
        sizeof(kScatterPairsInlineChunksShift20Spv),
        sizeof(kScatterPairsInlineChunksShift24Spv),
        sizeof(kScatterPairsInlineChunksShift28Spv)};
    const std::array<const uint32_t *, 8> scatter_pair_inline_raw64_data = {
        kScatterPairsInlineChunksRaw64Shift0Spv,
        kScatterPairsInlineChunksRaw64Shift4Spv,
        kScatterPairsInlineChunksRaw64Shift8Spv,
        kScatterPairsInlineChunksRaw64Shift12Spv,
        kScatterPairsInlineChunksRaw64Shift16Spv,
        kScatterPairsInlineChunksRaw64Shift20Spv,
        kScatterPairsInlineChunksRaw64Shift24Spv,
        kScatterPairsInlineChunksRaw64Shift28Spv};
    const std::array<size_t, 8> scatter_pair_inline_raw64_sizes = {
        sizeof(kScatterPairsInlineChunksRaw64Shift0Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift4Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift8Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift12Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift16Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift20Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift24Spv),
        sizeof(kScatterPairsInlineChunksRaw64Shift28Spv)};
    for (int pass = 0; pass < 8; ++pass) {
      if (subgroup_rank_enabled) {
        if (!rank_hist_subgroup[pass]) {
          PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                  rank_subgroup_data[pass],
                                  rank_subgroup_sizes[pass],
                                  PipelineStageType::compute};
          auto [pipeline, res] = dev->create_pipeline_unique(
              desc, fmt::format("vulkan_sort_rank_hist_subgroup_{}", pass));
          TI_ERROR_IF(res != RhiResult::success,
                      "Failed to create Vulkan sort subgroup rank pipeline {}: "
                      "RhiResult({})",
                      pass, res);
          rank_hist_subgroup[pass] = std::move(pipeline);
        }
      } else if (!rank_hist[pass]) {
        PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                rank_data[pass], rank_sizes[pass],
                                PipelineStageType::compute};
        auto [pipeline, res] = dev->create_pipeline_unique(
            desc, fmt::format("vulkan_sort_rank_hist_{}", pass));
        TI_ERROR_IF(res != RhiResult::success,
                    "Failed to create Vulkan sort rank pipeline {}: "
                    "RhiResult({})",
                    pass, res);
        rank_hist[pass] = std::move(pipeline);
      }

      if (!use_payload) {
        if (inline_chunk_offsets) {
          if (!scatter_keys_inline_chunks[pass]) {
            PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                    scatter_key_inline_data[pass],
                                    scatter_key_inline_sizes[pass],
                                    PipelineStageType::compute};
            auto [pipeline, res] = dev->create_pipeline_unique(
                desc, fmt::format("vulkan_sort_scatter_keys_inline_chunks_{}",
                                  pass));
            TI_ERROR_IF(res != RhiResult::success,
                        "Failed to create Vulkan sort inline key scatter "
                        "pipeline {}: RhiResult({})",
                        pass, res);
            scatter_keys_inline_chunks[pass] = std::move(pipeline);
          }
        } else if (!scatter_keys[pass]) {
          PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                  scatter_key_data[pass],
                                  scatter_key_sizes[pass],
                                  PipelineStageType::compute};
          auto [pipeline, res] = dev->create_pipeline_unique(
              desc, fmt::format("vulkan_sort_scatter_keys_{}", pass));
          TI_ERROR_IF(res != RhiResult::success,
                      "Failed to create Vulkan sort key scatter pipeline {}: "
                      "RhiResult({})",
                      pass, res);
          scatter_keys[pass] = std::move(pipeline);
        }
        continue;
      }

      if (raw64_values) {
        if (inline_chunk_offsets) {
          if (!scatter_pairs_inline_chunks_raw64[pass]) {
            PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                    scatter_pair_inline_raw64_data[pass],
                                    scatter_pair_inline_raw64_sizes[pass],
                                    PipelineStageType::compute};
            auto [pipeline, res] = dev->create_pipeline_unique(
                desc, fmt::format(
                          "vulkan_sort_scatter_pairs_inline_chunks_raw64_{}",
                          pass));
            TI_ERROR_IF(
                res != RhiResult::success,
                "Failed to create Vulkan sort inline raw64 pair scatter "
                "pipeline {}: RhiResult({})",
                pass, res);
            scatter_pairs_inline_chunks_raw64[pass] = std::move(pipeline);
          }
        } else if (!scatter_pairs_raw64[pass]) {
          PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                  scatter_pair_raw64_data[pass],
                                  scatter_pair_raw64_sizes[pass],
                                  PipelineStageType::compute};
          auto [pipeline, res] = dev->create_pipeline_unique(
              desc, fmt::format("vulkan_sort_scatter_pairs_raw64_{}", pass));
          TI_ERROR_IF(res != RhiResult::success,
                      "Failed to create Vulkan sort raw64 pair scatter "
                      "pipeline {}: RhiResult({})",
                      pass, res);
          scatter_pairs_raw64[pass] = std::move(pipeline);
        }
      } else if (inline_chunk_offsets) {
        if (!scatter_pairs_inline_chunks[pass]) {
          PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                  scatter_pair_inline_data[pass],
                                  scatter_pair_inline_sizes[pass],
                                  PipelineStageType::compute};
          auto [pipeline, res] = dev->create_pipeline_unique(
              desc, fmt::format("vulkan_sort_scatter_pairs_inline_chunks_{}",
                                pass));
          TI_ERROR_IF(res != RhiResult::success,
                      "Failed to create Vulkan sort inline pair scatter "
                      "pipeline {}: RhiResult({})",
                      pass, res);
          scatter_pairs_inline_chunks[pass] = std::move(pipeline);
        }
      } else if (!scatter_pairs[pass]) {
        PipelineSourceDesc desc{PipelineSourceType::spirv_binary,
                                scatter_pair_data[pass],
                                scatter_pair_sizes[pass],
                                PipelineStageType::compute};
        auto [pipeline, res] = dev->create_pipeline_unique(
            desc, fmt::format("vulkan_sort_scatter_pairs_{}", pass));
        TI_ERROR_IF(res != RhiResult::success,
                    "Failed to create Vulkan sort pair scatter pipeline {}: "
                    "RhiResult({})",
                    pass, res);
        scatter_pairs[pass] = std::move(pipeline);
      }
    }
  }

  void ensure_radix8_pipelines(Device *dev,
                               bool use_payload,
                               bool raw64_values) {
    ensure_device_only(dev);
    TI_ERROR_IF(!radix8_enabled,
                "Vulkan radix8 sort pipelines requested on an unsupported "
                "device.");
    if (!radix8_spine) {
      radix8_spine =
          create_pipeline(dev, kRadix8SpineSpv, "vulkan_sort_radix8_spine");
    }
    if (!radix8_upsweep) {
      radix8_upsweep =
          create_pipeline(dev, kRadix8UpsweepSpv,
                          "vulkan_sort_radix8_upsweep");
    }
    if (!use_payload) {
      if (!radix8_downsweep_keys) {
        radix8_downsweep_keys =
            create_pipeline(dev, kRadix8DownsweepKeysSpv,
                            "vulkan_sort_radix8_downsweep_keys");
      }
    } else if (raw64_values) {
      if (!radix8_downsweep_pairs_raw64) {
        radix8_downsweep_pairs_raw64 =
            create_pipeline(dev, kRadix8DownsweepPairsRaw64Spv,
                            "vulkan_sort_radix8_downsweep_pairs_raw64");
      }
    } else if (!radix8_downsweep_pairs) {
      radix8_downsweep_pairs =
          create_pipeline(dev, kRadix8DownsweepPairsSpv,
                          "vulkan_sort_radix8_downsweep_pairs");
    }
  }

  void ensure_sort_pipelines(Device *dev,
                             int key_type,
                             bool use_values,
                             bool raw64_values,
                             bool use_index_sort,
                             bool use_radix8,
                             bool inline_chunk_offsets) {
    ensure_device_only(dev);
    TI_ERROR_IF(use_radix8 && !radix8_enabled,
                "Vulkan radix8 sort was requested but is unavailable.");
    if (use_index_sort) {
      ensure_sort_init_index_pipeline(dev, key_type);
      ensure_gather_pipeline(dev);
    } else if (key_type == 1) {
      ensure_signed_key_pipelines(dev);
    }
    const bool radix_payload = use_values || use_index_sort;
    const bool radix_raw64_values = !use_index_sort && raw64_values;
    if (use_radix8) {
      ensure_radix8_pipelines(dev, radix_payload, radix_raw64_values);
    } else {
      ensure_radix4_pipelines(dev, radix_payload, radix_raw64_values,
                              inline_chunk_offsets);
    }
  }


  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc;
    RhiResult res = device->allocate_memory(
        {bytes, /*host_write=*/false, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage},
        &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan sort workspace: RhiResult({})",
                res);
    return alloc;
  }

  ShaderResourceSet *cached_resource_set(
      std::unique_ptr<ShaderResourceSet> &bindings,
      VulkanSortCpuProfileSample *profile = nullptr) {
    double start = profile ? profile_time_us() : 0.0;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
      if (profile) {
        profile->resource_set_create_calls++;
      }
    }
    if (profile) {
      profile->resource_set_calls++;
      profile->resource_set_us += profile_time_us() - start;
    }
    return bindings.get();
  }

  bool has_workspace_allocs() const {
    return key_in != kDeviceNullAllocation;
  }

  bool needs_workspace_realloc(size_t n,
                               bool use_values,
                               bool use_radix8,
                               size_t value_size,
                               size_t key_item_size = sizeof(uint32_t),
                               bool use_high_keys = false) const {
    const size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    return !(capacity >= n && num_blocks >= requested_units &&
             has_value_buffers >= use_values &&
             has_high_key_buffer >= use_high_keys &&
             workspace_uses_radix8 == use_radix8 &&
             key_bytes_per_item >= key_item_size &&
             (!use_values || value_bytes_per_item >= value_size));
  }

  void ensure_workspace(size_t n,
                        bool use_values,
                        bool use_radix8,
                        size_t value_size,
                        size_t key_item_size = sizeof(uint32_t),
                        bool use_high_keys = false) {
    size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    size_t requested_partitions =
        (n + kRadix8PartitionSize - 1) / kRadix8PartitionSize;
    const size_t requested_units =
        use_radix8 ? requested_partitions : requested_blocks;
    if (!needs_workspace_realloc(n, use_values, use_radix8, value_size,
                                 key_item_size, use_high_keys)) {
      return;
    }
    clear_resource_sets();
    clear_allocs();
    capacity = n;
    num_blocks = requested_units;
    has_value_buffers = use_values;
    has_high_key_buffer = use_high_keys;
    key_bytes_per_item = key_item_size;
    value_bytes_per_item = use_values ? value_size : 0;
    workspace_uses_radix8 = use_radix8;
    const size_t sort_key_bytes = n * sizeof(uint32_t);
    const size_t key_storage_bytes = n * key_item_size;
    key_in = alloc_storage(key_storage_bytes);
    key_out = alloc_storage(key_storage_bytes);
    cached_bytes = key_storage_bytes * 2;
    if (use_high_keys) {
      key_high = alloc_storage(sort_key_bytes);
      cached_bytes += sort_key_bytes;
    }
    if (use_radix8) {
      const size_t global_hist_bytes = kRadix8Bins * sizeof(uint32_t);
      const size_t partition_hist_bytes =
          requested_partitions * kRadix8Bins * sizeof(uint32_t);
      radix8_global_hist = alloc_storage(global_hist_bytes);
      radix8_partition_hist = alloc_storage(partition_hist_bytes);
      cached_bytes += global_hist_bytes + partition_hist_bytes;
    } else {
      const size_t rank_bytes = n * sizeof(uint32_t);
      const size_t table_bytes =
          requested_blocks * kRadixBins * sizeof(uint32_t);
      const size_t requested_chunks =
          (requested_blocks + kBlockSize - 1) / kBlockSize;
      const size_t chunk_table_bytes =
          requested_chunks * kRadixBins * sizeof(uint32_t);
      rank = alloc_storage(rank_bytes);
      hist = alloc_storage(table_bytes);
      offsets = alloc_storage(table_bytes);
      chunk_sums = alloc_storage(chunk_table_bytes);
      chunk_offsets = alloc_storage(chunk_table_bytes);
      cached_bytes +=
          rank_bytes + table_bytes * 2 + chunk_table_bytes * 2;
    }
    if (use_values) {
      const size_t value_bytes = n * value_size;
      value_in = alloc_storage(value_bytes);
      value_out = alloc_storage(value_bytes);
      cached_bytes += value_bytes * 2;
    }
  }
};

struct VulkanScanCache {
  Device *device{nullptr};
  size_t capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation workspace{kDeviceNullAllocation};
  DeviceAllocation dummy_sums{kDeviceNullAllocation};
  DeviceAllocation params{kDeviceNullAllocation};
  size_t dummy_sums_capacity{0};
  size_t params_capacity{0};
  std::unique_ptr<Pipeline> scan_i32_block;
  std::unique_ptr<Pipeline> scan_i32_block_subgroup;
  std::unique_ptr<Pipeline> scan_i32_add;
  std::unique_ptr<Pipeline> scan_i32_small_subgroup;
  std::unique_ptr<Pipeline> scan_f32_block;
  std::unique_ptr<Pipeline> scan_f32_block_subgroup;
  std::unique_ptr<Pipeline> scan_f32_add;
  std::unique_ptr<Pipeline> scan_f32_small_subgroup;
  std::unique_ptr<Pipeline> scan_u32_block;
  std::unique_ptr<Pipeline> scan_u32_block_subgroup;
  std::unique_ptr<Pipeline> scan_u32_add;
  std::unique_ptr<Pipeline> scan_u32_small_subgroup;
  std::unique_ptr<Pipeline> scan_u64_block;
  std::unique_ptr<Pipeline> scan_u64_add;
  std::unique_ptr<Pipeline> scan_i64_block;
  std::unique_ptr<Pipeline> scan_i64_add;
  std::unique_ptr<Pipeline> scan_f64_block;
  std::unique_ptr<Pipeline> scan_f64_add;
  std::array<std::unique_ptr<Pipeline>, 6> scan_block_strided;
  std::array<std::unique_ptr<Pipeline>, 6> scan_add_strided;
  std::array<std::unique_ptr<Pipeline>, 3> scan_small_strided;
  std::array<std::unique_ptr<Pipeline>, 6> scan_block_reverse;
  std::array<std::unique_ptr<Pipeline>, 6> scan_add_reverse;
  std::array<std::unique_ptr<Pipeline>, 3> scan_small_reverse;
  std::array<std::unique_ptr<Pipeline>, 6> scan_block_strided_reverse;
  std::array<std::unique_ptr<Pipeline>, 6> scan_add_strided_reverse;
  std::array<std::unique_ptr<Pipeline>, 3> scan_small_strided_reverse;
  bool subgroup_scan_enabled{false};
  VulkanResourceSetRing scan_small_bindings;
  VulkanResourceSetRing scan_small_strided_bindings;
  VulkanResourceSetRing scan_block_bindings;
  VulkanResourceSetRing scan_block_strided_bindings_ring;
  VulkanResourceSetRing scan_add_bindings;
  VulkanResourceSetRing scan_add_strided_bindings_ring;
  VulkanCommandReplayCache scan_command_replay;

  void reset_resource_sets() {
    scan_command_replay.reset();
    scan_small_bindings.reset();
    scan_small_strided_bindings.reset();
    scan_block_bindings.reset();
    scan_block_strided_bindings_ring.reset();
    scan_add_bindings.reset();
    scan_add_strided_bindings_ring.reset();
  }

  void reset_pipelines() {
    scan_i32_block.reset();
    scan_i32_block_subgroup.reset();
    scan_i32_add.reset();
    scan_i32_small_subgroup.reset();
    scan_f32_block.reset();
    scan_f32_block_subgroup.reset();
    scan_f32_add.reset();
    scan_f32_small_subgroup.reset();
    scan_u32_block.reset();
    scan_u32_block_subgroup.reset();
    scan_u32_add.reset();
    scan_u32_small_subgroup.reset();
    scan_u64_block.reset();
    scan_u64_add.reset();
    scan_i64_block.reset();
    scan_i64_add.reset();
    scan_f64_block.reset();
    scan_f64_add.reset();
    for (auto &pipeline : scan_block_strided) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_add_strided) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_small_strided) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_block_reverse) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_add_reverse) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_small_reverse) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_block_strided_reverse) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_add_strided_reverse) {
      pipeline.reset();
    }
    for (auto &pipeline : scan_small_strided_reverse) {
      pipeline.reset();
    }
    reset_resource_sets();
  }

  void clear_allocs() {
    reset_resource_sets();
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    if (device && dummy_sums != kDeviceNullAllocation) {
      device->dealloc_memory(dummy_sums);
    }
    if (device && params != kDeviceNullAllocation) {
      device->dealloc_memory(params);
    }
    workspace = kDeviceNullAllocation;
    dummy_sums = kDeviceNullAllocation;
    params = kDeviceNullAllocation;
    dummy_sums_capacity = 0;
    params_capacity = 0;
    capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanScanCache() {
    clear_allocs();
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      reset_pipelines();
    }
    device = dev;
    const bool subgroup_arithmetic_supported =
        dev->get_caps().get(DeviceCapability::spirv_has_subgroup_arithmetic) !=
        0;
    const bool subgroup_scan_allowed =
        get_environ_config("TI_VULKAN_SCAN_ENABLE_SUBGROUP", 0) != 0;
    subgroup_scan_enabled =
        subgroup_arithmetic_supported && subgroup_scan_allowed;
  }

  void ensure_scan_add_pipeline(Device *dev, int value_type) {
    ensure_device(dev);
    switch (value_type) {
      case 0:
        if (!scan_i32_add) {
          scan_i32_add =
              create_pipeline(dev, kScanI32AddSpv, "vulkan_scan_i32_add");
        }
        return;
      case 1:
        if (!scan_f32_add) {
          scan_f32_add =
              create_pipeline(dev, kScanF32AddSpv, "vulkan_scan_f32_add");
        }
        return;
      case 2:
        if (!scan_u32_add) {
          scan_u32_add =
              create_pipeline(dev, kScanU32AddSpv, "vulkan_scan_u32_add");
        }
        return;
      case 3:
        if (!scan_u64_add) {
          scan_u64_add =
              create_pipeline(dev, kScanU64AddSpv, "vulkan_scan_u64_add");
        }
        return;
      case 4:
        if (!scan_i64_add) {
          scan_i64_add =
              create_pipeline(dev, kScanI64AddSpv, "vulkan_scan_i64_add");
        }
        return;
      case 5:
        if (!scan_f64_add) {
          scan_f64_add =
              create_pipeline(dev, kScanF64AddSpv, "vulkan_scan_f64_add");
        }
        return;
      default:
        TI_ERROR("Unsupported Vulkan scan value type.");
    }
  }

  void ensure_scan_block_pipeline(Device *dev, int value_type, bool subgroup) {
    ensure_device(dev);
    if (subgroup) {
      TI_ERROR_IF(!subgroup_scan_enabled || value_type < 0 || value_type > 2,
                  "Unsupported Vulkan subgroup scan value type.");
    }
    switch (value_type) {
      case 0:
        if (subgroup) {
          if (!scan_i32_block_subgroup) {
            scan_i32_block_subgroup =
                create_pipeline(dev, kScanI32BlockSubgroupSpv,
                                "vulkan_scan_i32_block_subgroup");
          }
        } else if (!scan_i32_block) {
          scan_i32_block =
              create_pipeline(dev, kScanI32BlockSpv, "vulkan_scan_i32_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      case 1:
        if (subgroup) {
          if (!scan_f32_block_subgroup) {
            scan_f32_block_subgroup =
                create_pipeline(dev, kScanF32BlockSubgroupSpv,
                                "vulkan_scan_f32_block_subgroup");
          }
        } else if (!scan_f32_block) {
          scan_f32_block =
              create_pipeline(dev, kScanF32BlockSpv, "vulkan_scan_f32_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      case 2:
        if (subgroup) {
          if (!scan_u32_block_subgroup) {
            scan_u32_block_subgroup =
                create_pipeline(dev, kScanU32BlockSubgroupSpv,
                                "vulkan_scan_u32_block_subgroup");
          }
        } else if (!scan_u32_block) {
          scan_u32_block =
              create_pipeline(dev, kScanU32BlockSpv, "vulkan_scan_u32_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      case 3:
        TI_ERROR_IF(subgroup, "u64 subgroup scan is not supported.");
        if (!scan_u64_block) {
          scan_u64_block =
              create_pipeline(dev, kScanU64BlockSpv, "vulkan_scan_u64_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      case 4:
        TI_ERROR_IF(subgroup, "i64 subgroup scan is not supported.");
        if (!scan_i64_block) {
          scan_i64_block =
              create_pipeline(dev, kScanI64BlockSpv, "vulkan_scan_i64_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      case 5:
        TI_ERROR_IF(subgroup, "f64 subgroup scan is not supported.");
        if (!scan_f64_block) {
          scan_f64_block =
              create_pipeline(dev, kScanF64BlockSpv, "vulkan_scan_f64_block");
        }
        ensure_scan_add_pipeline(dev, value_type);
        return;
      default:
        TI_ERROR("Unsupported Vulkan scan value type.");
    }
  }

  void ensure_scan_small_pipeline(Device *dev, int value_type) {
    ensure_device(dev);
    TI_ERROR_IF(!subgroup_scan_enabled || value_type < 0 || value_type > 2,
                "Unsupported Vulkan small subgroup scan value type.");
    switch (value_type) {
      case 0:
        if (!scan_i32_small_subgroup) {
          scan_i32_small_subgroup =
              create_pipeline(dev, kScanI32SmallSubgroupSpv,
                              "vulkan_scan_i32_small_subgroup");
        }
        return;
      case 1:
        if (!scan_f32_small_subgroup) {
          scan_f32_small_subgroup =
              create_pipeline(dev, kScanF32SmallSubgroupSpv,
                              "vulkan_scan_f32_small_subgroup");
        }
        return;
      case 2:
        if (!scan_u32_small_subgroup) {
          scan_u32_small_subgroup =
              create_pipeline(dev, kScanU32SmallSubgroupSpv,
                              "vulkan_scan_u32_small_subgroup");
        }
        return;
      default:
        TI_ERROR("Unsupported Vulkan small subgroup scan value type.");
    }
  }

  void ensure_value_pipelines(Device *dev, int value_type) {
    ensure_scan_block_pipeline(dev, value_type, false);
    if (subgroup_scan_enabled && value_type >= 0 && value_type <= 2) {
      ensure_scan_block_pipeline(dev, value_type, true);
      ensure_scan_small_pipeline(dev, value_type);
    }
  }

  void ensure_pipelines(Device *dev) {
    ensure_value_pipelines(dev, 0);
    ensure_value_pipelines(dev, 1);
    ensure_value_pipelines(dev, 2);
    if (dev->get_caps().get(DeviceCapability::spirv_has_int64) != 0) {
      ensure_value_pipelines(dev, 3);
      ensure_value_pipelines(dev, 4);
    }
    if (dev->get_caps().get(DeviceCapability::spirv_has_float64) != 0) {
      ensure_value_pipelines(dev, 5);
    }
  }

  void prepare_resource_sets(Program *program,
                             size_t small_count,
                             size_t small_strided_count,
                             size_t block_count,
                             size_t block_strided_count,
                             size_t add_count,
                             size_t add_strided_count) {
    prepare_resource_set_replay(program, device, scan_small_bindings,
                                small_count);
    prepare_resource_set_replay(program, device, scan_small_strided_bindings,
                                small_strided_count);
    prepare_resource_set_replay(program, device, scan_block_bindings,
                                block_count);
    prepare_resource_set_replay(program, device,
                                scan_block_strided_bindings_ring,
                                block_strided_count);
    prepare_resource_set_replay(program, device, scan_add_bindings, add_count);
    prepare_resource_set_replay(program, device, scan_add_strided_bindings_ring,
                                add_strided_count);
  }

  ShaderResourceSet *next_small_resource_set(bool strided) {
    return (strided ? scan_small_strided_bindings : scan_small_bindings)
        .next(device);
  }

  ShaderResourceSet *next_block_resource_set(bool strided) {
    return (strided ? scan_block_strided_bindings_ring : scan_block_bindings)
        .next(device);
  }

  ShaderResourceSet *next_add_resource_set(bool strided) {
    return (strided ? scan_add_strided_bindings_ring : scan_add_bindings)
        .next(device);
  }

  void ensure_scan_strided_block_add_pipelines(Device *dev, int value_type) {
    ensure_device(dev);
    TI_ERROR_IF(value_type < 0 || value_type >= 6,
                "Unsupported Vulkan strided scan value type.");
    const auto &spv = vulkan_scan_strided_spv_set(value_type);
    if (!scan_block_strided[value_type]) {
      scan_block_strided[value_type] = create_pipeline_from_spv(
          dev, spv.block_spv, spv.block_bytes,
          fmt::format("vulkan_scan_{}_block_strided", spv.dtype_name));
    }
    if (!scan_add_strided[value_type]) {
      scan_add_strided[value_type] = create_pipeline_from_spv(
          dev, spv.add_spv, spv.add_bytes,
          fmt::format("vulkan_scan_{}_add_strided", spv.dtype_name));
    }
  }

  void ensure_scan_strided_small_pipeline(Device *dev, int value_type) {
    ensure_device(dev);
    TI_ERROR_IF(!subgroup_scan_enabled || value_type < 0 || value_type > 2,
                "Unsupported Vulkan strided small subgroup scan value type.");
    if (scan_small_strided[value_type]) {
      return;
    }
    const auto &spv = vulkan_scan_strided_spv_set(value_type);
    TI_ERROR_IF(!spv.small_spv,
                "Missing Vulkan strided small subgroup scan shader.");
    scan_small_strided[value_type] = create_pipeline_from_spv(
        dev, spv.small_spv, spv.small_bytes,
        fmt::format("vulkan_scan_{}_small_subgroup_strided", spv.dtype_name));
  }

  void ensure_scan_reverse_block_add_pipelines(Device *dev,
                                               int value_type,
                                               bool strided) {
    ensure_device(dev);
    TI_ERROR_IF(value_type < 0 || value_type >= 6,
                "Unsupported Vulkan reverse scan value type.");
    const auto &spv = vulkan_scan_reverse_spv_set(value_type);
    auto &block_pipeline =
        strided ? scan_block_strided_reverse[value_type]
                : scan_block_reverse[value_type];
    auto &add_pipeline = strided ? scan_add_strided_reverse[value_type]
                                 : scan_add_reverse[value_type];
    if (!block_pipeline) {
      block_pipeline = create_pipeline_from_spv(
          dev, strided ? spv.block_strided_spv : spv.block_spv,
          strided ? spv.block_strided_bytes : spv.block_bytes,
          fmt::format("vulkan_scan_{}_block{}_reverse", spv.dtype_name,
                      strided ? "_strided" : ""));
    }
    if (!add_pipeline) {
      add_pipeline = create_pipeline_from_spv(
          dev, strided ? spv.add_strided_spv : spv.add_spv,
          strided ? spv.add_strided_bytes : spv.add_bytes,
          fmt::format("vulkan_scan_{}_add{}_reverse", spv.dtype_name,
                      strided ? "_strided" : ""));
    }
  }

  void ensure_scan_reverse_small_pipeline(Device *dev,
                                          int value_type,
                                          bool strided) {
    ensure_device(dev);
    TI_ERROR_IF(!subgroup_scan_enabled || value_type < 0 || value_type > 2,
                "Unsupported Vulkan reverse small subgroup scan value type.");
    const auto &spv = vulkan_scan_reverse_spv_set(value_type);
    auto &pipeline =
        strided ? scan_small_strided_reverse[value_type]
                : scan_small_reverse[value_type];
    if (pipeline) {
      return;
    }
    const uint32_t *shader = strided ? spv.small_strided_spv : spv.small_spv;
    const size_t shader_bytes =
        strided ? spv.small_strided_bytes : spv.small_bytes;
    TI_ERROR_IF(!shader, "Missing Vulkan reverse small subgroup scan shader.");
    pipeline = create_pipeline_from_spv(
        dev, shader, shader_bytes,
        fmt::format("vulkan_scan_{}_small_subgroup{}_reverse", spv.dtype_name,
                    strided ? "_strided" : ""));
  }

  void ensure_strided_pipelines(Device *dev, int value_type) {
    ensure_scan_strided_block_add_pipelines(dev, value_type);
    if (value_type >= 0 && value_type < 3 && subgroup_scan_enabled &&
        !scan_small_strided[value_type]) {
      const auto &spv = vulkan_scan_strided_spv_set(value_type);
      TI_ERROR_IF(!spv.small_spv,
                  "Missing Vulkan strided small subgroup scan shader.");
      scan_small_strided[value_type] = create_pipeline_from_spv(
          dev, spv.small_spv, spv.small_bytes,
          fmt::format("vulkan_scan_{}_small_subgroup_strided",
                      spv.dtype_name));
    }
    ensure_params();
  }

  Pipeline *scan_small_pipeline(int value_type) const {
    switch (value_type) {
      case 0:
        return scan_i32_small_subgroup.get();
      case 1:
        return scan_f32_small_subgroup.get();
      case 2:
        return scan_u32_small_subgroup.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_block_pipeline(int value_type, bool subgroup) const {
    switch (value_type) {
      case 0:
        return subgroup ? scan_i32_block_subgroup.get() : scan_i32_block.get();
      case 1:
        return subgroup ? scan_f32_block_subgroup.get() : scan_f32_block.get();
      case 2:
        return subgroup ? scan_u32_block_subgroup.get() : scan_u32_block.get();
      case 3:
        return subgroup ? nullptr : scan_u64_block.get();
      case 4:
        return subgroup ? nullptr : scan_i64_block.get();
      case 5:
        return subgroup ? nullptr : scan_f64_block.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_add_pipeline(int value_type) const {
    switch (value_type) {
      case 0:
        return scan_i32_add.get();
      case 1:
        return scan_f32_add.get();
      case 2:
        return scan_u32_add.get();
      case 3:
        return scan_u64_add.get();
      case 4:
        return scan_i64_add.get();
      case 5:
        return scan_f64_add.get();
      default:
        return nullptr;
    }
  }

  Pipeline *scan_block_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return scan_block_strided[value_type].get();
  }

  Pipeline *scan_add_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return scan_add_strided[value_type].get();
  }

  Pipeline *scan_small_strided_pipeline(int value_type) const {
    if (value_type < 0 || value_type >= 3) {
      return nullptr;
    }
    return scan_small_strided[value_type].get();
  }

  Pipeline *scan_block_reverse_pipeline(int value_type, bool strided) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return (strided ? scan_block_strided_reverse[value_type]
                    : scan_block_reverse[value_type])
        .get();
  }

  Pipeline *scan_add_reverse_pipeline(int value_type, bool strided) const {
    if (value_type < 0 || value_type >= 6) {
      return nullptr;
    }
    return (strided ? scan_add_strided_reverse[value_type]
                    : scan_add_reverse[value_type])
        .get();
  }

  Pipeline *scan_small_reverse_pipeline(int value_type, bool strided) const {
    if (value_type < 0 || value_type >= 3) {
      return nullptr;
    }
    return (strided ? scan_small_strided_reverse[value_type]
                    : scan_small_reverse[value_type])
        .get();
  }

  const char *scan_small_scope(int value_type) const {
    switch (value_type) {
      case 0:
        return "vulkan_scan_i32_small_subgroup";
      case 1:
        return "vulkan_scan_f32_small_subgroup";
      case 2:
        return "vulkan_scan_u32_small_subgroup";
      default:
        return "vulkan_scan_unknown_small_subgroup";
    }
  }

  const char *scan_block_scope(int value_type, bool subgroup) const {
    switch (value_type) {
      case 0:
        return subgroup ? "vulkan_scan_i32_block_subgroup"
                        : "vulkan_scan_i32_block";
      case 1:
        return subgroup ? "vulkan_scan_f32_block_subgroup"
                        : "vulkan_scan_f32_block";
      case 2:
        return subgroup ? "vulkan_scan_u32_block_subgroup"
                        : "vulkan_scan_u32_block";
      case 3:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_u64_block";
      case 4:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_i64_block";
      case 5:
        return subgroup ? "vulkan_scan_unknown_block"
                        : "vulkan_scan_f64_block";
      default:
        return "vulkan_scan_unknown_block";
    }
  }

  const char *scan_add_scope(int value_type) const {
    switch (value_type) {
      case 0:
        return "vulkan_scan_i32_add";
      case 1:
        return "vulkan_scan_f32_add";
      case 2:
        return "vulkan_scan_u32_add";
      case 3:
        return "vulkan_scan_u64_add";
      case 4:
        return "vulkan_scan_i64_add";
      case 5:
        return "vulkan_scan_f64_add";
      default:
        return "vulkan_scan_unknown_add";
    }
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan scan workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_workspace_realloc(size_t bytes) const {
    return capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return workspace != kDeviceNullAllocation ||
           dummy_sums != kDeviceNullAllocation;
  }

  size_t allocated_bytes() const {
    return capacity + dummy_sums_capacity + params_capacity;
  }

  void clear_workspace_alloc() {
    if (device && workspace != kDeviceNullAllocation) {
      device->dealloc_memory(workspace);
    }
    workspace = kDeviceNullAllocation;
    capacity = 0;
    cached_bytes = allocated_bytes();
  }

  void ensure_dummy_sums() {
    if (dummy_sums == kDeviceNullAllocation) {
      dummy_sums_capacity = sizeof(uint64_t);
      dummy_sums = alloc_storage(dummy_sums_capacity);
    }
    cached_bytes = allocated_bytes();
  }

  void ensure_params() {
    constexpr size_t kParamsBytes = 3 * sizeof(uint32_t);
    if (params == kDeviceNullAllocation) {
      params = alloc_storage(kParamsBytes);
      params_capacity = kParamsBytes;
    }
    cached_bytes = allocated_bytes();
  }

  void ensure_workspace(size_t bytes) {
    ensure_dummy_sums();
    if (bytes == 0 || !needs_workspace_realloc(bytes)) {
      return;
    }
    clear_workspace_alloc();
    workspace = alloc_storage(bytes);
    capacity = bytes;
    cached_bytes = allocated_bytes();
  }
};

struct VulkanCompactCache {
  Device *device{nullptr};
  size_t prefix_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation prefix{kDeviceNullAllocation};
  VulkanScanCache scan;
  std::unique_ptr<Pipeline> compact_i32_flags;
  std::unique_ptr<Pipeline> compact_i32_scatter;
  std::unique_ptr<Pipeline> compact_i32_scatter_packet;
  VulkanResourceSetReplayRing<2> flags_bindings;
  VulkanResourceSetReplayRing<5> scatter_bindings;
  VulkanResourceSetReplayRing<6> scatter_packet_bindings;
  VulkanCommandReplayCache ndarray_fused_command_replay;
  VulkanCommandReplayCache dense_field_fused_command_replay;
  VulkanCommandReplayCache ndarray_flags_command_replay;
  VulkanCommandReplayCache ndarray_scatter_command_replay;
  VulkanCommandReplayCache dense_field_flags_command_replay;
  VulkanCommandReplayCache dense_field_scatter_command_replay;

  void reset_resource_sets() {
    flags_bindings.reset();
    scatter_bindings.reset();
    scatter_packet_bindings.reset();
    ndarray_fused_command_replay.reset();
    dense_field_fused_command_replay.reset();
    ndarray_flags_command_replay.reset();
    ndarray_scatter_command_replay.reset();
    dense_field_flags_command_replay.reset();
    dense_field_scatter_command_replay.reset();
  }

  void clear_allocs() {
    reset_resource_sets();
    if (device && prefix != kDeviceNullAllocation) {
      device->dealloc_memory(prefix);
    }
    prefix = kDeviceNullAllocation;
    prefix_capacity = 0;
    cached_bytes = scan.cached_bytes;
  }

  ~VulkanCompactCache() {
    clear_allocs();
  }

  void reset_pipelines() {
    compact_i32_flags.reset();
    compact_i32_scatter.reset();
    compact_i32_scatter_packet.reset();
    reset_resource_sets();
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      scan.ensure_device(dev);
      cached_bytes = prefix_capacity + scan.cached_bytes;
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      reset_pipelines();
    }
    device = dev;
    scan.ensure_device(dev);
    cached_bytes = prefix_capacity + scan.cached_bytes;
  }

  void ensure_compact_pipelines(Device *dev) {
    ensure_device(dev);
    if (!compact_i32_flags) {
      compact_i32_flags =
          create_pipeline(dev, kCompactI32FlagsSpv, "vulkan_compact_i32_flags");
    }
    if (!compact_i32_scatter) {
      compact_i32_scatter = create_pipeline(
          dev, kCompactI32ScatterSpv, "vulkan_compact_i32_scatter");
    }
    if (!compact_i32_scatter_packet) {
      compact_i32_scatter_packet = create_pipeline(
          dev, kCompactI32ScatterPacketSpv,
          "vulkan_compact_i32_scatter_packet");
    }
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan compact workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_prefix_realloc(size_t bytes) const {
    return prefix_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return prefix != kDeviceNullAllocation || scan.has_workspace_allocs();
  }

  void clear_prefix_alloc() {
    reset_resource_sets();
    if (device && prefix != kDeviceNullAllocation) {
      device->dealloc_memory(prefix);
    }
    prefix = kDeviceNullAllocation;
    prefix_capacity = 0;
    cached_bytes = scan.cached_bytes;
  }

  void ensure_prefix(size_t bytes) {
    if (bytes == 0 || !needs_prefix_realloc(bytes)) {
      cached_bytes = prefix_capacity + scan.cached_bytes;
      return;
    }
    clear_prefix_alloc();
    prefix = alloc_storage(bytes);
    prefix_capacity = bytes;
    cached_bytes = prefix_capacity + scan.cached_bytes;
  }

  size_t allocated_bytes() const {
    return prefix_capacity + scan.cached_bytes;
  }

  VulkanReplayResourceSet<2> bind_flags_resource_set(
      Program *program,
      DeviceAllocation flags_alloc,
      uint64_t flags_offset,
      DeviceAllocation prefix_alloc,
      size_t prefix_bytes) {
    return flags_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(flags_alloc, flags_offset, prefix_bytes),
            rw_buffer_request(prefix_alloc, 0, prefix_bytes)});
  }

  VulkanReplayResourceSet<5> bind_scatter_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_total_bytes,
      DeviceAllocation flags_alloc,
      uint64_t flags_offset,
      size_t prefix_bytes,
      DeviceAllocation prefix_alloc,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      DeviceAllocation count_alloc,
      uint64_t count_offset) {
    return scatter_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(values_alloc, values_offset, value_total_bytes),
            rw_buffer_request(flags_alloc, flags_offset, prefix_bytes),
            rw_buffer_request(prefix_alloc, 0, prefix_bytes),
            rw_buffer_request(output_alloc, output_offset, value_total_bytes),
            rw_buffer_request(count_alloc, count_offset, sizeof(int32_t))});
  }

  VulkanReplayResourceSet<6> bind_scatter_packet_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_total_bytes,
      DeviceAllocation flags_alloc,
      uint64_t flags_offset,
      size_t prefix_bytes,
      DeviceAllocation prefix_alloc,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      DeviceAllocation count_alloc,
      uint64_t count_offset,
      DeviceAllocation packet_alloc,
      uint64_t packet_offset) {
    return scatter_packet_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 6>{
            rw_buffer_request(values_alloc, values_offset, value_total_bytes),
            rw_buffer_request(flags_alloc, flags_offset, prefix_bytes),
            rw_buffer_request(prefix_alloc, 0, prefix_bytes),
            rw_buffer_request(output_alloc, output_offset, value_total_bytes),
            rw_buffer_request(count_alloc, count_offset, sizeof(int32_t)),
            rw_buffer_request(packet_alloc, packet_offset,
                              4 * sizeof(uint32_t))});
  }
};

struct VulkanHistogramCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> histogram_i32_clear;
  std::unique_ptr<Pipeline> histogram_i32_count_direct;
  std::unique_ptr<Pipeline> histogram_i32_count_private;
  std::unique_ptr<Pipeline> histogram_i32_count_private_shared;
  std::unique_ptr<Pipeline> histogram_i32_reduce_private;
  std::unique_ptr<Pipeline> histogram_i32_single_shared;
  std::unique_ptr<Pipeline> histogram_u32_count_direct;
  std::unique_ptr<Pipeline> histogram_u32_count_private;
  std::unique_ptr<Pipeline> histogram_u32_count_private_shared;
  std::unique_ptr<Pipeline> histogram_u32_single_shared;
  std::unique_ptr<Pipeline> histogram_i64_clear;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_direct;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_private;
  std::unique_ptr<Pipeline> histogram_i32_i64_count_private_shared;
  std::unique_ptr<Pipeline> histogram_i64_reduce_private;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_direct;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_private;
  std::unique_ptr<Pipeline> histogram_u32_i64_count_private_shared;
  VulkanResourceSetReplayRing<1> clear_bindings;
  VulkanResourceSetReplayRing<2> count_direct_bindings;
  VulkanResourceSetReplayRing<3> count_private_bindings;
  VulkanResourceSetReplayRing<3> count_private_shared_bindings;
  VulkanResourceSetReplayRing<2> reduce_private_bindings;
  VulkanResourceSetReplayRing<2> single_shared_bindings;
  VulkanCommandReplayCache ndarray_command_replay;
  VulkanCommandReplayCache dense_field_command_replay;

  void reset_resource_sets() {
    clear_bindings.reset();
    count_direct_bindings.reset();
    count_private_bindings.reset();
    count_private_shared_bindings.reset();
    reduce_private_bindings.reset();
    single_shared_bindings.reset();
    ndarray_command_replay.reset();
    dense_field_command_replay.reset();
  }

  void reset_pipelines() {
    histogram_i32_clear.reset();
    histogram_i32_count_direct.reset();
    histogram_i32_count_private.reset();
    histogram_i32_count_private_shared.reset();
    histogram_i32_reduce_private.reset();
    histogram_i32_single_shared.reset();
    histogram_u32_count_direct.reset();
    histogram_u32_count_private.reset();
    histogram_u32_count_private_shared.reset();
    histogram_u32_single_shared.reset();
    histogram_i64_clear.reset();
    histogram_i32_i64_count_direct.reset();
    histogram_i32_i64_count_private.reset();
    histogram_i32_i64_count_private_shared.reset();
    histogram_i64_reduce_private.reset();
    histogram_u32_i64_count_direct.reset();
    histogram_u32_i64_count_private.reset();
    histogram_u32_i64_count_private_shared.reset();
    reset_resource_sets();
  }

  void clear_allocs() {
    reset_resource_sets();
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanHistogramCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      reset_pipelines();
    }
    device = dev;
  }

  bool supports_i64_bins() const {
    return device &&
           device->get_caps().get(DeviceCapability::spirv_has_int64) != 0 &&
           device->get_caps().get(DeviceCapability::spirv_has_atomic_int64) !=
               0;
  }

  Pipeline *clear_pipeline(int bin_type) {
    if (bin_type == 4) {
      TI_ERROR_IF(!supports_i64_bins(),
                  "Vulkan histogram i64 bins are not supported by this device.");
      if (!histogram_i64_clear) {
        histogram_i64_clear = create_pipeline(
            device, kHistogramI64ClearSpv, "vulkan_histogram_i64_clear");
      }
      return histogram_i64_clear.get();
    }
    if (!histogram_i32_clear) {
      histogram_i32_clear = create_pipeline(
          device, kHistogramI32ClearSpv, "vulkan_histogram_i32_clear");
    }
    return histogram_i32_clear.get();
  }

  Pipeline *reduce_private_pipeline(int bin_type) {
    if (bin_type == 4) {
      TI_ERROR_IF(!supports_i64_bins(),
                  "Vulkan histogram i64 bins are not supported by this device.");
      if (!histogram_i64_reduce_private) {
        histogram_i64_reduce_private = create_pipeline(
            device, kHistogramI64ReducePrivateSpv,
            "vulkan_histogram_i64_reduce_private");
      }
      return histogram_i64_reduce_private.get();
    }
    if (!histogram_i32_reduce_private) {
      histogram_i32_reduce_private = create_pipeline(
          device, kHistogramI32ReducePrivateSpv,
          "vulkan_histogram_i32_reduce_private");
    }
    return histogram_i32_reduce_private.get();
  }

  Pipeline *count_direct_pipeline(int value_type, int bin_type) {
    if (bin_type == 4) {
      TI_ERROR_IF(!supports_i64_bins(),
                  "Vulkan histogram i64 bins are not supported by this device.");
      if (value_type == 2) {
        if (!histogram_u32_i64_count_direct) {
          histogram_u32_i64_count_direct = create_pipeline(
              device, kHistogramU32I64CountDirectSpv,
              "vulkan_histogram_u32_i64_count_direct");
        }
        return histogram_u32_i64_count_direct.get();
      }
      if (!histogram_i32_i64_count_direct) {
        histogram_i32_i64_count_direct = create_pipeline(
            device, kHistogramI32I64CountDirectSpv,
            "vulkan_histogram_i32_i64_count_direct");
      }
      return histogram_i32_i64_count_direct.get();
    }
    if (value_type == 2) {
      if (!histogram_u32_count_direct) {
        histogram_u32_count_direct = create_pipeline(
            device, kHistogramU32CountDirectSpv,
            "vulkan_histogram_u32_count_direct");
      }
      return histogram_u32_count_direct.get();
    }
    if (!histogram_i32_count_direct) {
      histogram_i32_count_direct = create_pipeline(
          device, kHistogramI32CountDirectSpv,
          "vulkan_histogram_i32_count_direct");
    }
    return histogram_i32_count_direct.get();
  }

  Pipeline *count_private_pipeline(int value_type, int bin_type) {
    if (bin_type == 4) {
      TI_ERROR_IF(!supports_i64_bins(),
                  "Vulkan histogram i64 bins are not supported by this device.");
      if (value_type == 2) {
        if (!histogram_u32_i64_count_private) {
          histogram_u32_i64_count_private = create_pipeline(
              device, kHistogramU32I64CountPrivateSpv,
              "vulkan_histogram_u32_i64_count_private");
        }
        return histogram_u32_i64_count_private.get();
      }
      if (!histogram_i32_i64_count_private) {
        histogram_i32_i64_count_private = create_pipeline(
            device, kHistogramI32I64CountPrivateSpv,
            "vulkan_histogram_i32_i64_count_private");
      }
      return histogram_i32_i64_count_private.get();
    }
    if (value_type == 2) {
      if (!histogram_u32_count_private) {
        histogram_u32_count_private = create_pipeline(
            device, kHistogramU32CountPrivateSpv,
            "vulkan_histogram_u32_count_private");
      }
      return histogram_u32_count_private.get();
    }
    if (!histogram_i32_count_private) {
      histogram_i32_count_private = create_pipeline(
          device, kHistogramI32CountPrivateSpv,
          "vulkan_histogram_i32_count_private");
    }
    return histogram_i32_count_private.get();
  }

  Pipeline *count_private_shared_pipeline(int value_type) {
    if (value_type == 2) {
      if (!histogram_u32_count_private_shared) {
        histogram_u32_count_private_shared = create_pipeline(
            device, kHistogramU32CountPrivateSharedSpv,
            "vulkan_histogram_u32_count_private_shared");
      }
      return histogram_u32_count_private_shared.get();
    }
    if (!histogram_i32_count_private_shared) {
      histogram_i32_count_private_shared = create_pipeline(
          device, kHistogramI32CountPrivateSharedSpv,
          "vulkan_histogram_i32_count_private_shared");
    }
    return histogram_i32_count_private_shared.get();
  }

  Pipeline *count_private_shared_pipeline(int value_type, int bin_type) {
    if (bin_type == 4) {
      TI_ERROR_IF(!supports_i64_bins(),
                  "Vulkan histogram i64 bins are not supported by this device.");
      if (value_type == 2) {
        if (!histogram_u32_i64_count_private_shared) {
          histogram_u32_i64_count_private_shared = create_pipeline(
              device, kHistogramU32I64CountPrivateSharedSpv,
              "vulkan_histogram_u32_i64_count_private_shared");
        }
        return histogram_u32_i64_count_private_shared.get();
      }
      if (!histogram_i32_i64_count_private_shared) {
        histogram_i32_i64_count_private_shared = create_pipeline(
            device, kHistogramI32I64CountPrivateSharedSpv,
            "vulkan_histogram_i32_i64_count_private_shared");
      }
      return histogram_i32_i64_count_private_shared.get();
    }
    return count_private_shared_pipeline(value_type);
  }

  Pipeline *single_shared_pipeline(int value_type) {
    if (value_type == 2) {
      if (!histogram_u32_single_shared) {
        histogram_u32_single_shared = create_pipeline(
            device, kHistogramU32SingleSharedSpv,
            "vulkan_histogram_u32_single_shared");
      }
      return histogram_u32_single_shared.get();
    }
    if (!histogram_i32_single_shared) {
      histogram_i32_single_shared = create_pipeline(
          device, kHistogramI32SingleSharedSpv,
          "vulkan_histogram_i32_single_shared");
    }
    return histogram_i32_single_shared.get();
  }

  const char *clear_scope(int bin_type) const {
    return bin_type == 4 ? "vulkan_histogram_i64_clear_bins"
                         : "vulkan_histogram_i32_clear_bins";
  }

  const char *reduce_private_scope(int bin_type) const {
    return bin_type == 4 ? "vulkan_histogram_i64_reduce_private"
                         : "vulkan_histogram_i32_reduce_private";
  }

  const char *count_direct_scope(int value_type, int bin_type) const {
    if (bin_type == 4) {
      return value_type == 2 ? "vulkan_histogram_u32_i64_count_direct"
                             : "vulkan_histogram_i32_i64_count_direct";
    }
    return value_type == 2 ? "vulkan_histogram_u32_count_direct"
                           : "vulkan_histogram_i32_count_direct";
  }

  const char *count_private_scope(int value_type,
                                  int bin_type,
                                  bool shared) const {
    if (bin_type == 4) {
      if (value_type == 2) {
        return shared ? "vulkan_histogram_u32_i64_count_private_shared"
                      : "vulkan_histogram_u32_i64_count_private";
      }
      return shared ? "vulkan_histogram_i32_i64_count_private_shared"
                    : "vulkan_histogram_i32_i64_count_private";
    }
    if (value_type == 2) {
      return shared ? "vulkan_histogram_u32_count_private_shared"
                    : "vulkan_histogram_u32_count_private";
    }
    return shared ? "vulkan_histogram_i32_count_private_shared"
                  : "vulkan_histogram_i32_count_private";
  }

  const char *single_shared_scope(int value_type) const {
    return value_type == 2 ? "vulkan_histogram_u32_single_shared"
                           : "vulkan_histogram_i32_single_shared";
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan histogram workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_partial_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return partial != kDeviceNullAllocation;
  }

  void clear_partial_alloc() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = partial_capacity;
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = partial_capacity;
  }

  VulkanReplayResourceSet<1> bind_clear_resource_set(Program *program,
                                                     DeviceAllocation alloc,
                                                     uint64_t offset,
                                                     size_t bytes) {
    return clear_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 1>{
            rw_buffer_request(alloc, offset, bytes)});
  }

  VulkanReplayResourceSet<2> bind_count_direct_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation bins_alloc,
      uint64_t bins_offset,
      size_t bin_bytes) {
    return count_direct_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(bins_alloc, bins_offset, bin_bytes)});
  }

  VulkanReplayResourceSet<3> bind_count_private_resource_set(
      Program *program,
      bool shared,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation bins_alloc,
      uint64_t bins_offset,
      size_t bin_bytes,
      DeviceAllocation partial_alloc,
      uint64_t partial_offset,
      size_t partial_bytes) {
    auto &bindings =
        shared ? count_private_shared_bindings : count_private_bindings;
    return bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(bins_alloc, bins_offset, bin_bytes),
            rw_buffer_request(partial_alloc, partial_offset, partial_bytes)});
  }

  VulkanReplayResourceSet<2> bind_reduce_private_resource_set(
      Program *program,
      DeviceAllocation partial_alloc,
      uint64_t partial_offset,
      size_t partial_bytes,
      DeviceAllocation bins_alloc,
      uint64_t bins_offset,
      size_t bin_bytes) {
    return reduce_private_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(partial_alloc, partial_offset, partial_bytes),
            rw_buffer_request(bins_alloc, bins_offset, bin_bytes)});
  }

  VulkanReplayResourceSet<2> bind_single_shared_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation bins_alloc,
      uint64_t bins_offset,
      size_t bin_bytes) {
    return single_shared_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(bins_alloc, bins_offset, bin_bytes)});
  }
};

struct VulkanReduceSpvSet {
  const uint32_t *private_spv{nullptr};
  size_t private_bytes{0};
  const uint32_t *private_strided_spv{nullptr};
  size_t private_strided_bytes{0};
  const uint32_t *final_spv{nullptr};
  size_t final_bytes{0};
  const uint32_t *single_spv{nullptr};
  size_t single_bytes{0};
  const uint32_t *single_strided_spv{nullptr};
  size_t single_strided_bytes{0};
  const char *dtype_name{nullptr};
  const char *op_name{nullptr};
};

#define TI_REDUCE_SPV_SET(TYPE_CAPS, TYPE_NAME, OP_CAPS, OP_NAME)       \
  VulkanReduceSpvSet {                                                  \
    kReduce##TYPE_CAPS##OP_CAPS##PrivateSpv,                            \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##PrivateSpv),                \
        kReduce##TYPE_CAPS##OP_CAPS##PrivateStridedSpv,                 \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##PrivateStridedSpv),         \
        kReduce##TYPE_CAPS##OP_CAPS##FinalSpv,                          \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##FinalSpv),                  \
        kReduce##TYPE_CAPS##OP_CAPS##SingleSpv,                         \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##SingleSpv),                 \
        kReduce##TYPE_CAPS##OP_CAPS##SingleStridedSpv,                  \
        sizeof(kReduce##TYPE_CAPS##OP_CAPS##SingleStridedSpv),          \
        TYPE_NAME, OP_NAME                                               \
  }

const VulkanReduceSpvSet &vulkan_reduce_spv_set(int value_type, int op) {
  static const std::array<std::array<VulkanReduceSpvSet, 3>, 6> table = {{
      {TI_REDUCE_SPV_SET(I32, "i32", Sum, "sum"),
       TI_REDUCE_SPV_SET(I32, "i32", Min, "min"),
       TI_REDUCE_SPV_SET(I32, "i32", Max, "max")},
      {TI_REDUCE_SPV_SET(F32, "f32", Sum, "sum"),
       TI_REDUCE_SPV_SET(F32, "f32", Min, "min"),
       TI_REDUCE_SPV_SET(F32, "f32", Max, "max")},
      {TI_REDUCE_SPV_SET(U32, "u32", Sum, "sum"),
       TI_REDUCE_SPV_SET(U32, "u32", Min, "min"),
       TI_REDUCE_SPV_SET(U32, "u32", Max, "max")},
      {TI_REDUCE_SPV_SET(U64, "u64", Sum, "sum"),
       TI_REDUCE_SPV_SET(U64, "u64", Min, "min"),
       TI_REDUCE_SPV_SET(U64, "u64", Max, "max")},
      {TI_REDUCE_SPV_SET(I64, "i64", Sum, "sum"),
       TI_REDUCE_SPV_SET(I64, "i64", Min, "min"),
       TI_REDUCE_SPV_SET(I64, "i64", Max, "max")},
      {TI_REDUCE_SPV_SET(F64, "f64", Sum, "sum"),
       TI_REDUCE_SPV_SET(F64, "f64", Min, "min"),
       TI_REDUCE_SPV_SET(F64, "f64", Max, "max")},
  }};
  return table[value_type][op];
}

#undef TI_REDUCE_SPV_SET

struct VulkanReducePipelineSet {
  std::array<std::unique_ptr<Pipeline>, 3> private_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> private_strided_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> final_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> single_pipelines;
  std::array<std::unique_ptr<Pipeline>, 3> single_strided_pipelines;

  void reset() {
    for (int i = 0; i < 3; ++i) {
      private_pipelines[i].reset();
      private_strided_pipelines[i].reset();
      final_pipelines[i].reset();
      single_pipelines[i].reset();
      single_strided_pipelines[i].reset();
    }
  }
};

struct VulkanReduceCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t params_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  DeviceAllocation params{kDeviceNullAllocation};
  std::array<VulkanReducePipelineSet, 6> reduce_pipelines;
  std::unique_ptr<Pipeline> reduce_i32_sum_atomic;
  VulkanResourceSetReplayRing<2> reduce_i32_sum_atomic_bindings;
  VulkanResourceSetReplayRing<2> reduce_single_bindings;
  VulkanResourceSetReplayRing<3> reduce_single_strided_bindings;
  VulkanResourceSetReplayRing<2> reduce_private_bindings;
  VulkanResourceSetReplayRing<3> reduce_private_strided_bindings;
  VulkanResourceSetReplayRing<2> reduce_final_bindings;
  VulkanCommandReplayCache reduce_i32_sum_atomic_command_replay;
  VulkanCommandReplayCache reduce_tree_command_replay;

  void reset_resource_sets() {
    reduce_i32_sum_atomic_bindings.reset();
    reduce_single_bindings.reset();
    reduce_single_strided_bindings.reset();
    reduce_private_bindings.reset();
    reduce_private_strided_bindings.reset();
    reduce_final_bindings.reset();
    reduce_i32_sum_atomic_command_replay.reset();
    reduce_tree_command_replay.reset();
  }

  void clear_allocs() {
    reset_resource_sets();
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    if (device && params != kDeviceNullAllocation) {
      device->dealloc_memory(params);
    }
    partial = kDeviceNullAllocation;
    params = kDeviceNullAllocation;
    partial_capacity = 0;
    params_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanReduceCache() {
    clear_allocs();
  }

  void reset_pipelines() {
    for (auto &pipelines : reduce_pipelines) {
      pipelines.reset();
    }
    reduce_i32_sum_atomic.reset();
    reset_resource_sets();
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      reset_pipelines();
    }
    device = dev;
  }

  Pipeline *i32_sum_atomic_pipeline(Device *dev) {
    ensure_device(dev);
    if (!reduce_i32_sum_atomic) {
      reduce_i32_sum_atomic = create_pipeline(
          dev, kReduceI32SumAtomicSpv, "vulkan_reduce_i32_sum_atomic");
    }
    return reduce_i32_sum_atomic.get();
  }

  VulkanReducePipelineSet &pipeline_set(int value_type) {
    return reduce_pipelines[value_type];
  }

  void ensure_pipeline_set(Device *dev, int value_type) {
    ensure_device(dev);
    auto &pipelines = pipeline_set(value_type);
    if (pipelines.private_pipelines[0]) {
      return;
    }
    for (int op = 0; op < 3; ++op) {
      const auto &spv = vulkan_reduce_spv_set(value_type, op);
      pipelines.private_pipelines[op] = create_pipeline_from_spv(
          dev, spv.private_spv, spv.private_bytes,
          fmt::format("vulkan_reduce_{}_{}_private", spv.dtype_name,
                      spv.op_name));
      pipelines.final_pipelines[op] = create_pipeline_from_spv(
          dev, spv.final_spv, spv.final_bytes,
          fmt::format("vulkan_reduce_{}_{}_final", spv.dtype_name,
                      spv.op_name));
      pipelines.single_pipelines[op] = create_pipeline_from_spv(
          dev, spv.single_spv, spv.single_bytes,
          fmt::format("vulkan_reduce_{}_{}_single", spv.dtype_name,
                      spv.op_name));
    }
  }

  void ensure_strided_pipeline_set(Device *dev, int value_type) {
    ensure_pipeline_set(dev, value_type);
    auto &pipelines = pipeline_set(value_type);
    if (pipelines.private_strided_pipelines[0]) {
      return;
    }
    for (int op = 0; op < 3; ++op) {
      const auto &spv = vulkan_reduce_spv_set(value_type, op);
      pipelines.private_strided_pipelines[op] = create_pipeline_from_spv(
          dev, spv.private_strided_spv, spv.private_strided_bytes,
          fmt::format("vulkan_reduce_{}_{}_private_strided", spv.dtype_name,
                      spv.op_name));
      pipelines.single_strided_pipelines[op] = create_pipeline_from_spv(
          dev, spv.single_strided_spv, spv.single_strided_bytes,
          fmt::format("vulkan_reduce_{}_{}_single_strided", spv.dtype_name,
                      spv.op_name));
    }
  }

  void ensure_op_pipelines(Device *dev,
                           int value_type,
                           int op,
                           bool strided_source,
                           bool use_single_shared) {
    ensure_device(dev);
    auto &pipelines = pipeline_set(value_type);
    const auto &spv = vulkan_reduce_spv_set(value_type, op);
    if (use_single_shared) {
      if (strided_source) {
        if (!pipelines.single_strided_pipelines[op]) {
          pipelines.single_strided_pipelines[op] = create_pipeline_from_spv(
              dev, spv.single_strided_spv, spv.single_strided_bytes,
              fmt::format("vulkan_reduce_{}_{}_single_strided",
                          spv.dtype_name, spv.op_name));
        }
      } else if (!pipelines.single_pipelines[op]) {
        pipelines.single_pipelines[op] = create_pipeline_from_spv(
            dev, spv.single_spv, spv.single_bytes,
            fmt::format("vulkan_reduce_{}_{}_single", spv.dtype_name,
                        spv.op_name));
      }
      return;
    }
    if (strided_source) {
      if (!pipelines.private_strided_pipelines[op]) {
        pipelines.private_strided_pipelines[op] = create_pipeline_from_spv(
            dev, spv.private_strided_spv, spv.private_strided_bytes,
            fmt::format("vulkan_reduce_{}_{}_private_strided", spv.dtype_name,
                        spv.op_name));
      }
    } else if (!pipelines.private_pipelines[op]) {
      pipelines.private_pipelines[op] = create_pipeline_from_spv(
          dev, spv.private_spv, spv.private_bytes,
          fmt::format("vulkan_reduce_{}_{}_private", spv.dtype_name,
                      spv.op_name));
    }
    if (!pipelines.final_pipelines[op]) {
      pipelines.final_pipelines[op] = create_pipeline_from_spv(
          dev, spv.final_spv, spv.final_bytes,
          fmt::format("vulkan_reduce_{}_{}_final", spv.dtype_name,
                      spv.op_name));
    }
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan reduce workspace: RhiResult({})",
                res);
    return alloc;
  }

  bool needs_partial_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return partial != kDeviceNullAllocation || params != kDeviceNullAllocation;
  }

  size_t allocated_bytes() const {
    return partial_capacity + params_capacity;
  }

  void clear_partial_alloc() {
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = allocated_bytes();
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = allocated_bytes();
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = allocated_bytes();
  }

  void ensure_params() {
    constexpr size_t kParamsBytes = 3 * sizeof(uint32_t);
    if (params == kDeviceNullAllocation) {
      params = alloc_storage(kParamsBytes);
      params_capacity = kParamsBytes;
    }
    cached_bytes = allocated_bytes();
  }

  VulkanReplayResourceSet<2> bind_i32_sum_atomic_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return reduce_i32_sum_atomic_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }

  VulkanReplayResourceSet<2> bind_single_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return reduce_single_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }

  VulkanReplayResourceSet<3> bind_single_strided_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes,
      DeviceAllocation params_alloc,
      size_t params_bytes) {
    return reduce_single_strided_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes),
            rw_buffer_request(params_alloc, 0, params_bytes)});
  }

  VulkanReplayResourceSet<2> bind_private_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation partial_alloc,
      size_t partial_bytes) {
    return reduce_private_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(partial_alloc, 0, partial_bytes)});
  }

  VulkanReplayResourceSet<3> bind_private_strided_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation partial_alloc,
      size_t partial_bytes,
      DeviceAllocation params_alloc,
      size_t params_bytes) {
    return reduce_private_strided_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(partial_alloc, 0, partial_bytes),
            rw_buffer_request(params_alloc, 0, params_bytes)});
  }

  VulkanReplayResourceSet<2> bind_final_resource_set(
      Program *program,
      DeviceAllocation partial_alloc,
      size_t partial_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return reduce_final_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(partial_alloc, 0, partial_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }
};

struct VulkanCheckCountSpv {
  const uint32_t *spv{nullptr};
  size_t bytes{0};
  const char *dtype_name{nullptr};
};

VulkanCheckCountSpv vulkan_check_count_spv(int value_type) {
  switch (value_type) {
    case 0:
      return {kCheckCountI32Spv, sizeof(kCheckCountI32Spv), "i32"};
    case 1:
      return {kCheckCountF32Spv, sizeof(kCheckCountF32Spv), "f32"};
    case 2:
      return {kCheckCountU32Spv, sizeof(kCheckCountU32Spv), "u32"};
    case 3:
      return {kCheckCountU64Spv, sizeof(kCheckCountU64Spv), "u64"};
    case 4:
      return {kCheckCountI64Spv, sizeof(kCheckCountI64Spv), "i64"};
    case 5:
      return {kCheckCountF64Spv, sizeof(kCheckCountF64Spv), "f64"};
  }
  return {};
}

struct VulkanCheckCountCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  std::array<std::unique_ptr<Pipeline>, 6> pipelines;
  VulkanResourceSetReplayRing<2> bindings;
  VulkanCommandReplayCache command_replay;

  void clear_allocs() {
    bindings.reset();
    command_replay.reset();
    cached_bytes = 0;
  }

  ~VulkanCheckCountCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void reset_pipelines() {
    for (auto &pipeline : pipelines) {
      pipeline.reset();
    }
    clear_allocs();
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      reset_pipelines();
    }
    device = dev;
  }

  Pipeline *pipeline_for(Device *dev, int value_type) {
    ensure_device(dev);
    TI_ERROR_IF(value_type < 0 || value_type >= 6,
                "Vulkan native check_count received an unsupported value "
                "type.");
    if (!pipelines[value_type]) {
      const auto info = vulkan_check_count_spv(value_type);
      TI_ERROR_IF(!info.spv || info.bytes == 0,
                  "Vulkan native check_count could not find shader for value "
                  "type {}.",
                  value_type);
      pipelines[value_type] = create_pipeline_from_spv(
          dev, info.spv, info.bytes,
          fmt::format("vulkan_check_count_{}", info.dtype_name));
    }
    return pipelines[value_type].get();
  }

  VulkanReplayResourceSet<2> bind_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t values_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(values_alloc, values_offset, values_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }
};

struct VulkanMetricReduceCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  std::unique_ptr<Pipeline> metric_reduce_f32;
  VulkanResourceSetReplayRing<3> bindings;
  VulkanCommandReplayCache command_replay;

  void clear_allocs() {
    bindings.reset();
    command_replay.reset();
    cached_bytes = 0;
  }

  ~VulkanMetricReduceCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void reset_pipelines() {
    metric_reduce_f32.reset();
    clear_allocs();
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      reset_pipelines();
    }
    device = dev;
  }

  Pipeline *pipeline_for(Device *dev, int value_type) {
    ensure_device(dev);
    TI_ERROR_IF(value_type != 1,
                "Vulkan native metric_reduce currently supports only f32.");
    if (!metric_reduce_f32) {
      metric_reduce_f32 = create_pipeline_from_spv(
          dev, kMetricReduceF32Spv, sizeof(kMetricReduceF32Spv),
          "vulkan_metric_reduce_f32");
    }
    return metric_reduce_f32.get();
  }

  VulkanReplayResourceSet<3> bind_resource_set(
      Program *program,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t values_bytes,
      DeviceAllocation other_alloc,
      uint64_t other_offset,
      size_t other_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(values_alloc, values_offset, values_bytes),
            rw_buffer_request(other_alloc, other_offset, other_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }
};

struct VulkanSparseAlgebraCache {
  Device *device{nullptr};
  std::uint64_t replay_generation{1};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> csr_spmv_f32;
  std::unique_ptr<Pipeline> bsr_spmv_f32;
  std::unique_ptr<Pipeline> sparse_axpy_f32;
  std::unique_ptr<Pipeline> sparse_diagonal_apply_f32;
  std::unique_ptr<Pipeline> sparse_diagonal_refresh_f32;
  std::unique_ptr<Pipeline> sparse_block_cholesky_refresh_f32;
  std::unique_ptr<Pipeline> sparse_block_cholesky_apply_f32;
  std::unique_ptr<Pipeline> sparse_dot_partial_f32;
  std::unique_ptr<Pipeline> sparse_sqrt_scalar_f32;
  std::unique_ptr<Pipeline> sparse_scalar_divide_f32;
  std::unique_ptr<Pipeline> sparse_cg_update_f32;
  std::unique_ptr<Pipeline> sparse_cg_direction_f32;
  std::unique_ptr<Pipeline> sparse_convergence_f32;
  std::unique_ptr<Pipeline> sparse_minres_scalar_f32;
  std::unique_ptr<Pipeline> sparse_minres_vector_state_f32;
  std::unique_ptr<Pipeline> sparse_minres_commit_f32;
  std::unique_ptr<Pipeline> sparse_bicgstab_scalar_f32;
  std::unique_ptr<Pipeline> sparse_bicgstab_direction_f32;
  std::unique_ptr<Pipeline> sparse_bicgstab_intermediate_f32;
  std::unique_ptr<Pipeline> sparse_bicgstab_commit_f32;
  std::unique_ptr<Pipeline> sparse_bicgstab_reconcile_f32;
  std::unique_ptr<Pipeline> sparse_gmres_multi_dot_partial_f32;
  std::unique_ptr<Pipeline> sparse_gmres_multi_dot_final_f32;
  std::unique_ptr<Pipeline> sparse_gmres_projection_f32;
  std::unique_ptr<Pipeline> sparse_gmres_basis_f32;
  std::unique_ptr<Pipeline> sparse_gmres_combine_f32;
  std::unique_ptr<Pipeline> sparse_gmres_scalar_f32;
  std::unique_ptr<Pipeline> sparse_triplet_pack_f32;
  std::unique_ptr<Pipeline> sparse_triplet_pack_packed_f32;
  std::unique_ptr<Pipeline> sparse_segment_flags_u64;
  std::unique_ptr<Pipeline> sparse_segment_scatter_u64;
  std::unique_ptr<Pipeline> sparse_segment_reduce_f32;
  std::unique_ptr<Pipeline> sparse_csr_finalize_u64;
  std::unique_ptr<Pipeline> sparse_assembly_finalize_control;
  VulkanResourceSetReplayRing<5> csr_spmv_bindings;
  VulkanResourceSetReplayRing<5> bsr_spmv_bindings;
  VulkanResourceSetReplayRing<2> sparse_axpy_bindings;
  VulkanResourceSetReplayRing<3> sparse_diagonal_apply_bindings;
  VulkanResourceSetReplayRing<4> sparse_diagonal_refresh_bindings;
  VulkanResourceSetReplayRing<4> sparse_block_cholesky_refresh_bindings;
  VulkanResourceSetReplayRing<3> sparse_block_cholesky_apply_bindings;
  VulkanResourceSetReplayRing<3> sparse_dot_partial_bindings;
  VulkanResourceSetReplayRing<1> sparse_sqrt_scalar_bindings;
  VulkanResourceSetReplayRing<4> sparse_scalar_divide_bindings;
  VulkanResourceSetReplayRing<5> sparse_cg_update_bindings;
  VulkanResourceSetReplayRing<3> sparse_cg_direction_bindings;
  VulkanResourceSetReplayRing<4> sparse_convergence_bindings;
  VulkanResourceSetReplayRing<4> sparse_minres_scalar_bindings;
  VulkanResourceSetReplayRing<3> sparse_minres_vector_state_bindings;
  VulkanResourceSetReplayRing<9> sparse_minres_commit_bindings;
  VulkanResourceSetReplayRing<7> sparse_triplet_pack_bindings;
  VulkanResourceSetReplayRing<5> sparse_bicgstab_scalar_bindings;
  VulkanResourceSetReplayRing<4> sparse_bicgstab_direction_bindings;
  VulkanResourceSetReplayRing<4> sparse_bicgstab_intermediate_bindings;
  VulkanResourceSetReplayRing<7> sparse_bicgstab_commit_bindings;
  VulkanResourceSetReplayRing<7> sparse_bicgstab_reconcile_bindings;
  VulkanResourceSetReplayRing<4> sparse_gmres_multi_dot_partial_bindings;
  VulkanResourceSetReplayRing<3> sparse_gmres_multi_dot_final_bindings;
  VulkanResourceSetReplayRing<5> sparse_gmres_projection_bindings;
  VulkanResourceSetReplayRing<4> sparse_gmres_basis_bindings;
  VulkanResourceSetReplayRing<4> sparse_gmres_combine_bindings;
  VulkanResourceSetReplayRing<10> sparse_gmres_scalar_bindings;
  VulkanResourceSetReplayRing<5> sparse_triplet_pack_packed_bindings;
  VulkanResourceSetReplayRing<4> sparse_segment_flags_bindings;
  VulkanResourceSetReplayRing<6> sparse_segment_scatter_bindings;
  VulkanResourceSetReplayRing<5> sparse_segment_reduce_bindings;
  VulkanResourceSetReplayRing<5> sparse_csr_finalize_bindings;
  VulkanResourceSetReplayRing<2> sparse_assembly_finalize_control_bindings;
  VulkanCommandReplayCache csr_spmv_command_replay;
  VulkanCommandReplayCache bsr_spmv_command_replay;
  VulkanCommandReplayCache sparse_axpy_command_replay;
  VulkanCommandReplayCache sparse_diagonal_apply_command_replay;
  VulkanCommandReplayCache sparse_diagonal_refresh_command_replay;
  VulkanCommandReplayCache sparse_block_cholesky_refresh_command_replay;
  VulkanCommandReplayCache sparse_block_cholesky_apply_command_replay;
  VulkanCommandReplayCache sparse_dot_partial_command_replay;
  VulkanCommandReplayCache sparse_sqrt_scalar_command_replay;
  VulkanCommandReplayCache sparse_scalar_divide_command_replay;
  VulkanCommandReplayCache sparse_cg_update_command_replay;
  VulkanCommandReplayCache sparse_cg_direction_command_replay;
  VulkanCommandReplayCache sparse_convergence_command_replay;
  VulkanCommandReplayCache sparse_minres_scalar_command_replay;
  VulkanCommandReplayCache sparse_minres_vector_state_command_replay;
  VulkanCommandReplayCache sparse_minres_commit_command_replay;
  VulkanCommandReplayCache sparse_triplet_pack_command_replay;
  VulkanCommandReplayCache sparse_bicgstab_scalar_command_replay;
  VulkanCommandReplayCache sparse_bicgstab_direction_command_replay;
  VulkanCommandReplayCache sparse_bicgstab_intermediate_command_replay;
  VulkanCommandReplayCache sparse_bicgstab_commit_command_replay;
  VulkanCommandReplayCache sparse_bicgstab_reconcile_command_replay;
  VulkanCommandReplayCache sparse_gmres_multi_dot_partial_command_replay;
  VulkanCommandReplayCache sparse_gmres_multi_dot_final_command_replay;
  VulkanCommandReplayCache sparse_gmres_projection_command_replay;
  VulkanCommandReplayCache sparse_gmres_basis_command_replay;
  VulkanCommandReplayCache sparse_gmres_combine_command_replay;
  VulkanCommandReplayCache sparse_gmres_scalar_command_replay;
  VulkanCommandReplayCache sparse_triplet_pack_packed_command_replay;
  VulkanCommandReplayCache sparse_segment_flags_command_replay;
  VulkanCommandReplayCache sparse_segment_scatter_command_replay;
  VulkanCommandReplayCache sparse_segment_reduce_command_replay;
  VulkanCommandReplayCache sparse_csr_finalize_command_replay;
  VulkanCommandReplayCache sparse_assembly_finalize_control_command_replay;

  void reset_resource_sets() {
    ++replay_generation;
    csr_spmv_bindings.reset();
    bsr_spmv_bindings.reset();
    sparse_axpy_bindings.reset();
    sparse_diagonal_apply_bindings.reset();
    sparse_diagonal_refresh_bindings.reset();
    sparse_block_cholesky_refresh_bindings.reset();
    sparse_block_cholesky_apply_bindings.reset();
    sparse_dot_partial_bindings.reset();
    sparse_sqrt_scalar_bindings.reset();
    sparse_scalar_divide_bindings.reset();
    sparse_cg_update_bindings.reset();
    sparse_cg_direction_bindings.reset();
    sparse_convergence_bindings.reset();
    sparse_minres_scalar_bindings.reset();
    sparse_minres_vector_state_bindings.reset();
    sparse_minres_commit_bindings.reset();
    sparse_triplet_pack_bindings.reset();
    sparse_bicgstab_scalar_bindings.reset();
    sparse_bicgstab_direction_bindings.reset();
    sparse_bicgstab_intermediate_bindings.reset();
    sparse_bicgstab_commit_bindings.reset();
    sparse_bicgstab_reconcile_bindings.reset();
    sparse_gmres_multi_dot_partial_bindings.reset();
    sparse_gmres_multi_dot_final_bindings.reset();
    sparse_gmres_projection_bindings.reset();
    sparse_gmres_basis_bindings.reset();
    sparse_gmres_combine_bindings.reset();
    sparse_gmres_scalar_bindings.reset();
    sparse_triplet_pack_packed_bindings.reset();
    sparse_segment_flags_bindings.reset();
    sparse_segment_scatter_bindings.reset();
    sparse_segment_reduce_bindings.reset();
    sparse_csr_finalize_bindings.reset();
    sparse_assembly_finalize_control_bindings.reset();
    csr_spmv_command_replay.reset();
    bsr_spmv_command_replay.reset();
    sparse_axpy_command_replay.reset();
    sparse_diagonal_apply_command_replay.reset();
    sparse_diagonal_refresh_command_replay.reset();
    sparse_block_cholesky_refresh_command_replay.reset();
    sparse_block_cholesky_apply_command_replay.reset();
    sparse_dot_partial_command_replay.reset();
    sparse_sqrt_scalar_command_replay.reset();
    sparse_scalar_divide_command_replay.reset();
    sparse_cg_update_command_replay.reset();
    sparse_cg_direction_command_replay.reset();
    sparse_convergence_command_replay.reset();
    sparse_minres_scalar_command_replay.reset();
    sparse_minres_vector_state_command_replay.reset();
    sparse_minres_commit_command_replay.reset();
    sparse_triplet_pack_command_replay.reset();
    sparse_bicgstab_scalar_command_replay.reset();
    sparse_bicgstab_direction_command_replay.reset();
    sparse_bicgstab_intermediate_command_replay.reset();
    sparse_bicgstab_commit_command_replay.reset();
    sparse_bicgstab_reconcile_command_replay.reset();
    sparse_gmres_multi_dot_partial_command_replay.reset();
    sparse_gmres_multi_dot_final_command_replay.reset();
    sparse_gmres_projection_command_replay.reset();
    sparse_gmres_basis_command_replay.reset();
    sparse_gmres_combine_command_replay.reset();
    sparse_gmres_scalar_command_replay.reset();
    sparse_triplet_pack_packed_command_replay.reset();
    sparse_segment_flags_command_replay.reset();
    sparse_segment_scatter_command_replay.reset();
    sparse_segment_reduce_command_replay.reset();
    sparse_csr_finalize_command_replay.reset();
    sparse_assembly_finalize_control_command_replay.reset();
  }

  void clear_allocs() {
    reset_resource_sets();
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanSparseAlgebraCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void reset_pipelines() {
    csr_spmv_f32.reset();
    bsr_spmv_f32.reset();
    sparse_axpy_f32.reset();
    sparse_diagonal_apply_f32.reset();
    sparse_diagonal_refresh_f32.reset();
    sparse_block_cholesky_refresh_f32.reset();
    sparse_block_cholesky_apply_f32.reset();
    sparse_dot_partial_f32.reset();
    sparse_sqrt_scalar_f32.reset();
    sparse_scalar_divide_f32.reset();
    sparse_cg_update_f32.reset();
    sparse_cg_direction_f32.reset();
    sparse_convergence_f32.reset();
    sparse_minres_scalar_f32.reset();
    sparse_minres_vector_state_f32.reset();
    sparse_minres_commit_f32.reset();
    sparse_bicgstab_scalar_f32.reset();
    sparse_bicgstab_direction_f32.reset();
    sparse_bicgstab_intermediate_f32.reset();
    sparse_bicgstab_commit_f32.reset();
    sparse_bicgstab_reconcile_f32.reset();
    sparse_gmres_multi_dot_partial_f32.reset();
    sparse_gmres_multi_dot_final_f32.reset();
    sparse_gmres_projection_f32.reset();
    sparse_gmres_basis_f32.reset();
    sparse_gmres_combine_f32.reset();
    sparse_gmres_scalar_f32.reset();
    sparse_triplet_pack_f32.reset();
    sparse_triplet_pack_packed_f32.reset();
    sparse_segment_flags_u64.reset();
    sparse_segment_scatter_u64.reset();
    sparse_segment_reduce_f32.reset();
    sparse_csr_finalize_u64.reset();
    sparse_assembly_finalize_control.reset();
    clear_allocs();
  }

 void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      reset_pipelines();
    }
    device = dev;
  }

  Pipeline *csr_spmv_pipeline(Device *dev) {
    ensure_device(dev);
    if (!csr_spmv_f32) {
      csr_spmv_f32 = create_pipeline_from_spv(
          dev, kCsrSpmvF32Spv, sizeof(kCsrSpmvF32Spv),
          "vulkan_csr_spmv_f32");
    }
    return csr_spmv_f32.get();
  }

  Pipeline *bsr_spmv_pipeline(Device *dev) {
    ensure_device(dev);
    if (!bsr_spmv_f32) {
      bsr_spmv_f32 = create_pipeline_from_spv(
          dev, kBsrSpmvF32Spv, sizeof(kBsrSpmvF32Spv),
          "vulkan_bsr_spmv_f32");
    }
    return bsr_spmv_f32.get();
  }

  Pipeline *sparse_axpy_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_axpy_f32) {
      sparse_axpy_f32 = create_pipeline_from_spv(
          dev, kSparseAxpyF32Spv, sizeof(kSparseAxpyF32Spv),
          "vulkan_sparse_axpy_f32");
    }
    return sparse_axpy_f32.get();
  }

  Pipeline *sparse_diagonal_apply_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_diagonal_apply_f32) {
      sparse_diagonal_apply_f32 = create_pipeline_from_spv(
          dev, kSparseDiagonalApplyF32Spv,
          sizeof(kSparseDiagonalApplyF32Spv),
          "vulkan_sparse_diagonal_apply_f32");
    }
    return sparse_diagonal_apply_f32.get();
  }

  Pipeline *sparse_diagonal_refresh_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_diagonal_refresh_f32) {
      sparse_diagonal_refresh_f32 = create_pipeline_from_spv(
          dev, kSparseDiagonalRefreshF32Spv,
          sizeof(kSparseDiagonalRefreshF32Spv),
          "vulkan_sparse_diagonal_refresh_f32");
    }
    return sparse_diagonal_refresh_f32.get();
  }

  Pipeline *sparse_block_cholesky_refresh_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_block_cholesky_refresh_f32) {
      sparse_block_cholesky_refresh_f32 = create_pipeline_from_spv(
          dev, kSparseBlockCholeskyRefreshF32Spv,
          sizeof(kSparseBlockCholeskyRefreshF32Spv),
          "vulkan_sparse_block_cholesky_refresh_f32");
    }
    return sparse_block_cholesky_refresh_f32.get();
  }

  Pipeline *sparse_block_cholesky_apply_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_block_cholesky_apply_f32) {
      sparse_block_cholesky_apply_f32 = create_pipeline_from_spv(
          dev, kSparseBlockCholeskyApplyF32Spv,
          sizeof(kSparseBlockCholeskyApplyF32Spv),
          "vulkan_sparse_block_cholesky_apply_f32");
    }
    return sparse_block_cholesky_apply_f32.get();
  }

  Pipeline *sparse_dot_partial_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_dot_partial_f32) {
      sparse_dot_partial_f32 = create_pipeline_from_spv(
          dev, kSparseDotPartialF32Spv, sizeof(kSparseDotPartialF32Spv),
          "vulkan_sparse_dot_partial_f32");
    }
    return sparse_dot_partial_f32.get();
  }

  Pipeline *sparse_sqrt_scalar_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_sqrt_scalar_f32) {
      sparse_sqrt_scalar_f32 = create_pipeline_from_spv(
          dev, kSparseSqrtScalarF32Spv, sizeof(kSparseSqrtScalarF32Spv),
          "vulkan_sparse_sqrt_scalar_f32");
    }
    return sparse_sqrt_scalar_f32.get();
  }

  Pipeline *sparse_scalar_divide_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_scalar_divide_f32) {
      sparse_scalar_divide_f32 = create_pipeline_from_spv(
          dev, kSparseScalarDivideF32Spv,
          sizeof(kSparseScalarDivideF32Spv),
          "vulkan_sparse_scalar_divide_f32");
    }
    return sparse_scalar_divide_f32.get();
  }

  Pipeline *sparse_cg_update_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_cg_update_f32) {
      sparse_cg_update_f32 = create_pipeline_from_spv(
          dev, kSparseCgUpdateF32Spv, sizeof(kSparseCgUpdateF32Spv),
          "vulkan_sparse_cg_update_f32");
    }
    return sparse_cg_update_f32.get();
  }

  Pipeline *sparse_cg_direction_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_cg_direction_f32) {
      sparse_cg_direction_f32 = create_pipeline_from_spv(
          dev, kSparseCgDirectionF32Spv,
          sizeof(kSparseCgDirectionF32Spv),
          "vulkan_sparse_cg_direction_f32");
    }
    return sparse_cg_direction_f32.get();
  }

  Pipeline *sparse_convergence_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_convergence_f32) {
      sparse_convergence_f32 = create_pipeline_from_spv(
          dev, kSparseConvergenceF32Spv, sizeof(kSparseConvergenceF32Spv),
          "vulkan_sparse_convergence_f32");
    }
    return sparse_convergence_f32.get();
  }

  Pipeline *sparse_minres_scalar_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_minres_scalar_f32) {
      sparse_minres_scalar_f32 = create_pipeline_from_spv(
          dev, kSparseMinresScalarF32Spv,
          sizeof(kSparseMinresScalarF32Spv),
          "vulkan_sparse_minres_scalar_f32");
    }
    return sparse_minres_scalar_f32.get();
  }

  Pipeline *sparse_minres_vector_state_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_minres_vector_state_f32) {
      sparse_minres_vector_state_f32 = create_pipeline_from_spv(
          dev, kSparseMinresVectorStateF32Spv,
          sizeof(kSparseMinresVectorStateF32Spv),
          "vulkan_sparse_minres_vector_state_f32");
    }
    return sparse_minres_vector_state_f32.get();
  }

  Pipeline *sparse_minres_commit_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_minres_commit_f32) {
      sparse_minres_commit_f32 = create_pipeline_from_spv(
          dev, kSparseMinresCommitF32Spv,
          sizeof(kSparseMinresCommitF32Spv),
          "vulkan_sparse_minres_commit_f32");
    }
    return sparse_minres_commit_f32.get();
  }

  Pipeline *sparse_bicgstab_scalar_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_bicgstab_scalar_f32) {
      sparse_bicgstab_scalar_f32 = create_pipeline_from_spv(
          dev, kSparseBicgstabScalarF32Spv,
          sizeof(kSparseBicgstabScalarF32Spv),
          "vulkan_sparse_bicgstab_scalar_f32");
    }
    return sparse_bicgstab_scalar_f32.get();
  }
  Pipeline *sparse_bicgstab_direction_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_bicgstab_direction_f32) {
      sparse_bicgstab_direction_f32 = create_pipeline_from_spv(
          dev, kSparseBicgstabDirectionF32Spv,
          sizeof(kSparseBicgstabDirectionF32Spv),
          "vulkan_sparse_bicgstab_direction_f32");
    }
    return sparse_bicgstab_direction_f32.get();
  }
  Pipeline *sparse_bicgstab_intermediate_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_bicgstab_intermediate_f32) {
      sparse_bicgstab_intermediate_f32 = create_pipeline_from_spv(
          dev, kSparseBicgstabIntermediateF32Spv,
          sizeof(kSparseBicgstabIntermediateF32Spv),
          "vulkan_sparse_bicgstab_intermediate_f32");
    }
    return sparse_bicgstab_intermediate_f32.get();
  }
  Pipeline *sparse_bicgstab_commit_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_bicgstab_commit_f32) {
      sparse_bicgstab_commit_f32 = create_pipeline_from_spv(
          dev, kSparseBicgstabCommitF32Spv,
          sizeof(kSparseBicgstabCommitF32Spv),
          "vulkan_sparse_bicgstab_commit_f32");
    }
    return sparse_bicgstab_commit_f32.get();
  }
  Pipeline *sparse_bicgstab_reconcile_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_bicgstab_reconcile_f32) {
      sparse_bicgstab_reconcile_f32 = create_pipeline_from_spv(
          dev, kSparseBicgstabReconcileF32Spv,
          sizeof(kSparseBicgstabReconcileF32Spv),
          "vulkan_sparse_bicgstab_reconcile_f32");
    }
    return sparse_bicgstab_reconcile_f32.get();
  }

  Pipeline *sparse_gmres_multi_dot_partial_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_multi_dot_partial_f32) {
      sparse_gmres_multi_dot_partial_f32 = create_pipeline_from_spv(
          dev, kSparseGmresMultiDotPartialF32Spv,
          sizeof(kSparseGmresMultiDotPartialF32Spv),
          "vulkan_sparse_gmres_multi_dot_partial_f32");
    }
    return sparse_gmres_multi_dot_partial_f32.get();
  }
  Pipeline *sparse_gmres_multi_dot_final_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_multi_dot_final_f32) {
      sparse_gmres_multi_dot_final_f32 = create_pipeline_from_spv(
          dev, kSparseGmresMultiDotFinalF32Spv,
          sizeof(kSparseGmresMultiDotFinalF32Spv),
          "vulkan_sparse_gmres_multi_dot_final_f32");
    }
    return sparse_gmres_multi_dot_final_f32.get();
  }
  Pipeline *sparse_gmres_projection_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_projection_f32) {
      sparse_gmres_projection_f32 = create_pipeline_from_spv(
          dev, kSparseGmresProjectionF32Spv,
          sizeof(kSparseGmresProjectionF32Spv),
          "vulkan_sparse_gmres_projection_f32");
    }
    return sparse_gmres_projection_f32.get();
  }
  Pipeline *sparse_gmres_basis_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_basis_f32) {
      sparse_gmres_basis_f32 = create_pipeline_from_spv(
          dev, kSparseGmresBasisF32Spv, sizeof(kSparseGmresBasisF32Spv),
          "vulkan_sparse_gmres_basis_f32");
    }
    return sparse_gmres_basis_f32.get();
  }
  Pipeline *sparse_gmres_combine_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_combine_f32) {
      sparse_gmres_combine_f32 = create_pipeline_from_spv(
          dev, kSparseGmresCombineF32Spv,
          sizeof(kSparseGmresCombineF32Spv),
          "vulkan_sparse_gmres_combine_f32");
    }
    return sparse_gmres_combine_f32.get();
  }
  Pipeline *sparse_gmres_scalar_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_gmres_scalar_f32) {
      sparse_gmres_scalar_f32 = create_pipeline_from_spv(
          dev, kSparseGmresScalarF32Spv,
          sizeof(kSparseGmresScalarF32Spv),
          "vulkan_sparse_gmres_scalar_f32");
    }
    return sparse_gmres_scalar_f32.get();
  }

  Pipeline *sparse_triplet_pack_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_triplet_pack_f32) {
      sparse_triplet_pack_f32 = create_pipeline_from_spv(
          dev, kSparseTripletPackF32Spv, sizeof(kSparseTripletPackF32Spv),
          "vulkan_sparse_triplet_pack_f32");
    }
    return sparse_triplet_pack_f32.get();
  }

  Pipeline *sparse_triplet_pack_packed_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_triplet_pack_packed_f32) {
      sparse_triplet_pack_packed_f32 = create_pipeline_from_spv(
          dev, kSparseTripletPackPackedF32Spv,
          sizeof(kSparseTripletPackPackedF32Spv),
          "vulkan_sparse_triplet_pack_packed_f32");
    }
    return sparse_triplet_pack_packed_f32.get();
  }

  Pipeline *sparse_segment_flags_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_segment_flags_u64) {
      sparse_segment_flags_u64 = create_pipeline_from_spv(
          dev, kSparseSegmentFlagsU64Spv,
          sizeof(kSparseSegmentFlagsU64Spv),
          "vulkan_sparse_segment_flags_u64");
    }
    return sparse_segment_flags_u64.get();
  }

  Pipeline *sparse_segment_scatter_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_segment_scatter_u64) {
      sparse_segment_scatter_u64 = create_pipeline_from_spv(
          dev, kSparseSegmentScatterU64Spv,
          sizeof(kSparseSegmentScatterU64Spv),
          "vulkan_sparse_segment_scatter_u64");
    }
    return sparse_segment_scatter_u64.get();
  }

  Pipeline *sparse_segment_reduce_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_segment_reduce_f32) {
      sparse_segment_reduce_f32 = create_pipeline_from_spv(
          dev, kSparseSegmentReduceF32Spv,
          sizeof(kSparseSegmentReduceF32Spv),
          "vulkan_sparse_segment_reduce_f32");
    }
    return sparse_segment_reduce_f32.get();
  }

  Pipeline *sparse_csr_finalize_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_csr_finalize_u64) {
      sparse_csr_finalize_u64 = create_pipeline_from_spv(
          dev, kSparseCsrFinalizeU64Spv,
          sizeof(kSparseCsrFinalizeU64Spv),
          "vulkan_sparse_csr_finalize_u64");
    }
    return sparse_csr_finalize_u64.get();
  }

  Pipeline *sparse_assembly_finalize_control_pipeline(Device *dev) {
    ensure_device(dev);
    if (!sparse_assembly_finalize_control) {
      sparse_assembly_finalize_control = create_pipeline_from_spv(
          dev, kSparseAssemblyFinalizeControlSpv,
          sizeof(kSparseAssemblyFinalizeControlSpv),
          "vulkan_sparse_assembly_finalize_control");
    }
    return sparse_assembly_finalize_control.get();
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan sparse algebra workspace: "
                "RhiResult({})",
                res);
    return alloc;
  }

  bool needs_partial_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  bool has_workspace_allocs() const {
    return partial != kDeviceNullAllocation;
  }

  void clear_partial_alloc() {
    ++replay_generation;
    sparse_dot_partial_bindings.reset();
    sparse_dot_partial_command_replay.reset();
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  void ensure_partial(size_t bytes) {
    if (bytes == 0 || !needs_partial_realloc(bytes)) {
      cached_bytes = partial_capacity;
      return;
    }
    clear_partial_alloc();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = partial_capacity;
  }

  VulkanReplayResourceSet<5> bind_csr_spmv(
      Program *program,
      DeviceAllocation row_offsets,
      size_t row_offsets_bytes,
      DeviceAllocation column_indices,
      size_t column_indices_bytes,
      DeviceAllocation values,
      size_t values_bytes,
      DeviceAllocation x,
      size_t x_bytes,
      DeviceAllocation y,
      size_t y_bytes) {
    return csr_spmv_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(row_offsets, 0, row_offsets_bytes),
            rw_buffer_request(column_indices, 0, column_indices_bytes),
            rw_buffer_request(values, 0, values_bytes),
            rw_buffer_request(x, 0, x_bytes),
            rw_buffer_request(y, 0, y_bytes)});
  }

  VulkanReplayResourceSet<5> bind_bsr_spmv(
      Program *program,
      DeviceAllocation row_offsets,
      size_t row_offsets_bytes,
      DeviceAllocation column_indices,
      size_t column_indices_bytes,
      DeviceAllocation values,
      size_t values_bytes,
      DeviceAllocation x,
      size_t x_bytes,
      DeviceAllocation y,
      size_t y_bytes) {
    return bsr_spmv_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(row_offsets, 0, row_offsets_bytes),
            rw_buffer_request(column_indices, 0, column_indices_bytes),
            rw_buffer_request(values, 0, values_bytes),
            rw_buffer_request(x, 0, x_bytes),
            rw_buffer_request(y, 0, y_bytes)});
  }

  VulkanReplayResourceSet<2> bind_sparse_axpy(Program *program,
                                               DeviceAllocation x,
                                               size_t x_bytes,
                                               DeviceAllocation y,
                                               size_t y_bytes) {
    return sparse_axpy_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(x, 0, x_bytes),
            rw_buffer_request(y, 0, y_bytes)});
  }

  VulkanReplayResourceSet<3> bind_sparse_diagonal_apply(
      Program *program,
      DeviceAllocation inverse_diagonal,
      size_t inverse_bytes,
      DeviceAllocation input,
      size_t input_bytes,
      DeviceAllocation output,
      size_t output_bytes) {
    return sparse_diagonal_apply_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(inverse_diagonal, 0, inverse_bytes),
            rw_buffer_request(input, 0, input_bytes),
            rw_buffer_request(output, 0, output_bytes)});
  }

  VulkanReplayResourceSet<4> bind_sparse_diagonal_refresh(
      Program *program,
      DeviceAllocation values,
      size_t values_bytes,
      DeviceAllocation diagonal_offsets,
      size_t offsets_bytes,
      DeviceAllocation staging_inverse,
      size_t inverse_bytes,
      DeviceAllocation status,
      size_t status_bytes) {
    return sparse_diagonal_refresh_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(values, 0, values_bytes),
            rw_buffer_request(diagonal_offsets, 0, offsets_bytes),
            rw_buffer_request(staging_inverse, 0, inverse_bytes),
            rw_buffer_request(status, 0, status_bytes)});
  }

  VulkanReplayResourceSet<4> bind_sparse_block_cholesky_refresh(
      Program *program,
      DeviceAllocation values,
      size_t values_bytes,
      DeviceAllocation diagonal_block_offsets,
      size_t offsets_bytes,
      DeviceAllocation staging_factors,
      size_t factor_bytes,
      DeviceAllocation status,
      size_t status_bytes) {
    return sparse_block_cholesky_refresh_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(values, 0, values_bytes),
            rw_buffer_request(diagonal_block_offsets, 0, offsets_bytes),
            rw_buffer_request(staging_factors, 0, factor_bytes),
            rw_buffer_request(status, 0, status_bytes)});
  }

  VulkanReplayResourceSet<3> bind_sparse_block_cholesky_apply(
      Program *program,
      DeviceAllocation factor_blocks,
      size_t factor_bytes,
      DeviceAllocation input,
      size_t input_bytes,
      DeviceAllocation output,
      size_t output_bytes) {
    return sparse_block_cholesky_apply_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(factor_blocks, 0, factor_bytes),
            rw_buffer_request(input, 0, input_bytes),
            rw_buffer_request(output, 0, output_bytes)});
  }

  VulkanReplayResourceSet<3> bind_sparse_dot_partial(
      Program *program,
      DeviceAllocation x,
      size_t x_bytes,
      DeviceAllocation y,
      size_t y_bytes,
      DeviceAllocation partial_alloc,
      size_t partial_bytes) {
    return sparse_dot_partial_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(x, 0, x_bytes),
            rw_buffer_request(y, 0, y_bytes),
            rw_buffer_request(partial_alloc, 0, partial_bytes)});
  }

  VulkanReplayResourceSet<1> bind_sparse_sqrt_scalar(
      Program *program,
      DeviceAllocation value,
      size_t value_bytes) {
    return sparse_sqrt_scalar_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 1>{
            rw_buffer_request(value, 0, value_bytes)});
  }

  VulkanReplayResourceSet<4> bind_sparse_scalar_divide(
      Program *program,
      DeviceAllocation numerator,
      DeviceAllocation denominator,
      DeviceAllocation quotient,
      DeviceAllocation status,
      size_t scalar_bytes) {
    return sparse_scalar_divide_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(numerator, 0, scalar_bytes),
            rw_buffer_request(denominator, 0, scalar_bytes),
            rw_buffer_request(quotient, 0, scalar_bytes),
            rw_buffer_request(status, 0, sizeof(int32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_cg_update(
      Program *program,
      DeviceAllocation direction,
      DeviceAllocation applied_direction,
      DeviceAllocation alpha,
      DeviceAllocation solution,
      DeviceAllocation residual,
      size_t vector_bytes) {
    return sparse_cg_update_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(direction, 0, vector_bytes),
            rw_buffer_request(applied_direction, 0, vector_bytes),
            rw_buffer_request(alpha, 0, sizeof(float32)),
            rw_buffer_request(solution, 0, vector_bytes),
            rw_buffer_request(residual, 0, vector_bytes)});
  }

  VulkanReplayResourceSet<3> bind_sparse_cg_direction(
      Program *program,
      DeviceAllocation residual,
      DeviceAllocation beta,
      DeviceAllocation direction,
      size_t vector_bytes) {
    return sparse_cg_direction_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(residual, 0, vector_bytes),
            rw_buffer_request(beta, 0, sizeof(float32)),
            rw_buffer_request(direction, 0, vector_bytes)});
  }

  VulkanReplayResourceSet<4> bind_sparse_convergence(
      Program *program,
      DeviceAllocation residual_squared,
      DeviceAllocation status,
      DeviceAllocation completed_iterations,
      DeviceAllocation rhs_squared) {
    return sparse_convergence_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(residual_squared, 0, sizeof(float32)),
            rw_buffer_request(status, 0, sizeof(int32_t)),
            rw_buffer_request(completed_iterations, 0, sizeof(int32_t)),
            rw_buffer_request(rhs_squared, 0, sizeof(float32))});
  }

  VulkanReplayResourceSet<4> bind_sparse_minres_scalar(
      Program *program,
      DeviceAllocation initial_residual_squared,
      DeviceAllocation rhs_squared,
      DeviceAllocation dot,
      DeviceAllocation state) {
    return sparse_minres_scalar_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(initial_residual_squared, 0, sizeof(float32)),
            rw_buffer_request(rhs_squared, 0, sizeof(float32)),
            rw_buffer_request(dot, 0, sizeof(float32)),
            rw_buffer_request(state, 0, 33 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<3> bind_sparse_minres_vector_state(
      Program *program,
      DeviceAllocation source,
      DeviceAllocation destination,
      DeviceAllocation state,
      size_t vector_bytes) {
    return sparse_minres_vector_state_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(source, 0, vector_bytes),
            rw_buffer_request(destination, 0, vector_bytes),
            rw_buffer_request(state, 0, 33 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<9> bind_sparse_minres_commit(
      Program *program,
      const std::array<DeviceAllocation, 9> &allocs,
      size_t vector_bytes) {
    return sparse_minres_commit_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 9>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, vector_bytes),
            rw_buffer_request(allocs[4], 0, vector_bytes),
            rw_buffer_request(allocs[5], 0, vector_bytes),
            rw_buffer_request(allocs[6], 0, vector_bytes),
            rw_buffer_request(allocs[7], 0, vector_bytes),
            rw_buffer_request(allocs[8], 0, 33 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_bicgstab_scalar(
      Program *program,
      const std::array<DeviceAllocation, 5> &allocs) {
    return sparse_bicgstab_scalar_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(allocs[0], 0, sizeof(float32)),
            rw_buffer_request(allocs[1], 0, sizeof(float32)),
            rw_buffer_request(allocs[2], 0, sizeof(float32)),
            rw_buffer_request(allocs[3], 0, sizeof(float32)),
            rw_buffer_request(allocs[4], 0, 24 * sizeof(uint32_t))});
  }
  VulkanReplayResourceSet<4> bind_sparse_bicgstab_direction(
      Program *program,
      const std::array<DeviceAllocation, 4> &allocs,
      size_t vector_bytes) {
    return sparse_bicgstab_direction_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, 24 * sizeof(uint32_t))});
  }
  VulkanReplayResourceSet<4> bind_sparse_bicgstab_intermediate(
      Program *program,
      const std::array<DeviceAllocation, 4> &allocs,
      size_t vector_bytes) {
    return sparse_bicgstab_intermediate_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, 24 * sizeof(uint32_t))});
  }
  VulkanReplayResourceSet<7> bind_sparse_bicgstab_commit(
      Program *program,
      const std::array<DeviceAllocation, 7> &allocs,
      size_t vector_bytes) {
    return sparse_bicgstab_commit_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 7>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, vector_bytes),
            rw_buffer_request(allocs[4], 0, vector_bytes),
            rw_buffer_request(allocs[5], 0, vector_bytes),
            rw_buffer_request(allocs[6], 0, 24 * sizeof(uint32_t))});
  }
  VulkanReplayResourceSet<7> bind_sparse_bicgstab_reconcile(
      Program *program,
      const std::array<DeviceAllocation, 7> &allocs,
      size_t vector_bytes) {
    return sparse_bicgstab_reconcile_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 7>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, vector_bytes),
            rw_buffer_request(allocs[4], 0, vector_bytes),
            rw_buffer_request(allocs[5], 0, vector_bytes),
            rw_buffer_request(allocs[6], 0, 24 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<4> bind_sparse_gmres_multi_dot_partial(
      Program *program,
      const std::array<DeviceAllocation, 4> &allocs,
      size_t basis_bytes,
      size_t vector_bytes,
      size_t partial_bytes) {
    return sparse_gmres_multi_dot_partial_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(allocs[0], 0, basis_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, partial_bytes),
            rw_buffer_request(allocs[3], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<3> bind_sparse_gmres_multi_dot_final(
      Program *program,
      const std::array<DeviceAllocation, 3> &allocs,
      size_t partial_bytes,
      size_t projection_bytes) {
    return sparse_gmres_multi_dot_final_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(allocs[0], 0, partial_bytes),
            rw_buffer_request(allocs[1], 0, projection_bytes),
            rw_buffer_request(allocs[2], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_gmres_projection(
      Program *program,
      const std::array<DeviceAllocation, 5> &allocs,
      size_t basis_bytes,
      size_t vector_bytes,
      size_t projection_bytes,
      size_t hessenberg_bytes) {
    return sparse_gmres_projection_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(allocs[0], 0, basis_bytes),
            rw_buffer_request(allocs[1], 0, vector_bytes),
            rw_buffer_request(allocs[2], 0, projection_bytes),
            rw_buffer_request(allocs[3], 0, hessenberg_bytes),
            rw_buffer_request(allocs[4], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<4> bind_sparse_gmres_basis(
      Program *program,
      const std::array<DeviceAllocation, 4> &allocs,
      size_t vector_bytes,
      size_t basis_bytes) {
    return sparse_gmres_basis_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(allocs[0], 0, vector_bytes),
            rw_buffer_request(allocs[1], 0, basis_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<4> bind_sparse_gmres_combine(
      Program *program,
      const std::array<DeviceAllocation, 4> &allocs,
      size_t basis_bytes,
      size_t coefficient_bytes,
      size_t vector_bytes) {
    return sparse_gmres_combine_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(allocs[0], 0, basis_bytes),
            rw_buffer_request(allocs[1], 0, coefficient_bytes),
            rw_buffer_request(allocs[2], 0, vector_bytes),
            rw_buffer_request(allocs[3], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<10> bind_sparse_gmres_scalar(
      Program *program,
      const std::array<DeviceAllocation, 10> &allocs,
      size_t hessenberg_bytes,
      size_t restart_bytes,
      size_t g_bytes) {
    return sparse_gmres_scalar_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 10>{
            rw_buffer_request(allocs[0], 0, sizeof(float32)),
            rw_buffer_request(allocs[1], 0, sizeof(float32)),
            rw_buffer_request(allocs[2], 0, sizeof(float32)),
            rw_buffer_request(allocs[3], 0, sizeof(float32)),
            rw_buffer_request(allocs[4], 0, hessenberg_bytes),
            rw_buffer_request(allocs[5], 0, restart_bytes),
            rw_buffer_request(allocs[6], 0, restart_bytes),
            rw_buffer_request(allocs[7], 0, g_bytes),
            rw_buffer_request(allocs[8], 0, restart_bytes),
            rw_buffer_request(allocs[9], 0, 32 * sizeof(uint32_t))});
  }

  VulkanReplayResourceSet<7> bind_sparse_triplet_pack(
      Program *program,
      DeviceAllocation rows,
      DeviceAllocation columns,
      DeviceAllocation values,
      DeviceAllocation keys,
      DeviceAllocation sorted_values,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity) {
    return sparse_triplet_pack_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 7>{
            rw_buffer_request(rows, 0, capacity * sizeof(int32_t)),
            rw_buffer_request(columns, 0, capacity * sizeof(int32_t)),
            rw_buffer_request(values, 0, capacity * sizeof(float32)),
            rw_buffer_request(keys, 0, capacity * sizeof(uint64_t)),
            rw_buffer_request(sorted_values, 0,
                              capacity * sizeof(float32)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_triplet_pack_packed(
      Program *program,
      DeviceAllocation packed_triplets,
      DeviceAllocation keys,
      DeviceAllocation sorted_values,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity) {
    return sparse_triplet_pack_packed_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(packed_triplets, 0,
                              (capacity * 3 + 2) * sizeof(int32_t)),
            rw_buffer_request(keys, 0, capacity * sizeof(uint64_t)),
            rw_buffer_request(sorted_values, 0,
                              capacity * sizeof(float32)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<4> bind_sparse_segment_flags(
      Program *program,
      DeviceAllocation keys,
      DeviceAllocation segment_ids,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity) {
    return sparse_segment_flags_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(keys, 0, capacity * sizeof(uint64_t)),
            rw_buffer_request(segment_ids, 0,
                              capacity * sizeof(int32_t)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<6> bind_sparse_segment_scatter(
      Program *program,
      DeviceAllocation keys,
      DeviceAllocation segment_ids,
      DeviceAllocation unique_keys,
      DeviceAllocation segment_offsets,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity) {
    return sparse_segment_scatter_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 6>{
            rw_buffer_request(keys, 0, capacity * sizeof(uint64_t)),
            rw_buffer_request(segment_ids, 0,
                              capacity * sizeof(int32_t)),
            rw_buffer_request(unique_keys, 0,
                              capacity * sizeof(uint64_t)),
            rw_buffer_request(segment_offsets, 0,
                              (capacity + 1) * sizeof(int32_t)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_segment_reduce(
      Program *program,
      DeviceAllocation sorted_values,
      DeviceAllocation segment_offsets,
      DeviceAllocation unique_values,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity) {
    return sparse_segment_reduce_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(sorted_values, 0,
                              capacity * sizeof(float32)),
            rw_buffer_request(segment_offsets, 0,
                              (capacity + 1) * sizeof(int32_t)),
            rw_buffer_request(unique_values, 0,
                              capacity * sizeof(float32)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<5> bind_sparse_csr_finalize(
      Program *program,
      DeviceAllocation unique_keys,
      DeviceAllocation column_indices,
      DeviceAllocation row_offsets,
      DeviceAllocation active_count,
      DeviceAllocation control,
      size_t capacity,
      size_t rows) {
    return sparse_csr_finalize_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(unique_keys, 0,
                              capacity * sizeof(uint64_t)),
            rw_buffer_request(column_indices, 0,
                              capacity * sizeof(int32_t)),
            rw_buffer_request(row_offsets, 0,
                              (rows + 1) * sizeof(int32_t)),
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }

  VulkanReplayResourceSet<2> bind_sparse_assembly_finalize_control(
      Program *program,
      DeviceAllocation active_count,
      DeviceAllocation control) {
    return sparse_assembly_finalize_control_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(active_count, 0, sizeof(int32_t)),
            rw_buffer_request(control, 0, 2 * sizeof(int32_t))});
  }
};

struct VulkanTransformCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  std::unique_ptr<Pipeline> transform_i32_affine_dense;
  std::unique_ptr<Pipeline> transform_i32_affine;
  std::unique_ptr<Pipeline> transform_f32_affine;
  std::unique_ptr<Pipeline> transform_indexed_i32_affine;
  std::unique_ptr<Pipeline> transform_indexed_f32_affine;
  std::unique_ptr<Pipeline> transform_u64_affine;
  std::unique_ptr<Pipeline> transform_f64_affine;
  std::unique_ptr<ShaderResourceSet> dense_i32_affine_bindings;
  std::unique_ptr<ShaderResourceSet> affine_bindings;
  std::unique_ptr<ShaderResourceSet> indexed_affine_bindings;
  VulkanRwBufferReplay<2> dense_i32_affine_replay;
  VulkanRwBufferReplay<2> affine_replay;
  VulkanRwBufferReplay<3> indexed_affine_replay;
  VulkanCommandReplayCache dense_i32_affine_command_replay;
  VulkanCommandReplayCache affine_command_replay;
  VulkanCommandReplayCache indexed_affine_command_replay;

  void clear_allocs() {
    dense_i32_affine_command_replay.reset();
    affine_command_replay.reset();
    indexed_affine_command_replay.reset();
    dense_i32_affine_bindings.reset();
    affine_bindings.reset();
    indexed_affine_bindings.reset();
    dense_i32_affine_replay.reset();
    affine_replay.reset();
    indexed_affine_replay.reset();
    cached_bytes = 0;
  }

  ~VulkanTransformCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      transform_i32_affine_dense.reset();
      transform_i32_affine.reset();
      transform_f32_affine.reset();
      transform_indexed_i32_affine.reset();
      transform_indexed_f32_affine.reset();
      transform_u64_affine.reset();
      transform_f64_affine.reset();
      affine_bindings.reset();
      indexed_affine_bindings.reset();
    }
    device = dev;
  }

  Pipeline *dense_i32_pipeline(Device *dev) {
    ensure_pipelines(dev);
    if (!transform_i32_affine_dense) {
      transform_i32_affine_dense =
          create_pipeline(dev, kTransformI32AffineDenseSpv,
                          "vulkan_transform_i32_affine_dense");
    }
    return transform_i32_affine_dense.get();
  }

  Pipeline *pipeline_for(Device *dev, int value_type, bool has_float64) {
    ensure_pipelines(dev);
    if (value_type == 0 || value_type == 2) {
      if (!transform_i32_affine) {
        transform_i32_affine = create_pipeline(
            dev, kTransformI32AffineSpv, "vulkan_transform_i32_affine");
      }
      return transform_i32_affine.get();
    }
    if (value_type == 1) {
      if (!transform_f32_affine) {
        transform_f32_affine = create_pipeline(
            dev, kTransformF32AffineSpv, "vulkan_transform_f32_affine");
      }
      return transform_f32_affine.get();
    }
    if (value_type == 3 || value_type == 4) {
      if (!transform_u64_affine) {
        transform_u64_affine = create_pipeline(
            dev, kTransformU64AffineSpv, "vulkan_transform_u64_affine");
      }
      return transform_u64_affine.get();
    }
    if (value_type == 5) {
      TI_ERROR_IF(!has_float64,
                  "Vulkan native f64 transform requires shader Float64 "
                  "device capability.");
      if (!transform_f64_affine) {
        transform_f64_affine = create_pipeline(
            dev, kTransformF64AffineSpv, "vulkan_transform_f64_affine");
      }
      return transform_f64_affine.get();
    }
    TI_ERROR("Unsupported Vulkan transform value type.");
  }

  Pipeline *indexed_pipeline_for(Device *dev, int value_type) {
    ensure_pipelines(dev);
    if (value_type == 0 || value_type == 2) {
      if (!transform_indexed_i32_affine) {
        transform_indexed_i32_affine = create_pipeline(
            dev, kTransformIndexedI32AffineSpv,
            "vulkan_transform_indexed_i32_affine");
      }
      return transform_indexed_i32_affine.get();
    }
    if (value_type == 1) {
      if (!transform_indexed_f32_affine) {
        transform_indexed_f32_affine = create_pipeline(
            dev, kTransformIndexedF32AffineSpv,
            "vulkan_transform_indexed_f32_affine");
      }
      return transform_indexed_f32_affine.get();
    }
    TI_ERROR("Unsupported Vulkan indexed transform value type.");
  }

  ShaderResourceSet *cached_affine_resource_set() {
    if (!affine_bindings) {
      affine_bindings.reset(device->create_resource_set());
    }
    return affine_bindings.get();
  }

  ShaderResourceSet *cached_dense_i32_affine_resource_set() {
    if (!dense_i32_affine_bindings) {
      dense_i32_affine_bindings.reset(device->create_resource_set());
    }
    return dense_i32_affine_bindings.get();
  }

  ShaderResourceSet *cached_indexed_affine_resource_set() {
    if (!indexed_affine_bindings) {
      indexed_affine_bindings.reset(device->create_resource_set());
    }
    return indexed_affine_bindings.get();
  }
};

struct VulkanAddMergeCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  std::array<std::unique_ptr<Pipeline>, 6> pipelines;
  std::unique_ptr<ShaderResourceSet> bindings;
  VulkanRwBufferReplay<2> binding_replay;
  VulkanCommandReplayCache command_replay;

  void clear_allocs() {
    command_replay.reset();
    bindings.reset();
    binding_replay.reset();
    cached_bytes = 0;
  }

  ~VulkanAddMergeCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void ensure_device(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      for (auto &pipeline : pipelines) {
        pipeline.reset();
      }
    }
    device = dev;
  }

  Pipeline *pipeline_for(Device *dev, int value_type, bool has_float64) {
    ensure_device(dev);
    TI_ERROR_IF(value_type < 0 || value_type > 5,
                "Unsupported Vulkan add-merge value type.");
    if (value_type == 5) {
      TI_ERROR_IF(!has_float64,
                  "Vulkan f64 add-merge requires shader Float64 device "
                  "capability.");
    }
    auto &pipeline = pipelines[value_type];
    if (pipeline) {
      return pipeline.get();
    }
    switch (value_type) {
      case 1:
        pipeline =
            create_pipeline(dev, kAddMergeF32Spv, "vulkan_add_merge_f32");
        break;
      case 2:
        pipeline =
            create_pipeline(dev, kAddMergeU32Spv, "vulkan_add_merge_u32");
        break;
      case 3:
        pipeline =
            create_pipeline(dev, kAddMergeU64Spv, "vulkan_add_merge_u64");
        break;
      case 4:
        pipeline =
            create_pipeline(dev, kAddMergeI64Spv, "vulkan_add_merge_i64");
        break;
      case 5:
        pipeline =
            create_pipeline(dev, kAddMergeF64Spv, "vulkan_add_merge_f64");
        break;
      default:
        pipeline =
            create_pipeline(dev, kAddMergeI32Spv, "vulkan_add_merge_i32");
        break;
    }
    return pipeline.get();
  }

  ShaderResourceSet *cached_resource_set() {
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }
};

struct VulkanIndexedCopyCache {
  Device *device{nullptr};
  size_t cached_bytes{0};
  std::unique_ptr<Pipeline> gather_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_dense_u32_by_i32;
  std::unique_ptr<Pipeline> gather_strided_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_strided_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_i32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_f32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_u32_by_i32;
  std::unique_ptr<Pipeline> scatter_add_u64_by_i32;
  std::unique_ptr<Pipeline> scatter_add_i64_by_i32;
  std::unique_ptr<Pipeline> scatter_add_f64_by_i32;
  std::array<std::unique_ptr<Pipeline>, 6> scatter_add_strided;
  std::array<std::unique_ptr<Pipeline>, 6> scatter_add_packed;
  std::unique_ptr<ShaderResourceSet> gather_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_bindings;
  std::unique_ptr<ShaderResourceSet> gather_strided_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_strided_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_i32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_f32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_u32_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_u64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_i64_bindings;
  std::unique_ptr<ShaderResourceSet> scatter_add_f64_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6> scatter_add_strided_bindings;
  std::array<std::unique_ptr<ShaderResourceSet>, 6> scatter_add_packed_bindings;
  VulkanRwBufferReplay<3> gather_replay;
  VulkanRwBufferReplay<3> scatter_replay;
  VulkanRwBufferReplay<4> gather_strided_replay;
  VulkanRwBufferReplay<4> scatter_strided_replay;
  std::array<VulkanRwBufferReplay<3>, 6> scatter_add_replay;
  std::array<VulkanRwBufferReplay<3>, 6> scatter_add_strided_replay;
  std::array<VulkanRwBufferReplay<3>, 6> scatter_add_packed_replay;
  VulkanCommandReplayCache gather_command_replay;
  VulkanCommandReplayCache scatter_command_replay;
  VulkanCommandReplayCache gather_strided_command_replay;
  VulkanCommandReplayCache scatter_strided_command_replay;
  std::array<VulkanCommandReplayCache, 6> scatter_add_command_replay;
  std::array<VulkanCommandReplayCache, 6> scatter_add_strided_command_replay;
  std::array<VulkanCommandReplayCache, 6> scatter_add_packed_command_replay;
  DeviceAllocation indexed_copy_params{kDeviceNullAllocation};

  void clear_allocs() {
    reset_binding_replay();
    if (device && indexed_copy_params != kDeviceNullAllocation) {
      device->dealloc_memory(indexed_copy_params);
    }
    indexed_copy_params = kDeviceNullAllocation;
    cached_bytes = 0;
  }

  ~VulkanIndexedCopyCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void reset_binding_replay() {
    gather_replay.reset();
    scatter_replay.reset();
    gather_strided_replay.reset();
    scatter_strided_replay.reset();
    gather_command_replay.reset();
    scatter_command_replay.reset();
    gather_strided_command_replay.reset();
    scatter_strided_command_replay.reset();
    for (auto &replay : scatter_add_replay) {
      replay.reset();
    }
    for (auto &replay : scatter_add_strided_replay) {
      replay.reset();
    }
    for (auto &replay : scatter_add_packed_replay) {
      replay.reset();
    }
    for (auto &replay : scatter_add_command_replay) {
      replay.reset();
    }
    for (auto &replay : scatter_add_strided_command_replay) {
      replay.reset();
    }
    for (auto &replay : scatter_add_packed_command_replay) {
      replay.reset();
    }
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      gather_u32_by_i32.reset();
      scatter_u32_by_i32.reset();
      scatter_dense_u32_by_i32.reset();
      gather_strided_u32_by_i32.reset();
      scatter_strided_u32_by_i32.reset();
      scatter_add_i32_by_i32.reset();
      scatter_add_f32_by_i32.reset();
      scatter_add_u32_by_i32.reset();
      scatter_add_u64_by_i32.reset();
      scatter_add_i64_by_i32.reset();
      scatter_add_f64_by_i32.reset();
      for (auto &pipeline : scatter_add_strided) {
        pipeline.reset();
      }
      for (auto &pipeline : scatter_add_packed) {
        pipeline.reset();
      }
      gather_bindings.reset();
      scatter_bindings.reset();
      gather_strided_bindings.reset();
      scatter_strided_bindings.reset();
      scatter_add_i32_bindings.reset();
      scatter_add_f32_bindings.reset();
      scatter_add_u32_bindings.reset();
      scatter_add_u64_bindings.reset();
      scatter_add_i64_bindings.reset();
      scatter_add_f64_bindings.reset();
      for (auto &bindings : scatter_add_strided_bindings) {
        bindings.reset();
      }
      for (auto &bindings : scatter_add_packed_bindings) {
        bindings.reset();
      }
      reset_binding_replay();
    }
    device = dev;
  }

  Pipeline *indexed_copy_pipeline(bool scatter, bool strided) {
    if (!scatter && !strided) {
      if (!gather_u32_by_i32) {
        gather_u32_by_i32 = create_pipeline(
            device, kGatherU32ByI32Spv, "vulkan_gather_u32_by_i32");
      }
      return gather_u32_by_i32.get();
    }
    if (scatter && !strided) {
      if (!scatter_u32_by_i32) {
        scatter_u32_by_i32 = create_pipeline(
            device, kScatterU32ByI32Spv, "vulkan_scatter_u32_by_i32");
      }
      return scatter_u32_by_i32.get();
    }
    if (!scatter && strided) {
      if (!gather_strided_u32_by_i32) {
        gather_strided_u32_by_i32 = create_pipeline(
            device, kGatherStridedU32ByI32Spv,
            "vulkan_gather_strided_u32_by_i32");
      }
      return gather_strided_u32_by_i32.get();
    }
    if (!scatter_strided_u32_by_i32) {
      scatter_strided_u32_by_i32 = create_pipeline(
          device, kScatterStridedU32ByI32Spv,
          "vulkan_scatter_strided_u32_by_i32");
    }
    return scatter_strided_u32_by_i32.get();
  }

  Pipeline *indexed_copy_dense_u32_scatter_pipeline() {
    if (!scatter_dense_u32_by_i32) {
      scatter_dense_u32_by_i32 = create_pipeline(
          device, kScatterDenseU32ByI32Spv,
          "vulkan_scatter_dense_u32_by_i32");
    }
    return scatter_dense_u32_by_i32.get();
  }

  void ensure_scatter_add_strided_pipeline(int value_type) {
    if (scatter_add_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddF32ByI32StridedSpv,
            "vulkan_scatter_add_f32_by_i32_strided");
        return;
      case 2:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddU32ByI32StridedSpv,
            "vulkan_scatter_add_u32_by_i32_strided");
        return;
      case 3:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddU64ByI32StridedSpv,
            "vulkan_scatter_add_u64_by_i32_strided");
        return;
      case 4:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddI64ByI32StridedSpv,
            "vulkan_scatter_add_i64_by_i32_strided");
        return;
      case 5:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddF64ByI32StridedSpv,
            "vulkan_scatter_add_f64_by_i32_strided");
        return;
      default:
        scatter_add_strided[value_type] = create_pipeline(
            device, kScatterAddI32ByI32StridedSpv,
            "vulkan_scatter_add_i32_by_i32_strided");
        return;
    }
  }

  void ensure_scatter_add_packed_pipeline(int value_type) {
    if (scatter_add_packed[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddF32ByI32PackedSpv,
            "vulkan_scatter_add_f32_by_i32_packed");
        return;
      case 2:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddU32ByI32PackedSpv,
            "vulkan_scatter_add_u32_by_i32_packed");
        return;
      case 3:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddU64ByI32PackedSpv,
            "vulkan_scatter_add_u64_by_i32_packed");
        return;
      case 4:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddI64ByI32PackedSpv,
            "vulkan_scatter_add_i64_by_i32_packed");
        return;
      case 5:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddF64ByI32PackedSpv,
            "vulkan_scatter_add_f64_by_i32_packed");
        return;
      default:
        scatter_add_packed[value_type] = create_pipeline(
            device, kScatterAddI32ByI32PackedSpv,
            "vulkan_scatter_add_i32_by_i32_packed");
        return;
    }
  }

  ShaderResourceSet *cached_resource_set(bool scatter) {
    auto &bindings = scatter ? scatter_bindings : gather_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_strided_resource_set(bool scatter) {
    auto &bindings = scatter ? scatter_strided_bindings : gather_strided_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_scatter_add_resource_set(int value_type) {
    auto &bindings = value_type == 1   ? scatter_add_f32_bindings
                     : value_type == 2 ? scatter_add_u32_bindings
                     : value_type == 3 ? scatter_add_u64_bindings
                     : value_type == 4 ? scatter_add_i64_bindings
                     : value_type == 5 ? scatter_add_f64_bindings
                                       : scatter_add_i32_bindings;
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_scatter_add_strided_resource_set(int value_type) {
    auto &bindings = scatter_add_strided_bindings[value_type];
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  ShaderResourceSet *cached_scatter_add_packed_resource_set(int value_type) {
    auto &bindings = scatter_add_packed_bindings[value_type];
    if (!bindings) {
      bindings.reset(device->create_resource_set());
    }
    return bindings.get();
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(
        res != RhiResult::success,
        "Failed to allocate Vulkan indexed-copy workspace: RhiResult({})",
        res);
    return alloc;
  }

  void ensure_indexed_copy_params() {
    constexpr size_t kParamsBytes = 7 * sizeof(uint32_t);
    if (indexed_copy_params == kDeviceNullAllocation) {
      indexed_copy_params = alloc_storage(kParamsBytes);
    }
    cached_bytes = std::max(cached_bytes, kParamsBytes);
  }

  Pipeline *scatter_add_pipeline(int value_type) {
    if (value_type == 1) {
      if (device->get_caps().get(DeviceCapability::spirv_has_atomic_float_add) ==
          0) {
        return nullptr;
      }
      if (!scatter_add_f32_by_i32) {
        scatter_add_f32_by_i32 = create_pipeline(
            device, kScatterAddF32ByI32Spv, "vulkan_scatter_add_f32_by_i32");
      }
      return scatter_add_f32_by_i32.get();
    }
    if (value_type == 3) {
      if (device->get_caps().get(DeviceCapability::spirv_has_int64) == 0 ||
          device->get_caps().get(DeviceCapability::spirv_has_atomic_int64) ==
              0) {
        return nullptr;
      }
      if (!scatter_add_u64_by_i32) {
        scatter_add_u64_by_i32 = create_pipeline(
            device, kScatterAddU64ByI32Spv, "vulkan_scatter_add_u64_by_i32");
      }
      return scatter_add_u64_by_i32.get();
    }
    if (value_type == 4) {
      if (device->get_caps().get(DeviceCapability::spirv_has_int64) == 0 ||
          device->get_caps().get(DeviceCapability::spirv_has_atomic_int64) ==
              0) {
        return nullptr;
      }
      if (!scatter_add_i64_by_i32) {
        scatter_add_i64_by_i32 = create_pipeline(
            device, kScatterAddI64ByI32Spv, "vulkan_scatter_add_i64_by_i32");
      }
      return scatter_add_i64_by_i32.get();
    }
    if (value_type == 5) {
      if (device->get_caps().get(DeviceCapability::spirv_has_float64) == 0 ||
          device->get_caps().get(
              DeviceCapability::spirv_has_atomic_float64_add) == 0) {
        return nullptr;
      }
      if (!scatter_add_f64_by_i32) {
        scatter_add_f64_by_i32 = create_pipeline(
            device, kScatterAddF64ByI32Spv, "vulkan_scatter_add_f64_by_i32");
      }
      return scatter_add_f64_by_i32.get();
    }
    if (value_type == 2) {
      if (!scatter_add_u32_by_i32) {
        scatter_add_u32_by_i32 = create_pipeline(
            device, kScatterAddU32ByI32Spv, "vulkan_scatter_add_u32_by_i32");
      }
      return scatter_add_u32_by_i32.get();
    }
    if (!scatter_add_i32_by_i32) {
      scatter_add_i32_by_i32 = create_pipeline(
          device, kScatterAddI32ByI32Spv, "vulkan_scatter_add_i32_by_i32");
    }
    return scatter_add_i32_by_i32.get();
  }

  Pipeline *scatter_add_strided_pipeline(int value_type) const {
    return scatter_add_strided[value_type].get();
  }

  Pipeline *scatter_add_packed_pipeline(int value_type) const {
    return scatter_add_packed[value_type].get();
  }
};

struct VulkanBucketBuilderCache {
  Device *device{nullptr};
  size_t partial_capacity{0};
  size_t cached_bytes{0};
  DeviceAllocation partial{kDeviceNullAllocation};
  std::unique_ptr<Pipeline> clear_i32;
  std::unique_ptr<Pipeline> count_i32;
  std::unique_ptr<Pipeline> count_private_shared_i32;
  std::unique_ptr<Pipeline> prefix_i32;
  std::unique_ptr<Pipeline> prefix_chunks_i32;
  std::unique_ptr<Pipeline> scatter_i32;
  std::unique_ptr<Pipeline> scatter_f32;
  std::unique_ptr<Pipeline> scatter_u32;
  std::unique_ptr<Pipeline> scatter_raw64;
  std::unique_ptr<Pipeline> scatter_raw_words;
  std::unique_ptr<Pipeline> scatter_private_shared_i32;
  std::unique_ptr<Pipeline> scatter_private_shared_f32;
  std::unique_ptr<Pipeline> scatter_private_shared_u32;
  std::unique_ptr<Pipeline> scatter_private_shared_raw64;
  std::unique_ptr<Pipeline> scatter_private_shared_raw_words;
  std::unique_ptr<Pipeline> grouped_reduce_zero_i32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_f32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_u32;
  std::unique_ptr<Pipeline> grouped_reduce_zero_u64;
  std::unique_ptr<Pipeline> grouped_reduce_zero_i64;
  std::unique_ptr<Pipeline> grouped_reduce_zero_f64;
  std::array<std::unique_ptr<Pipeline>, 6> grouped_reduce_zero_strided;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_i32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_f32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_u32;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_u64;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_i64;
  std::unique_ptr<Pipeline> grouped_reduce_atomic_sum_f64;
  std::array<std::unique_ptr<Pipeline>, 6> grouped_reduce_atomic_sum_strided;
  std::unique_ptr<Pipeline> grouped_reduce_sum_i32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_f32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_u32;
  std::unique_ptr<Pipeline> grouped_reduce_sum_u64;
  std::unique_ptr<Pipeline> grouped_reduce_sum_i64;
  std::unique_ptr<Pipeline> grouped_reduce_sum_f64;
  std::array<VulkanResourceSetReplayRing<1>, 6>
      grouped_reduce_zero_ring_bindings;
  std::array<VulkanResourceSetReplayRing<3>, 6>
      grouped_reduce_atomic_ring_bindings;
  std::array<VulkanResourceSetReplayRing<1>, 6>
      grouped_reduce_zero_strided_ring_bindings;
  std::array<VulkanResourceSetReplayRing<3>, 6>
      grouped_reduce_atomic_strided_ring_bindings;
  std::array<VulkanResourceSetReplayRing<3>, 6>
      grouped_reduce_sum_ring_bindings;
  VulkanResourceSetReplayRing<2> bucket_clear_replay_bindings;
  VulkanResourceSetReplayRing<2> bucket_count_replay_bindings;
  VulkanResourceSetReplayRing<2> bucket_count_private_replay_bindings;
  VulkanResourceSetReplayRing<2> bucket_prefix_replay_bindings;
  VulkanResourceSetReplayRing<2> bucket_prefix_chunks_replay_bindings;
  std::array<VulkanResourceSetReplayRing<4>, 8> bucket_scatter_replay_bindings;
  std::array<VulkanResourceSetReplayRing<5>, 8>
      bucket_scatter_private_replay_bindings;
  VulkanCommandReplayCache bucket_ndarray_command_replay;
  VulkanCommandReplayCache bucket_dense_field_command_replay;
  VulkanCommandReplayCache grouped_reduce_atomic_command_replay;
  VulkanCommandReplayCache grouped_reduce_atomic_strided_command_replay;
  VulkanCommandReplayCache grouped_reduce_sum_command_replay;

  void reset_bucket_resource_sets() {
    bucket_clear_replay_bindings.reset();
    bucket_count_replay_bindings.reset();
    bucket_count_private_replay_bindings.reset();
    bucket_prefix_replay_bindings.reset();
    bucket_prefix_chunks_replay_bindings.reset();
    for (auto &ring : bucket_scatter_replay_bindings) {
      ring.reset();
    }
    for (auto &ring : bucket_scatter_private_replay_bindings) {
      ring.reset();
    }
    bucket_ndarray_command_replay.reset();
    bucket_dense_field_command_replay.reset();
  }

  void reset_grouped_reduce_binding_replay() {
    for (auto &ring : grouped_reduce_zero_ring_bindings) {
      ring.reset();
    }
    for (auto &ring : grouped_reduce_atomic_ring_bindings) {
      ring.reset();
    }
    for (auto &ring : grouped_reduce_zero_strided_ring_bindings) {
      ring.reset();
    }
    for (auto &ring : grouped_reduce_atomic_strided_ring_bindings) {
      ring.reset();
    }
    for (auto &ring : grouped_reduce_sum_ring_bindings) {
      ring.reset();
    }
    grouped_reduce_atomic_command_replay.reset();
    grouped_reduce_atomic_strided_command_replay.reset();
    grouped_reduce_sum_command_replay.reset();
  }

  void clear_allocs() {
    reset_bucket_resource_sets();
    reset_grouped_reduce_binding_replay();
    if (device && partial != kDeviceNullAllocation) {
      device->dealloc_memory(partial);
    }
    partial = kDeviceNullAllocation;
    partial_capacity = 0;
    cached_bytes = 0;
  }

  ~VulkanBucketBuilderCache() {
    clear_allocs();
  }

  size_t allocated_bytes() const noexcept {
    return cached_bytes;
  }

  void ensure_pipelines(Device *dev) {
    if (device == dev) {
      return;
    }
    if (device && device != dev) {
      clear_allocs();
      clear_i32.reset();
      count_i32.reset();
      count_private_shared_i32.reset();
      prefix_i32.reset();
      prefix_chunks_i32.reset();
      scatter_i32.reset();
      scatter_f32.reset();
      scatter_u32.reset();
      scatter_raw64.reset();
      scatter_raw_words.reset();
      scatter_private_shared_i32.reset();
      scatter_private_shared_f32.reset();
      scatter_private_shared_u32.reset();
      scatter_private_shared_raw64.reset();
      scatter_private_shared_raw_words.reset();
      grouped_reduce_zero_i32.reset();
      grouped_reduce_zero_f32.reset();
      grouped_reduce_zero_u32.reset();
      grouped_reduce_zero_u64.reset();
      grouped_reduce_zero_i64.reset();
      grouped_reduce_zero_f64.reset();
      for (auto &pipeline : grouped_reduce_zero_strided) {
        pipeline.reset();
      }
      grouped_reduce_atomic_sum_i32.reset();
      grouped_reduce_atomic_sum_f32.reset();
      grouped_reduce_atomic_sum_u32.reset();
      grouped_reduce_atomic_sum_u64.reset();
      grouped_reduce_atomic_sum_i64.reset();
      grouped_reduce_atomic_sum_f64.reset();
      for (auto &pipeline : grouped_reduce_atomic_sum_strided) {
        pipeline.reset();
      }
      grouped_reduce_sum_i32.reset();
      grouped_reduce_sum_f32.reset();
      grouped_reduce_sum_u32.reset();
      grouped_reduce_sum_u64.reset();
      grouped_reduce_sum_i64.reset();
      grouped_reduce_sum_f64.reset();
      reset_grouped_reduce_binding_replay();
      reset_bucket_resource_sets();
    }
    device = dev;
  }

  Pipeline *bucket_clear_i32_pipeline() {
    if (!clear_i32) {
      clear_i32 =
          create_pipeline(device, kBucketClearI32Spv, "vulkan_bucket_clear_i32");
    }
    return clear_i32.get();
  }

  Pipeline *bucket_count_i32_pipeline() {
    if (!count_i32) {
      count_i32 =
          create_pipeline(device, kBucketCountI32Spv, "vulkan_bucket_count_i32");
    }
    return count_i32.get();
  }

  Pipeline *bucket_count_private_shared_i32_pipeline() {
    if (!count_private_shared_i32) {
      count_private_shared_i32 = create_pipeline(
          device, kBucketCountPrivateSharedI32Spv,
          "vulkan_bucket_count_private_shared_i32");
    }
    return count_private_shared_i32.get();
  }

  Pipeline *bucket_prefix_i32_pipeline() {
    if (!prefix_i32) {
      prefix_i32 = create_pipeline(device, kBucketPrefixI32Spv,
                                   "vulkan_bucket_prefix_i32");
    }
    return prefix_i32.get();
  }

  Pipeline *bucket_prefix_chunks_i32_pipeline() {
    if (!prefix_chunks_i32) {
      prefix_chunks_i32 = create_pipeline(
          device, kBucketPrefixChunksI32Spv,
          "vulkan_bucket_prefix_chunks_i32");
    }
    return prefix_chunks_i32.get();
  }

  Pipeline *grouped_reduce_zero_pipeline(int value_type) {
    if (value_type == 1) {
      if (!grouped_reduce_zero_f32) {
        grouped_reduce_zero_f32 = create_pipeline(
            device, kGroupedReduceZeroF32Spv, "vulkan_grouped_reduce_zero_f32");
      }
      return grouped_reduce_zero_f32.get();
    }
    if (value_type == 3) {
      if (!grouped_reduce_zero_u64) {
        grouped_reduce_zero_u64 = create_pipeline(
            device, kGroupedReduceZeroU64Spv, "vulkan_grouped_reduce_zero_u64");
      }
      return grouped_reduce_zero_u64.get();
    }
    if (value_type == 4) {
      if (!grouped_reduce_zero_i64) {
        grouped_reduce_zero_i64 = create_pipeline(
            device, kGroupedReduceZeroI64Spv, "vulkan_grouped_reduce_zero_i64");
      }
      return grouped_reduce_zero_i64.get();
    }
    if (value_type == 5) {
      if (!grouped_reduce_zero_f64) {
        grouped_reduce_zero_f64 = create_pipeline(
            device, kGroupedReduceZeroF64Spv, "vulkan_grouped_reduce_zero_f64");
      }
      return grouped_reduce_zero_f64.get();
    }
    if (value_type == 2) {
      if (!grouped_reduce_zero_u32) {
        grouped_reduce_zero_u32 = create_pipeline(
            device, kGroupedReduceZeroU32Spv, "vulkan_grouped_reduce_zero_u32");
      }
      return grouped_reduce_zero_u32.get();
    }
    if (!grouped_reduce_zero_i32) {
      grouped_reduce_zero_i32 = create_pipeline(
          device, kGroupedReduceZeroI32Spv, "vulkan_grouped_reduce_zero_i32");
    }
    return grouped_reduce_zero_i32.get();
  }

  void ensure_grouped_reduce_zero_strided_pipeline(int value_type) {
    if (grouped_reduce_zero_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroF32StridedSpv,
            "vulkan_grouped_reduce_zero_f32_strided");
        return;
      case 2:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroU32StridedSpv,
            "vulkan_grouped_reduce_zero_u32_strided");
        return;
      case 3:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroU64StridedSpv,
            "vulkan_grouped_reduce_zero_u64_strided");
        return;
      case 4:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroI64StridedSpv,
            "vulkan_grouped_reduce_zero_i64_strided");
        return;
      case 5:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroF64StridedSpv,
            "vulkan_grouped_reduce_zero_f64_strided");
        return;
      default:
        grouped_reduce_zero_strided[value_type] = create_pipeline(
            device, kGroupedReduceZeroI32StridedSpv,
            "vulkan_grouped_reduce_zero_i32_strided");
        return;
    }
  }

  Pipeline *grouped_reduce_zero_strided_pipeline(int value_type) const {
    return grouped_reduce_zero_strided[value_type].get();
  }

  Pipeline *grouped_reduce_atomic_pipeline(int value_type) {
    if (value_type == 1) {
      if (!grouped_reduce_atomic_sum_f32) {
        grouped_reduce_atomic_sum_f32 = create_pipeline(
            device, kGroupedReduceAtomicSumF32Spv,
            "vulkan_grouped_reduce_atomic_sum_f32");
      }
      return grouped_reduce_atomic_sum_f32.get();
    }
    if (value_type == 3) {
      if (!grouped_reduce_atomic_sum_u64) {
        grouped_reduce_atomic_sum_u64 = create_pipeline(
            device, kGroupedReduceAtomicSumU64Spv,
            "vulkan_grouped_reduce_atomic_sum_u64");
      }
      return grouped_reduce_atomic_sum_u64.get();
    }
    if (value_type == 4) {
      if (!grouped_reduce_atomic_sum_i64) {
        grouped_reduce_atomic_sum_i64 = create_pipeline(
            device, kGroupedReduceAtomicSumI64Spv,
            "vulkan_grouped_reduce_atomic_sum_i64");
      }
      return grouped_reduce_atomic_sum_i64.get();
    }
    if (value_type == 5) {
      if (!grouped_reduce_atomic_sum_f64) {
        grouped_reduce_atomic_sum_f64 = create_pipeline(
            device, kGroupedReduceAtomicSumF64Spv,
            "vulkan_grouped_reduce_atomic_sum_f64");
      }
      return grouped_reduce_atomic_sum_f64.get();
    }
    if (value_type == 2) {
      if (!grouped_reduce_atomic_sum_u32) {
        grouped_reduce_atomic_sum_u32 = create_pipeline(
            device, kGroupedReduceAtomicSumU32Spv,
            "vulkan_grouped_reduce_atomic_sum_u32");
      }
      return grouped_reduce_atomic_sum_u32.get();
    }
    if (!grouped_reduce_atomic_sum_i32) {
      grouped_reduce_atomic_sum_i32 =
          create_pipeline(device, kGroupedReduceAtomicSumI32Spv,
                          "vulkan_grouped_reduce_atomic_sum_i32");
    }
    return grouped_reduce_atomic_sum_i32.get();
  }

  void ensure_grouped_reduce_atomic_strided_pipeline(int value_type) {
    if (grouped_reduce_atomic_sum_strided[value_type]) {
      return;
    }
    switch (value_type) {
      case 1:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumF32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_f32_strided");
        return;
      case 2:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumU32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_u32_strided");
        return;
      case 3:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumU64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_u64_strided");
        return;
      case 4:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumI64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_i64_strided");
        return;
      case 5:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumF64StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_f64_strided");
        return;
      default:
        grouped_reduce_atomic_sum_strided[value_type] = create_pipeline(
            device, kGroupedReduceAtomicSumI32StridedSpv,
            "vulkan_grouped_reduce_atomic_sum_i32_strided");
        return;
    }
  }

  Pipeline *grouped_reduce_atomic_strided_pipeline(int value_type) const {
    return grouped_reduce_atomic_sum_strided[value_type].get();
  }

  Pipeline *bucket_scatter_pipeline(int value_type) {
    if (value_type == 7) {
      if (!scatter_raw_words) {
        scatter_raw_words = create_pipeline(
            device, kBucketScatterRawWordsSpv,
            "vulkan_bucket_scatter_raw_words");
      }
      return scatter_raw_words.get();
    }
    if (value_type == 1) {
      if (!scatter_f32) {
        scatter_f32 = create_pipeline(device, kBucketScatterF32Spv,
                                      "vulkan_bucket_scatter_f32");
      }
      return scatter_f32.get();
    }
    if (value_type == 2) {
      if (!scatter_u32) {
        scatter_u32 = create_pipeline(device, kBucketScatterU32Spv,
                                      "vulkan_bucket_scatter_u32");
      }
      return scatter_u32.get();
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      if (!scatter_raw64) {
        scatter_raw64 = create_pipeline(device, kBucketScatterRaw64Spv,
                                        "vulkan_bucket_scatter_raw64");
      }
      return scatter_raw64.get();
    }
    if (!scatter_i32) {
      scatter_i32 = create_pipeline(device, kBucketScatterI32Spv,
                                    "vulkan_bucket_scatter_i32");
    }
    return scatter_i32.get();
  }

  Pipeline *bucket_scatter_private_pipeline(int value_type) {
    if (value_type == 7) {
      if (!scatter_private_shared_raw_words) {
        scatter_private_shared_raw_words = create_pipeline(
            device, kBucketScatterPrivateSharedRawWordsSpv,
            "vulkan_bucket_scatter_private_shared_raw_words");
      }
      return scatter_private_shared_raw_words.get();
    }
    if (value_type == 1) {
      if (!scatter_private_shared_f32) {
        scatter_private_shared_f32 = create_pipeline(
            device, kBucketScatterPrivateSharedF32Spv,
            "vulkan_bucket_scatter_private_shared_f32");
      }
      return scatter_private_shared_f32.get();
    }
    if (value_type == 2) {
      if (!scatter_private_shared_u32) {
        scatter_private_shared_u32 = create_pipeline(
            device, kBucketScatterPrivateSharedU32Spv,
            "vulkan_bucket_scatter_private_shared_u32");
      }
      return scatter_private_shared_u32.get();
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      if (!scatter_private_shared_raw64) {
        scatter_private_shared_raw64 = create_pipeline(
            device, kBucketScatterPrivateSharedRaw64Spv,
            "vulkan_bucket_scatter_private_shared_raw64");
      }
      return scatter_private_shared_raw64.get();
    }
    if (!scatter_private_shared_i32) {
      scatter_private_shared_i32 = create_pipeline(
          device, kBucketScatterPrivateSharedI32Spv,
          "vulkan_bucket_scatter_private_shared_i32");
    }
    return scatter_private_shared_i32.get();
  }

  const char *bucket_scatter_scope(int value_type) const {
    if (value_type == 7) {
      return "vulkan_bucket_scatter_raw_words";
    }
    if (value_type == 1) {
      return "vulkan_bucket_scatter_f32";
    }
    if (value_type == 2) {
      return "vulkan_bucket_scatter_u32";
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return "vulkan_bucket_scatter_raw64";
    }
    return "vulkan_bucket_scatter_i32";
  }

  const char *bucket_scatter_private_scope(int value_type) const {
    if (value_type == 7) {
      return "vulkan_bucket_scatter_private_shared_raw_words";
    }
    if (value_type == 1) {
      return "vulkan_bucket_scatter_private_shared_f32";
    }
    if (value_type == 2) {
      return "vulkan_bucket_scatter_private_shared_u32";
    }
    if (value_type == 3 || value_type == 4 || value_type == 5) {
      return "vulkan_bucket_scatter_private_shared_raw64";
    }
    return "vulkan_bucket_scatter_private_shared_i32";
  }

  size_t bucket_scatter_replay_index(int value_type) const {
    if (value_type == 7) {
      return 7;
    }
    if (value_type >= 0 && value_type <= 5) {
      return static_cast<size_t>(value_type);
    }
    return 0;
  }

  VulkanReplayResourceSet<2> bind_bucket_clear_resource_set(
      Program *program,
      DeviceAllocation offsets_alloc,
      uint64_t offsets_offset,
      size_t offset_bytes,
      DeviceAllocation cursor_alloc,
      uint64_t cursor_offset,
      size_t cursor_bytes) {
    return bucket_clear_replay_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(offsets_alloc, offsets_offset, offset_bytes),
            rw_buffer_request(cursor_alloc, cursor_offset, cursor_bytes)});
  }

  VulkanReplayResourceSet<2> bind_bucket_count_resource_set(
      Program *program,
      DeviceAllocation keys_alloc,
      uint64_t keys_offset,
      size_t key_bytes,
      DeviceAllocation offsets_alloc,
      uint64_t offsets_offset,
      size_t offset_bytes) {
    return bucket_count_replay_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(keys_alloc, keys_offset, key_bytes),
            rw_buffer_request(offsets_alloc, offsets_offset, offset_bytes)});
  }

  VulkanReplayResourceSet<2> bind_bucket_count_private_resource_set(
      Program *program,
      DeviceAllocation keys_alloc,
      uint64_t keys_offset,
      size_t key_bytes,
      DeviceAllocation partial_alloc,
      size_t private_partial_bytes) {
    return bucket_count_private_replay_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(keys_alloc, keys_offset, key_bytes),
            rw_buffer_request(partial_alloc, 0, private_partial_bytes)});
  }

  VulkanReplayResourceSet<2> bind_bucket_prefix_resource_set(
      Program *program,
      DeviceAllocation offsets_alloc,
      uint64_t offsets_offset,
      size_t offset_bytes,
      DeviceAllocation cursor_alloc,
      uint64_t cursor_offset,
      size_t cursor_bytes) {
    return bucket_prefix_replay_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(offsets_alloc, offsets_offset, offset_bytes),
            rw_buffer_request(cursor_alloc, cursor_offset, cursor_bytes)});
  }

  VulkanReplayResourceSet<2> bind_bucket_prefix_chunks_resource_set(
      Program *program,
      DeviceAllocation partial_alloc,
      size_t private_partial_bytes,
      DeviceAllocation offsets_alloc,
      uint64_t offsets_offset,
      size_t offset_bytes) {
    return bucket_prefix_chunks_replay_bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 2>{
            rw_buffer_request(partial_alloc, 0, private_partial_bytes),
            rw_buffer_request(offsets_alloc, offsets_offset, offset_bytes)});
  }

  VulkanReplayResourceSet<4> bind_bucket_scatter_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation keys_alloc,
      uint64_t keys_offset,
      size_t key_bytes,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation cursor_alloc,
      uint64_t cursor_offset,
      size_t cursor_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset) {
    auto &bindings =
        bucket_scatter_replay_bindings[bucket_scatter_replay_index(value_type)];
    return bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 4>{
            rw_buffer_request(keys_alloc, keys_offset, key_bytes),
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(cursor_alloc, cursor_offset, cursor_bytes),
            rw_buffer_request(output_alloc, output_offset, value_bytes)});
  }

  VulkanReplayResourceSet<5> bind_bucket_scatter_private_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation keys_alloc,
      uint64_t keys_offset,
      size_t key_bytes,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t value_bytes,
      DeviceAllocation partial_alloc,
      size_t private_partial_bytes,
      DeviceAllocation offsets_alloc,
      uint64_t offsets_offset,
      size_t offset_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset) {
    auto &bindings = bucket_scatter_private_replay_bindings
        [bucket_scatter_replay_index(value_type)];
    return bindings.bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 5>{
            rw_buffer_request(keys_alloc, keys_offset, key_bytes),
            rw_buffer_request(values_alloc, values_offset, value_bytes),
            rw_buffer_request(partial_alloc, 0, private_partial_bytes),
            rw_buffer_request(offsets_alloc, offsets_offset, offset_bytes),
            rw_buffer_request(output_alloc, output_offset, value_bytes)});
  }

  Pipeline *grouped_reduce_sum_pipeline(int value_type) {
    switch (value_type) {
      case 1:
        if (!grouped_reduce_sum_f32) {
          grouped_reduce_sum_f32 = create_pipeline(
              device, kGroupedReduceSumF32Spv, "vulkan_grouped_reduce_sum_f32");
        }
        return grouped_reduce_sum_f32.get();
      case 2:
        if (!grouped_reduce_sum_u32) {
          grouped_reduce_sum_u32 = create_pipeline(
              device, kGroupedReduceSumU32Spv, "vulkan_grouped_reduce_sum_u32");
        }
        return grouped_reduce_sum_u32.get();
      case 3:
        if (!grouped_reduce_sum_u64) {
          grouped_reduce_sum_u64 = create_pipeline(
              device, kGroupedReduceSumU64Spv, "vulkan_grouped_reduce_sum_u64");
        }
        return grouped_reduce_sum_u64.get();
      case 4:
        if (!grouped_reduce_sum_i64) {
          grouped_reduce_sum_i64 = create_pipeline(
              device, kGroupedReduceSumI64Spv, "vulkan_grouped_reduce_sum_i64");
        }
        return grouped_reduce_sum_i64.get();
      case 5:
        if (!grouped_reduce_sum_f64) {
          grouped_reduce_sum_f64 = create_pipeline(
              device, kGroupedReduceSumF64Spv, "vulkan_grouped_reduce_sum_f64");
        }
        return grouped_reduce_sum_f64.get();
      default:
        if (!grouped_reduce_sum_i32) {
          grouped_reduce_sum_i32 = create_pipeline(
              device, kGroupedReduceSumI32Spv, "vulkan_grouped_reduce_sum_i32");
        }
        return grouped_reduce_sum_i32.get();
    }
  }

  const char *grouped_reduce_sum_scope(int value_type) const {
    switch (value_type) {
      case 1:
        return "vulkan_grouped_reduce_sum_f32";
      case 2:
        return "vulkan_grouped_reduce_sum_u32";
      case 3:
        return "vulkan_grouped_reduce_sum_u64";
      case 4:
        return "vulkan_grouped_reduce_sum_i64";
      case 5:
        return "vulkan_grouped_reduce_sum_f64";
      default:
        return "vulkan_grouped_reduce_sum_i32";
    }
  }

  VulkanReplayResourceSet<1> bind_grouped_reduce_zero_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return grouped_reduce_zero_ring_bindings[value_type].bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 1>{
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }

  VulkanReplayResourceSet<3> bind_grouped_reduce_atomic_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation keys_alloc,
      uint64_t keys_offset,
      size_t input_bytes,
      DeviceAllocation values_alloc,
      uint64_t values_offset,
      size_t values_bytes,
      DeviceAllocation output_alloc,
      uint64_t output_offset,
      size_t output_bytes) {
    return grouped_reduce_atomic_ring_bindings[value_type].bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(keys_alloc, keys_offset, input_bytes),
            rw_buffer_request(values_alloc, values_offset, values_bytes),
            rw_buffer_request(output_alloc, output_offset, output_bytes)});
  }

  VulkanReplayResourceSet<1> bind_grouped_reduce_zero_strided_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation output_alloc,
      size_t output_bytes) {
    return grouped_reduce_zero_strided_ring_bindings[value_type].bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 1>{
            rw_buffer_request(output_alloc, 0, output_bytes)});
  }

  VulkanReplayResourceSet<3> bind_grouped_reduce_atomic_strided_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation keys_alloc,
      size_t keys_bytes,
      DeviceAllocation values_alloc,
      size_t values_bytes,
      DeviceAllocation output_alloc,
      size_t output_bytes) {
    return grouped_reduce_atomic_strided_ring_bindings[value_type].bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(keys_alloc, 0, keys_bytes),
            rw_buffer_request(values_alloc, 0, values_bytes),
            rw_buffer_request(output_alloc, 0, output_bytes)});
  }

  VulkanReplayResourceSet<3> bind_grouped_reduce_sum_resource_set(
      Program *program,
      int value_type,
      DeviceAllocation offsets_alloc,
      size_t offset_bytes,
      DeviceAllocation scratch_alloc,
      size_t scratch_bytes,
      DeviceAllocation output_alloc,
      size_t output_bytes) {
    return grouped_reduce_sum_ring_bindings[value_type].bind(
        program, device,
        std::array<VulkanRwBufferBindingRequest, 3>{
            rw_buffer_request(offsets_alloc, 0, offset_bytes),
            rw_buffer_request(scratch_alloc, 0, scratch_bytes),
            rw_buffer_request(output_alloc, 0, output_bytes)});
  }

  DeviceAllocation alloc_storage(size_t bytes) {
    DeviceAllocation alloc{kDeviceNullAllocation};
    Device::AllocParams params;
    params.size = bytes;
    params.usage = AllocUsage::Storage;
    RhiResult res = device->allocate_memory(params, &alloc);
    TI_ERROR_IF(res != RhiResult::success,
                "Failed to allocate Vulkan bucket builder workspace: "
                "RhiResult({})",
                res);
    return alloc;
  }

  bool needs_workspace_realloc(size_t bytes) const {
    return partial_capacity < bytes;
  }

  void ensure_workspace(size_t bytes) {
    if (bytes == 0 || !needs_workspace_realloc(bytes)) {
      return;
    }
    clear_allocs();
    partial = alloc_storage(bytes);
    partial_capacity = bytes;
    cached_bytes = bytes;
  }
};

template <typename Cache, typename Prepare>
PrimitiveWorkspaceArena::Lease<Cache> acquire_vulkan_cache(
    Program *program,
    PrimitiveWorkspaceFamily family,
    Prepare prepare) {
  TI_ERROR_IF(program == nullptr,
              "Vulkan primitive workspace requires a Program owner");
  auto lease = program->primitive_workspace_arena().acquire<Cache>(
      {PrimitiveWorkspaceBackend::vulkan, family, 0, 0},
      [] { return std::make_shared<Cache>(); });
  prepare(*lease);
  return lease;
}

auto get_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanRadixSortCache>(
      program, PrimitiveWorkspaceFamily::ordering,
      [device](auto &cache) { cache.ensure_device_only(device); });
}

auto get_scan_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanScanCache>(
      program, PrimitiveWorkspaceFamily::scan,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_compact_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanCompactCache>(
      program, PrimitiveWorkspaceFamily::compact,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_histogram_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanHistogramCache>(
      program, PrimitiveWorkspaceFamily::histogram,
      [device](auto &cache) { cache.ensure_pipelines(device); });
}

auto get_reduce_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanReduceCache>(
      program, PrimitiveWorkspaceFamily::reduce,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_check_count_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanCheckCountCache>(
      program, PrimitiveWorkspaceFamily::check,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_metric_reduce_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanMetricReduceCache>(
      program, PrimitiveWorkspaceFamily::metric,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_sparse_algebra_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanSparseAlgebraCache>(
      program, PrimitiveWorkspaceFamily::sparse_algebra,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_transform_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanTransformCache>(
      program, PrimitiveWorkspaceFamily::transform,
      [device](auto &cache) { cache.ensure_pipelines(device); });
}

auto get_add_merge_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanAddMergeCache>(
      program, PrimitiveWorkspaceFamily::scatter_add,
      [device](auto &cache) { cache.ensure_device(device); });
}

auto get_indexed_copy_cache(Program *program, Device *device) {
  return acquire_vulkan_cache<VulkanIndexedCopyCache>(
      program, PrimitiveWorkspaceFamily::indexed,
      [device](auto &cache) { cache.ensure_pipelines(device); });
}

auto get_bucket_builder_cache(Program *program, Device *device) {
  // Bucket construction and grouped reduction intentionally share pipelines
  // and scratch. Keep one arena entry so reporting does not double-count it.
  return acquire_vulkan_cache<VulkanBucketBuilderCache>(
      program, PrimitiveWorkspaceFamily::bucket,
      [device](auto &cache) { cache.ensure_pipelines(device); });
}

void dispatch_pipeline(CommandList *cmdlist,
                       Pipeline *pipeline,
                       ShaderResourceSet *bindings,
                       uint32_t groups,
                       uint32_t groups_y,
                       uint32_t groups_z,
                       const char *scope_name,
                       VulkanSortCpuProfileSample *profile = nullptr) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->bind_pipeline(pipeline);
    profile->bind_pipeline_calls++;
    profile->bind_pipeline_us += profile_time_us() - start;
  } else {
    cmdlist->bind_pipeline(pipeline);
  }
  RhiResult bind_status;
  if (profile) {
    double start = profile_time_us();
    bind_status = cmdlist->bind_shader_resources(bindings);
    profile->bind_shader_resources_calls++;
    profile->bind_shader_resources_us += profile_time_us() - start;
  } else {
    bind_status = cmdlist->bind_shader_resources(bindings);
  }
  TI_ERROR_IF(bind_status != RhiResult::success,
              "Vulkan sort resource binding failed: RhiResult({})",
              bind_status);
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->begin_profiler_scope(scope_name);
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->begin_profiler_scope(scope_name);
  }
  RhiResult dispatch_status;
  if (profile) {
    double start = profile_time_us();
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
    profile->dispatch_calls++;
    profile->dispatch_us += profile_time_us() - start;
  } else {
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
  }
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->end_profiler_scope();
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->end_profiler_scope();
  }
  TI_ERROR_IF(dispatch_status != RhiResult::success,
              "Vulkan sort dispatch failed: RhiResult({})", dispatch_status);
}

void dispatch_pipeline_with_push_constants(CommandList *cmdlist,
                                           Pipeline *pipeline,
                                           ShaderResourceSet *bindings,
                                           const void *push_data,
                                           uint32_t push_bytes,
                                           uint32_t groups,
                                           uint32_t groups_y,
                                           uint32_t groups_z,
                                           const char *scope_name = nullptr,
                                           VulkanSortCpuProfileSample *profile =
                                               nullptr) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->bind_pipeline(pipeline);
    profile->bind_pipeline_calls++;
    profile->bind_pipeline_us += profile_time_us() - start;
  } else {
    cmdlist->bind_pipeline(pipeline);
  }
  RhiResult bind_status;
  if (profile) {
    double start = profile_time_us();
    bind_status = cmdlist->bind_shader_resources(bindings);
    profile->bind_shader_resources_calls++;
    profile->bind_shader_resources_us += profile_time_us() - start;
  } else {
    bind_status = cmdlist->bind_shader_resources(bindings);
  }
  TI_ERROR_IF(bind_status != RhiResult::success,
              "Vulkan sort resource binding failed: RhiResult({})",
              bind_status);
  static_cast<vulkan::VulkanCommandList *>(cmdlist)->push_constants(push_data,
                                                                    push_bytes);
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->begin_profiler_scope(scope_name);
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->begin_profiler_scope(scope_name);
  }
  RhiResult dispatch_status;
  if (profile) {
    double start = profile_time_us();
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
    profile->dispatch_calls++;
    profile->dispatch_us += profile_time_us() - start;
  } else {
    dispatch_status = cmdlist->dispatch(groups, groups_y, groups_z);
  }
  if (scope_name && profile) {
    double start = profile_time_us();
    cmdlist->end_profiler_scope();
    profile->profiler_scope_us += profile_time_us() - start;
  } else if (scope_name) {
    cmdlist->end_profiler_scope();
  }
  TI_ERROR_IF(dispatch_status != RhiResult::success,
              "Vulkan sort dispatch failed: RhiResult({})", dispatch_status);
}

VulkanCommandReplayKey make_vulkan_histogram_command_key(
    bool dense_field,
    int value_type,
    int bin_type,
    bool use_single_shared,
    bool use_private,
    bool use_private_shared,
    DeviceAllocation values_alloc,
    uint64_t values_offset,
    size_t value_bytes,
    DeviceAllocation bins_alloc,
    uint64_t bins_offset,
    size_t bin_bytes,
    DeviceAllocation partial_alloc,
    size_t partial_bytes,
    uint32_t value_groups,
    uint32_t bin_groups,
    uint32_t partial_groups,
    size_t num_chunks,
    Pipeline *clear_pipeline,
    Pipeline *count_direct_pipeline,
    Pipeline *count_private_pipeline,
    Pipeline *count_private_shared_pipeline,
    Pipeline *reduce_private_pipeline,
    Pipeline *single_shared_pipeline,
    ShaderResourceSet *single_shared_bindings,
    ShaderResourceSet *clear_bindings,
    ShaderResourceSet *count_direct_bindings,
    ShaderResourceSet *count_private_bindings,
    ShaderResourceSet *reduce_private_bindings) {
  VulkanCommandReplayKey key;
  key.push(dense_field ? 1 : 0);
  key.push(static_cast<uint64_t>(value_type));
  key.push(static_cast<uint64_t>(bin_type));
  key.push(use_single_shared ? 1 : 0);
  key.push(use_private ? 1 : 0);
  key.push(use_private_shared ? 1 : 0);
  key.push(values_alloc.alloc_id);
  key.push(vulkan_allocation_generation(values_alloc));
  key.push(values_offset);
  key.push(static_cast<uint64_t>(value_bytes));
  key.push(bins_alloc.alloc_id);
  key.push(vulkan_allocation_generation(bins_alloc));
  key.push(bins_offset);
  key.push(static_cast<uint64_t>(bin_bytes));
  key.push(partial_alloc.alloc_id);
  key.push(vulkan_allocation_generation(partial_alloc));
  key.push(static_cast<uint64_t>(partial_bytes));
  key.push(value_groups);
  key.push(bin_groups);
  key.push(partial_groups);
  key.push(static_cast<uint64_t>(num_chunks));
  key.push_ptr(clear_pipeline);
  key.push_ptr(count_direct_pipeline);
  key.push_ptr(count_private_pipeline);
  key.push_ptr(count_private_shared_pipeline);
  key.push_ptr(reduce_private_pipeline);
  key.push_ptr(single_shared_pipeline);
  key.push_ptr(single_shared_bindings);
  key.push_ptr(clear_bindings);
  key.push_ptr(count_direct_bindings);
  key.push_ptr(count_private_bindings);
  key.push_ptr(reduce_private_bindings);
  return key;
}

template <typename Ptr>
void profiled_rw_buffer(ShaderResourceSet *bindings,
                        uint32_t binding,
                        Ptr ptr,
                        size_t bytes,
                        VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    bindings->rw_buffer(binding, ptr, bytes);
    profile->rw_buffer_calls++;
    profile->rw_buffer_us += profile_time_us() - start;
  } else {
    bindings->rw_buffer(binding, ptr, bytes);
  }
}

template <size_t N>
void profiled_replay_rw_buffer(VulkanRwBufferReplay<N> &replay,
                               ShaderResourceSet *bindings,
                               uint32_t binding,
                               DeviceAllocation alloc,
                               uint64_t offset,
                               size_t bytes,
                               VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    const bool updated = replay.rw_buffer(bindings, binding, alloc, offset,
                                          bytes);
    if (updated) {
      profile->rw_buffer_calls++;
      profile->rw_buffer_us += profile_time_us() - start;
    }
  } else {
    replay.rw_buffer(bindings, binding, alloc, offset, bytes);
  }
}

void profiled_buffer_barrier(CommandList *cmdlist,
                             DeviceAllocation alloc,
                             VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_barrier(alloc);
    profile->buffer_barrier_calls++;
    profile->buffer_barrier_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_barrier(alloc);
  }
}

template <typename Ptr>
void profiled_buffer_fill(CommandList *cmdlist,
                          Ptr ptr,
                          size_t bytes,
                          uint32_t data,
                          VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_fill(ptr, bytes, data);
    profile->buffer_fill_calls++;
    profile->buffer_fill_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_fill(ptr, bytes, data);
  }
}

template <typename DstPtr, typename SrcPtr>
void profiled_buffer_copy(CommandList *cmdlist,
                          DstPtr dst,
                          SrcPtr src,
                          size_t bytes,
                          VulkanSortCpuProfileSample *profile) {
  if (profile) {
    double start = profile_time_us();
    cmdlist->buffer_copy(dst, src, bytes);
    profile->buffer_copy_calls++;
    profile->buffer_copy_us += profile_time_us() - start;
  } else {
    cmdlist->buffer_copy(dst, src, bytes);
  }
}

std::vector<size_t> scan_level_lengths(size_t n) {
  std::vector<size_t> levels;
  while (n > 0) {
    levels.push_back(n);
    if (n <= kBlockSize) {
      break;
    }
    n = (n + kBlockSize - 1) / kBlockSize;
  }
  return levels;
}

size_t scan_workspace_bytes(const std::vector<size_t> &levels,
                            size_t item_size) {
  size_t items = 0;
  for (size_t i = 1; i < levels.size(); ++i) {
    items += levels[i];
  }
  return items * item_size;
}

size_t vulkan_scan_value_type_size(int value_type) {
  switch (value_type) {
    case 0:
    case 1:
    case 2:
      return sizeof(uint32_t);
    case 3:
    case 4:
    case 5:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

size_t strided_binding_bytes(size_t n,
                             size_t value_size,
                             size_t offset,
                             size_t stride) {
  if (n == 0) {
    return value_size;
  }
  return offset + (n - 1) * stride + value_size;
}

size_t vulkan_sort_key_type_size(int key_type) {
  switch (key_type) {
    case 0:
    case 1:
    case 2:
      return sizeof(uint32_t);
    case 3:
    case 4:
    case 5:
      return sizeof(uint64_t);
    default:
      return 0;
  }
}

DataType vulkan_sort_key_data_type(int key_type) {
  switch (key_type) {
    case 0:
      return PrimitiveType::u32;
    case 1:
      return PrimitiveType::i32;
    case 2:
      return PrimitiveType::f32;
    case 3:
      return PrimitiveType::u64;
    case 4:
      return PrimitiveType::i64;
    case 5:
      return PrimitiveType::f64;
    default:
      return nullptr;
  }
}

DataType vulkan_sort_value_data_type(int value_type) {
  switch (value_type) {
    case 0:
      return PrimitiveType::i32;
    case 1:
      return PrimitiveType::f32;
    case 2:
      return PrimitiveType::u32;
    case 3:
      return PrimitiveType::u64;
    case 4:
      return PrimitiveType::i64;
    case 5:
      return PrimitiveType::f64;
    default:
      return nullptr;
  }
}

DevicePtr scan_level_ptr(DeviceAllocation data_alloc,
                         DeviceAllocation workspace,
                         const std::vector<size_t> &workspace_offsets,
                         size_t level,
                         size_t item_size,
                         size_t data_offset = 0) {
  if (level == 0) {
    return data_alloc.get_ptr(data_offset);
  }
  return workspace.get_ptr(workspace_offsets[level - 1] * item_size);
}

struct VulkanScanDispatchPlan {
  DeviceAllocation data_alloc{kDeviceNullAllocation};
  size_t data_offset{0};
  size_t n{0};
  int value_type{0};
  size_t item_size{sizeof(int32_t)};
  size_t workspace_bytes{0};
  bool use_small_subgroup{false};
  bool reverse{false};
  size_t data_bytes{0};
  DeviceAllocation workspace_alloc{kDeviceNullAllocation};
  DeviceAllocation dummy_sums_alloc{kDeviceNullAllocation};
  DeviceAllocation params_alloc{kDeviceNullAllocation};
  std::vector<size_t> levels;
  std::vector<size_t> workspace_offsets;
  Pipeline *scan_small{nullptr};
  Pipeline *scan_block{nullptr};
  Pipeline *scan_add{nullptr};
  Pipeline *scan_block_strided{nullptr};
  Pipeline *scan_add_strided{nullptr};
  Pipeline *scan_small_reverse{nullptr};
  Pipeline *scan_block_reverse{nullptr};
  Pipeline *scan_add_reverse{nullptr};
  const char *scan_small_scope{nullptr};
  const char *scan_block_scope{nullptr};
  const char *scan_add_scope{nullptr};
  const char *scan_block_strided_scope{nullptr};
  const char *scan_add_strided_scope{nullptr};
  const char *scan_small_reverse_scope{nullptr};
  const char *scan_block_reverse_scope{nullptr};
  const char *scan_add_reverse_scope{nullptr};
  bool member_source{false};
  size_t offset{0};
  size_t stride{0};
  size_t params_bytes{0};
  ShaderResourceSet *small_bindings{nullptr};
  std::vector<ShaderResourceSet *> block_bindings;
  std::vector<ShaderResourceSet *> add_bindings;
};

VulkanCommandReplayKey make_vulkan_compact_fused_command_key(
    bool dense_field,
    int value_type,
    DeviceAllocation values_alloc,
    uint64_t values_offset,
    size_t value_total_bytes,
    DeviceAllocation flags_alloc,
    uint64_t flags_offset,
    DeviceAllocation output_alloc,
    uint64_t output_offset,
    DeviceAllocation count_alloc,
    uint64_t count_offset,
    DeviceAllocation prefix_alloc,
    size_t prefix_bytes,
    uint32_t flag_groups,
    uint32_t word_groups,
    Pipeline *flags_pipeline,
    Pipeline *scatter_pipeline,
    ShaderResourceSet *flags_resource_set,
    ShaderResourceSet *scatter_resource_set,
    const VulkanScanDispatchPlan &scan_plan) {
  VulkanCommandReplayKey key;
  key.push(dense_field ? 1 : 0);
  key.push(static_cast<uint64_t>(value_type));
  key.push(values_alloc.alloc_id);
  key.push(vulkan_allocation_generation(values_alloc));
  key.push(values_offset);
  key.push(static_cast<uint64_t>(value_total_bytes));
  key.push(flags_alloc.alloc_id);
  key.push(vulkan_allocation_generation(flags_alloc));
  key.push(flags_offset);
  key.push(output_alloc.alloc_id);
  key.push(vulkan_allocation_generation(output_alloc));
  key.push(output_offset);
  key.push(count_alloc.alloc_id);
  key.push(vulkan_allocation_generation(count_alloc));
  key.push(count_offset);
  key.push(prefix_alloc.alloc_id);
  key.push(vulkan_allocation_generation(prefix_alloc));
  key.push(static_cast<uint64_t>(prefix_bytes));
  key.push(flag_groups);
  key.push(word_groups);
  key.push_ptr(flags_pipeline);
  key.push_ptr(scatter_pipeline);
  key.push_ptr(flags_resource_set);
  key.push_ptr(scatter_resource_set);
  key.push(scan_plan.data_alloc.alloc_id);
  key.push(vulkan_allocation_generation(scan_plan.data_alloc));
  key.push(static_cast<uint64_t>(scan_plan.data_offset));
  key.push(static_cast<uint64_t>(scan_plan.n));
  key.push(static_cast<uint64_t>(scan_plan.workspace_bytes));
  key.push(scan_plan.workspace_alloc.alloc_id);
  key.push(vulkan_allocation_generation(scan_plan.workspace_alloc));
  key.push_ptr(scan_plan.scan_block);
  key.push_ptr(scan_plan.scan_add);
  key.push_ptr(scan_plan.small_bindings);
  return key;
}

VulkanScanDispatchPlan prepare_vulkan_scan(Program *program,
                                           VulkanScanCache &cache,
                                           DeviceAllocation data_alloc,
                                           size_t n,
                                           int value_type,
                                           bool member_source = false,
                                           size_t offset = 0,
                                           size_t stride = 0,
                                           size_t data_offset = 0,
                                           bool reverse = false) {
  VulkanScanDispatchPlan plan;
  plan.data_alloc = data_alloc;
  plan.data_offset = data_offset;
  plan.n = n;
  plan.value_type = value_type;
  plan.item_size = vulkan_scan_value_type_size(value_type);
  plan.member_source = member_source;
  plan.reverse = reverse;
  plan.offset = offset;
  plan.stride = stride;
  TI_ERROR_IF(plan.item_size == 0,
              "Vulkan native scan received an unsupported value type.");
  if (n <= 1) {
    return plan;
  }
  if (member_source) {
    cache.ensure_params();
    plan.params_alloc = cache.params;
    plan.params_bytes = 3 * sizeof(uint32_t);
  }

  const int small_subgroup_threshold =
      get_environ_config("TI_VULKAN_SCAN_SMALL_SUBGROUP_MAX_N", 4096);
  const bool use_32bit_value = plan.item_size == sizeof(uint32_t);
  const bool small_subgroup_supported = use_32bit_value &&
                                        cache.subgroup_scan_enabled &&
                                        value_type >= 0 && value_type <= 2;
  plan.use_small_subgroup =
      small_subgroup_supported && small_subgroup_threshold > 0 &&
      n <= static_cast<size_t>(small_subgroup_threshold);
  if (plan.use_small_subgroup) {
    if (reverse) {
      cache.ensure_scan_reverse_small_pipeline(cache.device, value_type,
                                               member_source);
    } else if (member_source) {
      cache.ensure_scan_strided_small_pipeline(cache.device, value_type);
    } else {
      cache.ensure_scan_small_pipeline(cache.device, value_type);
    }
    plan.scan_small = reverse
                          ? cache.scan_small_reverse_pipeline(value_type,
                                                             member_source)
                          : (member_source
                                 ? cache.scan_small_strided_pipeline(value_type)
                                 : cache.scan_small_pipeline(value_type));
    plan.scan_small_scope = member_source
                                ? "vulkan_scan_small_subgroup_strided"
                                : cache.scan_small_scope(value_type);
    plan.scan_small_reverse = plan.scan_small;
    plan.scan_small_reverse_scope =
        member_source ? "vulkan_scan_small_subgroup_strided_reverse"
                      : "vulkan_scan_small_subgroup_reverse";
    plan.data_bytes =
        member_source ? strided_binding_bytes(n, plan.item_size, offset, stride)
                      : n * plan.item_size;
    TI_ERROR_IF(!plan.scan_small,
                "Vulkan native scan could not find a small-scan pipeline.");
    cache.prepare_resource_sets(program, member_source ? 0 : 1,
                                member_source ? 1 : 0, 0, 0, 0, 0);
    plan.small_bindings = cache.next_small_resource_set(member_source);
    return plan;
  }

  plan.levels = scan_level_lengths(n);
  plan.workspace_bytes = scan_workspace_bytes(plan.levels, plan.item_size);
  if (cache.has_workspace_allocs() &&
      cache.needs_workspace_realloc(plan.workspace_bytes)) {
    program->synchronize();
  }
  cache.ensure_workspace(plan.workspace_bytes);

  plan.workspace_offsets.reserve(plan.levels.size() > 0 ? plan.levels.size() - 1
                                                        : 0);
  size_t workspace_offset = 0;
  for (size_t i = 1; i < plan.levels.size(); ++i) {
    plan.workspace_offsets.push_back(workspace_offset);
    workspace_offset += plan.levels[i];
  }

  plan.workspace_alloc = cache.workspace;
  plan.dummy_sums_alloc = cache.dummy_sums;
  const int subgroup_block_min_n_config =
      get_environ_config("TI_VULKAN_SCAN_SUBGROUP_BLOCK_MIN_N", 1048576);
  const size_t subgroup_block_min_n =
      subgroup_block_min_n_config <= 0
          ? 0
          : static_cast<size_t>(subgroup_block_min_n_config);
  const bool use_subgroup_block = cache.subgroup_scan_enabled &&
                                  !member_source &&
                                  value_type >= 0 && value_type <= 2 &&
                                  !reverse &&
                                  n >= subgroup_block_min_n;
  if (!member_source) {
    cache.ensure_scan_block_pipeline(cache.device, value_type,
                                     use_subgroup_block);
  } else {
    cache.ensure_scan_strided_block_add_pipelines(cache.device, value_type);
    cache.ensure_scan_block_pipeline(cache.device, value_type, false);
  }
  if (reverse) {
    cache.ensure_scan_reverse_block_add_pipelines(cache.device, value_type,
                                                  member_source);
  }
  plan.scan_block = cache.scan_block_pipeline(value_type, use_subgroup_block);
  plan.scan_block_scope =
      cache.scan_block_scope(value_type, use_subgroup_block);
  plan.scan_add = cache.scan_add_pipeline(value_type);
  plan.scan_add_scope = cache.scan_add_scope(value_type);
  if (member_source) {
    plan.scan_block_strided = cache.scan_block_strided_pipeline(value_type);
    plan.scan_add_strided = cache.scan_add_strided_pipeline(value_type);
    plan.scan_block_strided_scope = "vulkan_scan_block_strided";
    plan.scan_add_strided_scope = "vulkan_scan_add_strided";
  }
  if (reverse) {
    plan.scan_block_reverse =
        cache.scan_block_reverse_pipeline(value_type, member_source);
    plan.scan_add_reverse =
        cache.scan_add_reverse_pipeline(value_type, member_source);
    plan.scan_block_reverse_scope =
        member_source ? "vulkan_scan_block_strided_reverse"
                      : "vulkan_scan_block_reverse";
    plan.scan_add_reverse_scope =
        member_source ? "vulkan_scan_add_strided_reverse"
                      : "vulkan_scan_add_reverse";
  }
  TI_ERROR_IF(!plan.scan_block || !plan.scan_add ||
                  (member_source &&
                   (!plan.scan_block_strided || !plan.scan_add_strided)) ||
                  (reverse &&
                   (!plan.scan_block_reverse || !plan.scan_add_reverse)),
              "Vulkan native scan could not find a scan pipeline.");
  const size_t block_count = plan.levels.size();
  const size_t add_count = plan.levels.size() > 1 ? plan.levels.size() - 1 : 0;
  const size_t block_strided_count = member_source && block_count > 0 ? 1 : 0;
  const size_t add_strided_count = member_source && add_count > 0 ? 1 : 0;
  cache.prepare_resource_sets(program, 0, 0,
                              block_count - block_strided_count,
                              block_strided_count,
                              add_count - add_strided_count,
                              add_strided_count);
  plan.block_bindings.reserve(block_count);
  for (size_t level = 0; level < block_count; ++level) {
    const bool strided_level = member_source && level == 0;
    plan.block_bindings.push_back(
        cache.next_block_resource_set(strided_level));
  }
  plan.add_bindings.reserve(add_count);
  if (add_count > 0) {
    for (size_t level = plan.levels.size() - 1; level-- > 0;) {
      const bool strided_level = member_source && level == 0;
      plan.add_bindings.push_back(cache.next_add_resource_set(strided_level));
    }
  }
  return plan;
}

VulkanScanDispatchPlan prepare_vulkan_i32_scan(Program *program,
                                               VulkanScanCache &cache,
                                               DeviceAllocation data_alloc,
                                               size_t n) {
  return prepare_vulkan_scan(program, cache, data_alloc, n, 0);
}

void record_vulkan_scan(Device * /*op_device*/,
                        CommandList *cmdlist,
                        const VulkanScanDispatchPlan &plan,
                        bool profiler_scopes) {
  if (plan.n <= 1) {
    return;
  }
  const std::array<uint32_t, 3> param_words{
      static_cast<uint32_t>(plan.n),
      static_cast<uint32_t>(plan.offset / sizeof(uint32_t)),
      static_cast<uint32_t>(plan.stride / sizeof(uint32_t)),
  };
  auto bind_params = [&plan, &param_words, cmdlist](ShaderResourceSet *bindings) {
    if (!plan.member_source) {
      return;
    }
    for (uint32_t i = 0; i < param_words.size(); ++i) {
      cmdlist->buffer_fill(plan.params_alloc.get_ptr(i * sizeof(uint32_t)),
                           sizeof(uint32_t), param_words[i]);
    }
    cmdlist->buffer_barrier(plan.params_alloc);
    bindings->rw_buffer(plan.use_small_subgroup ? 1 : 2,
                        plan.params_alloc.get_ptr(0), plan.params_bytes);
  };
  if (plan.use_small_subgroup) {
    ShaderResourceSet *bindings = plan.small_bindings;
    TI_ERROR_IF(!bindings, "Vulkan scan missing replay resource set.");
    bindings->rw_buffer(0, plan.data_alloc.get_ptr(plan.data_offset),
                        plan.data_bytes);
    bind_params(bindings);
    dispatch_pipeline(cmdlist, plan.scan_small, bindings, 1, 1, 1,
                      profiler_scopes
                          ? (plan.reverse ? plan.scan_small_reverse_scope
                                          : plan.scan_small_scope)
                          : nullptr);
    cmdlist->buffer_barrier(plan.data_alloc);
    return;
  }

  auto scope_name = [profiler_scopes](const char *name) {
    return profiler_scopes ? name : nullptr;
  };
  auto barrier_level = [&plan](CommandList *cmdlist, size_t level) {
    if (level == 0) {
      cmdlist->buffer_barrier(plan.data_alloc);
    } else {
      cmdlist->buffer_barrier(plan.workspace_alloc);
    }
  };
  for (size_t level = 0; level < plan.levels.size(); ++level) {
    DevicePtr level_ptr =
        scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                       plan.workspace_offsets, level, plan.item_size,
                       plan.data_offset);
    const bool strided_level = plan.member_source && level == 0;
    const size_t level_bytes =
        strided_level ? strided_binding_bytes(plan.n, plan.item_size,
                                              plan.offset, plan.stride)
                      : plan.levels[level] * plan.item_size;
    DevicePtr sums_ptr = plan.dummy_sums_alloc.get_ptr(0);
    size_t sums_bytes = plan.item_size;
    if (level + 1 < plan.levels.size()) {
      sums_ptr = scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                                plan.workspace_offsets, level + 1,
                                plan.item_size);
      sums_bytes = plan.levels[level + 1] * plan.item_size;
    }
    TI_ERROR_IF(level >= plan.block_bindings.size(),
                "Vulkan scan missing block replay resource set.");
    ShaderResourceSet *bindings = plan.block_bindings[level];
    bindings->rw_buffer(0, level_ptr, level_bytes);
    bindings->rw_buffer(1, sums_ptr, sums_bytes);
    if (strided_level) {
      bind_params(bindings);
    }
    const uint32_t groups = static_cast<uint32_t>(
        (plan.levels[level] + kBlockSize - 1) / kBlockSize);
    const bool reverse_level = plan.reverse && level == 0;
    Pipeline *pipeline = reverse_level
                             ? plan.scan_block_reverse
                             : (strided_level ? plan.scan_block_strided
                                              : plan.scan_block);
    const char *scope = reverse_level
                            ? plan.scan_block_reverse_scope
                            : (strided_level ? plan.scan_block_strided_scope
                                             : plan.scan_block_scope);
    dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                      scope_name(scope));
    barrier_level(cmdlist, level);
    if (level + 1 < plan.levels.size()) {
      cmdlist->buffer_barrier(plan.workspace_alloc);
    }
  }
  if (plan.levels.size() > 1) {
    size_t add_binding_index = 0;
    for (size_t level = plan.levels.size() - 1; level-- > 0;) {
      DevicePtr level_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level, plan.item_size,
                         plan.data_offset);
      DevicePtr offsets_ptr =
          scan_level_ptr(plan.data_alloc, plan.workspace_alloc,
                         plan.workspace_offsets, level + 1, plan.item_size);
      const bool strided_level = plan.member_source && level == 0;
      const size_t level_bytes =
          strided_level ? strided_binding_bytes(plan.n, plan.item_size,
                                                plan.offset, plan.stride)
                        : plan.levels[level] * plan.item_size;
      const size_t offsets_bytes = plan.levels[level + 1] * plan.item_size;
      TI_ERROR_IF(add_binding_index >= plan.add_bindings.size(),
                  "Vulkan scan missing add replay resource set.");
      ShaderResourceSet *bindings = plan.add_bindings[add_binding_index++];
      bindings->rw_buffer(0, level_ptr, level_bytes);
      bindings->rw_buffer(1, offsets_ptr, offsets_bytes);
      if (strided_level) {
        bind_params(bindings);
      }
      const uint32_t groups = static_cast<uint32_t>(
          (plan.levels[level] + kBlockSize - 1) / kBlockSize);
      const bool reverse_level = plan.reverse && level == 0;
      Pipeline *pipeline = reverse_level
                               ? plan.scan_add_reverse
                               : (strided_level ? plan.scan_add_strided
                                                : plan.scan_add);
      const char *scope = reverse_level
                              ? plan.scan_add_reverse_scope
                              : (strided_level ? plan.scan_add_strided_scope
                                               : plan.scan_add_scope);
      dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                        scope_name(scope));
      barrier_level(cmdlist, level);
    }
  }
}

void record_vulkan_i32_scan(Device *op_device,
                            CommandList *cmdlist,
                            const VulkanScanDispatchPlan &plan,
                            bool profiler_scopes) {
  record_vulkan_scan(op_device, cmdlist, plan, profiler_scopes);
}

size_t enqueue_vulkan_scan(Program *program,
                           VulkanScanCache &cache,
                           DeviceAllocation data_alloc,
                           size_t n,
                           int value_type,
                           bool profiler_scopes,
                           bool member_source = false,
                           size_t offset = 0,
                           size_t stride = 0,
                           size_t data_offset = 0,
                           bool reverse = false) {
  auto plan = prepare_vulkan_scan(program, cache, data_alloc, n, value_type,
                                  member_source, offset, stride, data_offset,
                                  reverse);
  if (plan.n <= 1) {
    return 0;
  }
  auto record_scan =
      [plan, profiler_scopes](Device *op_device, CommandList *cmdlist) {
        record_vulkan_scan(op_device, cmdlist, plan, profiler_scopes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(60);
  command_key.push(reverse ? 1 : 0);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(member_source ? 1 : 0);
  command_key.push(data_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(data_alloc));
  command_key.push(static_cast<uint64_t>(data_offset));
  command_key.push(static_cast<uint64_t>(n));
  command_key.push(static_cast<uint64_t>(plan.item_size));
  command_key.push(static_cast<uint64_t>(plan.data_bytes));
  command_key.push(plan.workspace_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(plan.workspace_alloc));
  command_key.push(static_cast<uint64_t>(plan.workspace_bytes));
  command_key.push(plan.dummy_sums_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(plan.dummy_sums_alloc));
  command_key.push(plan.params_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(plan.params_alloc));
  command_key.push(static_cast<uint64_t>(plan.params_bytes));
  command_key.push(static_cast<uint64_t>(offset));
  command_key.push(static_cast<uint64_t>(stride));
  command_key.push(plan.use_small_subgroup ? 1 : 0);
  command_key.push(static_cast<uint64_t>(plan.levels.size()));
  command_key.push(static_cast<uint64_t>(plan.workspace_offsets.size()));
  command_key.push_ptr(plan.scan_small);
  command_key.push_ptr(plan.scan_block);
  command_key.push_ptr(plan.scan_add);
  command_key.push_ptr(plan.scan_block_strided);
  command_key.push_ptr(plan.scan_add_strided);
  command_key.push_ptr(plan.scan_small_reverse);
  command_key.push_ptr(plan.scan_block_reverse);
  command_key.push_ptr(plan.scan_add_reverse);
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scan requires a compute device.");
  if (!cache.scan_command_replay.submit_or_record(program, device, command_key,
                                                  profiler_scopes,
                                                  record_scan)) {
    program->enqueue_compute_op_lambda(record_scan, {});
  }
  return plan.workspace_bytes;
}

size_t enqueue_vulkan_i32_scan(Program *program,
                               VulkanScanCache &cache,
                               DeviceAllocation data_alloc,
                               size_t n,
                               bool profiler_scopes) {
  return enqueue_vulkan_scan(program, cache, data_alloc, n, 0,
                             profiler_scopes);
}

}  // namespace

bool Program::vulkan_radix_sort_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scan_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scan_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_compact_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_histogram_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_histogram_value_type_available(int value_type,
                                                    int bin_type) const {
  if (compile_config().arch != Arch::vulkan ||
      !(value_type == 0 || value_type == 2)) {
    return false;
  }
  if (bin_type == 0) {
    return true;
  }
  if (bin_type == 4) {
    const auto caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  return false;
}

bool Program::vulkan_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_reduce_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_check_count_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_check_count_value_type_available(int value_type) const {
  return vulkan_reduce_value_type_available(value_type);
}

bool Program::vulkan_metric_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_metric_reduce_value_type_available(int value_type) const {
  return compile_config().arch == Arch::vulkan && value_type == 1;
}

bool Program::vulkan_sparse_algebra_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_sparse_assembly_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return false;
  }
  Device *device = program_impl_->get_compute_device();
  return device &&
         device->get_caps().get(DeviceCapability::spirv_has_int64) != 0;
}

VulkanSparseAssemblyDispatchInfo Program::vulkan_sparse_assemble_csr(
    Ndarray *packed_triplets,
    Ndarray *triplet_rows,
    Ndarray *triplet_columns,
    Ndarray *triplet_values,
    Ndarray *sorted_keys,
    Ndarray *sorted_values,
    Ndarray *segment_ids,
    Ndarray *unique_keys,
    Ndarray *segment_offsets,
    Ndarray *unique_values,
    Ndarray *row_offsets,
    Ndarray *column_indices,
    Ndarray *active_count,
    Ndarray *control,
    std::size_t capacity,
    std::size_t rows,
    std::size_t cols) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_assembly_available(),
              "Vulkan sparse assembly requires Vulkan shaderInt64 support.");
  TI_ERROR_IF(capacity == 0 ||
                  capacity >=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Vulkan sparse assembly capacity must be in [1, INT_MAX).");
  TI_ERROR_IF(rows == 0 || cols == 0 ||
                  rows >=
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                  cols >
                      static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Vulkan sparse assembly rows must be in [1, INT_MAX), and "
              "columns in [1, INT_MAX].");
  const bool packed_input = packed_triplets != nullptr;
  TI_ERROR_IF(
      packed_input
          ? (triplet_rows || triplet_columns || triplet_values)
          : (!triplet_rows || !triplet_columns || !triplet_values),
      "Vulkan sparse assembly requires exactly one input layout: packed "
      "builder storage or three separate triplet arrays.");
  std::vector<Ndarray *> arrays{
      sorted_keys,     sorted_values,  segment_ids, unique_keys,
      segment_offsets, unique_values,  row_offsets, column_indices,
      active_count,    control};
  if (packed_input) {
    arrays.push_back(packed_triplets);
  } else {
    arrays.push_back(triplet_rows);
    arrays.push_back(triplet_columns);
    arrays.push_back(triplet_values);
  }
  for (Ndarray *array : arrays) {
    TI_ERROR_IF(!array,
                "Vulkan sparse assembly received a null ndarray.");
    TI_ERROR_IF(array->shape.size() != 1 ||
                    !array->get_element_shape().empty(),
                "Vulkan sparse assembly expects scalar 1D ndarrays.");
  }
  auto check_array = [](const char *name, Ndarray *array, DataType dtype,
                        std::size_t count, std::size_t item_bytes) {
    TI_ERROR_IF(array->get_element_data_type() != dtype ||
                    array->get_nelement() != count ||
                    array->get_element_size() != item_bytes,
                "Vulkan sparse assembly {} has an unexpected dtype, shape, "
                "or byte width; expected exactly {} scalar entries.",
                name, count);
  };
  if (packed_input) {
    check_array("packed triplets", packed_triplets, PrimitiveType::i32,
                capacity * 3 + 2, sizeof(int32_t));
  } else {
    check_array("triplet rows", triplet_rows, PrimitiveType::i32, capacity,
                sizeof(int32_t));
    check_array("triplet columns", triplet_columns, PrimitiveType::i32,
                capacity, sizeof(int32_t));
    check_array("triplet values", triplet_values, PrimitiveType::f32,
                capacity, sizeof(float32));
  }
  check_array("sorted keys", sorted_keys, PrimitiveType::u64, capacity,
              sizeof(uint64_t));
  check_array("sorted values", sorted_values, PrimitiveType::f32, capacity,
              sizeof(float32));
  check_array("segment ids", segment_ids, PrimitiveType::i32, capacity,
              sizeof(int32_t));
  check_array("unique keys", unique_keys, PrimitiveType::u64, capacity,
              sizeof(uint64_t));
  check_array("segment offsets", segment_offsets, PrimitiveType::i32,
              capacity + 1, sizeof(int32_t));
  check_array("unique values", unique_values, PrimitiveType::f32, capacity,
              sizeof(float32));
  check_array("row offsets", row_offsets, PrimitiveType::i32, rows + 1,
              sizeof(int32_t));
  check_array("column indices", column_indices, PrimitiveType::i32, capacity,
              sizeof(int32_t));
  check_array("active count", active_count, PrimitiveType::i32, 1,
              sizeof(int32_t));
  check_array("control", control, PrimitiveType::i32, 2, sizeof(int32_t));

  // Preserve the original output indices so replay keys stay easy to audit:
  // 0..2 inputs, 3..11 staging/control, 12 active count, 13 packed input.
  std::array<DeviceAllocation, 14> allocs{};
  if (packed_input) {
    allocs[13] = packed_triplets->get_device_allocation();
  } else {
    allocs[0] = triplet_rows->get_device_allocation();
    allocs[1] = triplet_columns->get_device_allocation();
    allocs[2] = triplet_values->get_device_allocation();
  }
  const std::array<Ndarray *, 10> outputs{
      sorted_keys,    sorted_values, segment_ids, unique_keys,
      segment_offsets, unique_values, row_offsets, column_indices,
      control,        active_count};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const std::size_t allocation_index = i < 8 ? i + 3 : (i == 8 ? 11 : 12);
    allocs[allocation_index] = outputs[i]->get_device_allocation();
  }
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    if (allocs[i] == kDeviceNullAllocation) {
      continue;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (allocs[j] == kDeviceNullAllocation) {
        continue;
      }
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse assembly buffers must not alias.");
    }
  }
  std::vector<const Ndarray *> resources(arrays.begin(), arrays.end());
  retain_ndarrays_for_external_submission(resources.data(), resources.size());

  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse assembly requires a compute device.");
  constexpr int kKeyTypeU64 = 3;
  constexpr int kValueTypeF32 = 1;
  constexpr bool kUseValues = true;
  constexpr bool kUseIndexSort = true;
  constexpr std::size_t kSortValueBytes = sizeof(float32);
  constexpr std::size_t kSortKeyBytes = sizeof(uint64_t);
  bool sort_growth_sync = false;
  bool scan_growth_sync = false;
  bool use_radix8 = false;
  const std::size_t max_scan_count = std::max(capacity, rows + 1);
  const auto scan_levels = scan_level_lengths(max_scan_count);
  const std::size_t scan_workspace_request =
      scan_workspace_bytes(scan_levels, sizeof(int32_t));
  {
    auto sort_cache_lease = get_cache(this, device);
    auto &sort_cache = *sort_cache_lease;
    use_radix8 = sort_cache.radix8_enabled;
    sort_growth_sync =
        sort_cache.has_workspace_allocs() &&
        sort_cache.needs_workspace_realloc(
            capacity, kUseValues, use_radix8, kSortValueBytes,
            kSortKeyBytes, kUseIndexSort);
  }
  {
    auto scan_cache_lease = get_scan_cache(this, device);
    auto &scan_cache = *scan_cache_lease;
    scan_growth_sync =
        scan_cache.has_workspace_allocs() &&
        scan_cache.needs_workspace_realloc(scan_workspace_request);
  }
  VulkanSparseAssemblyDispatchInfo result;
  result.workspace_growth_synchronized =
      sort_growth_sync || scan_growth_sync;
  if (result.workspace_growth_synchronized) {
    synchronize();
  }
  {
    auto sort_cache_lease = get_cache(this, device);
    auto &sort_cache = *sort_cache_lease;
    sort_cache.ensure_workspace(capacity, kUseValues, use_radix8,
                                kSortValueBytes, kSortKeyBytes,
                                kUseIndexSort);
    result.radix_sort_workspace_bytes = sort_cache.cached_bytes;
  }
  {
    auto scan_cache_lease = get_scan_cache(this, device);
    auto &scan_cache = *scan_cache_lease;
    scan_cache.ensure_workspace(scan_workspace_request);
    result.scan_workspace_bytes = scan_cache.allocated_bytes();
  }

  fill_ndarray_fast_u32(control, 0);
  const uint32_t groups = static_cast<uint32_t>(
      (capacity + kBlockSize - 1) / kBlockSize);
  const std::array<uint32_t, 4> assembly_params{
      static_cast<uint32_t>(capacity), static_cast<uint32_t>(rows),
      static_cast<uint32_t>(cols), 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(assembly_params.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;

  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline =
        packed_input ? cache.sparse_triplet_pack_packed_pipeline(device)
                     : cache.sparse_triplet_pack_pipeline(device);
    ShaderResourceSet *bindings =
        packed_input
            ? cache
                  .bind_sparse_triplet_pack_packed(
                      this, allocs[13], allocs[3], allocs[4], allocs[12],
                      allocs[11], capacity)
                  .bindings
            : cache
                  .bind_sparse_triplet_pack(
                      this, allocs[0], allocs[1], allocs[2], allocs[3],
                      allocs[4], allocs[12], allocs[11], capacity)
                  .bindings;
    auto record = [allocs, pipeline, bindings, assembly_params, push_bytes,
                   profiler_scopes, packed_input](Device *,
                                                   CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, assembly_params.data(), push_bytes,
          static_cast<uint32_t>((assembly_params[0] + kBlockSize - 1) /
                                kBlockSize),
          1, 1,
          profiler_scopes
              ? (packed_input ? "vulkan_sparse_triplet_pack_packed_f32"
                              : "vulkan_sparse_triplet_pack_f32")
              : nullptr);
      cmdlist->buffer_barrier(allocs[3]);
      cmdlist->buffer_barrier(allocs[4]);
      cmdlist->buffer_barrier(allocs[12]);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(packed_input ? 153 : 148);
    if (packed_input) {
      for (std::size_t i : {13u, 3u, 4u, 12u, 11u}) {
        key.push(allocs[i].alloc_id);
        key.push(vulkan_allocation_generation(allocs[i]));
      }
    } else {
      for (std::size_t i : {0u, 1u, 2u, 3u, 4u, 12u, 11u}) {
        key.push(allocs[i].alloc_id);
        key.push(vulkan_allocation_generation(allocs[i]));
      }
    }
    for (uint32_t word : assembly_params) {
      key.push(word);
    }
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    auto &replay =
        packed_input ? cache.sparse_triplet_pack_packed_command_replay
                     : cache.sparse_triplet_pack_command_replay;
    if (!replay.submit_or_record(this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }

  vulkan_radix_sort_u32_ndarray(sorted_keys, sorted_values, kKeyTypeU64,
                                kValueTypeF32);

  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline = cache.sparse_segment_flags_pipeline(device);
    ShaderResourceSet *bindings =
        cache
            .bind_sparse_segment_flags(this, allocs[3], allocs[5], allocs[12],
                                       allocs[11], capacity)
            .bindings;
    const std::array<uint32_t, 4> params{
        static_cast<uint32_t>(capacity), 0u, 0u, 0u};
    auto record = [allocs, pipeline, bindings, params, push_bytes, groups,
                   profiler_scopes](Device *, CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, params.data(), push_bytes, groups, 1, 1,
          profiler_scopes ? "vulkan_sparse_segment_flags_u64" : nullptr);
      cmdlist->buffer_barrier(allocs[5]);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(149);
    for (std::size_t i : {3u, 5u, 12u, 11u}) {
      key.push(allocs[i].alloc_id);
      key.push(vulkan_allocation_generation(allocs[i]));
    }
    key.push(static_cast<uint64_t>(capacity));
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    if (!cache.sparse_segment_flags_command_replay.submit_or_record(
            this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }

  vulkan_inclusive_scan_ndarray(segment_ids, 0);

  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline = cache.sparse_segment_scatter_pipeline(device);
    ShaderResourceSet *bindings =
        cache
            .bind_sparse_segment_scatter(this, allocs[3], allocs[5],
                                         allocs[6], allocs[7], allocs[12],
                                         allocs[11], capacity)
            .bindings;
    const std::array<uint32_t, 4> params{
        static_cast<uint32_t>(capacity), 0u, 0u, 0u};
    auto record = [allocs, pipeline, bindings, params, push_bytes, groups,
                   profiler_scopes](Device *, CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, params.data(), push_bytes, groups, 1, 1,
          profiler_scopes ? "vulkan_sparse_segment_scatter_u64" : nullptr);
      cmdlist->buffer_barrier(allocs[6]);
      cmdlist->buffer_barrier(allocs[7]);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(150);
    for (std::size_t i : {3u, 5u, 6u, 7u, 12u, 11u}) {
      key.push(allocs[i].alloc_id);
      key.push(vulkan_allocation_generation(allocs[i]));
    }
    key.push(static_cast<uint64_t>(capacity));
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    if (!cache.sparse_segment_scatter_command_replay.submit_or_record(
            this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }

  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline = cache.sparse_segment_reduce_pipeline(device);
    ShaderResourceSet *bindings =
        cache
            .bind_sparse_segment_reduce(this, allocs[4], allocs[7], allocs[8],
                                        allocs[12], allocs[11], capacity)
            .bindings;
    const std::array<uint32_t, 4> params{
        static_cast<uint32_t>(capacity), 0u, 0u, 0u};
    auto record = [allocs, pipeline, bindings, params, push_bytes, groups,
                   profiler_scopes](Device *, CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, params.data(), push_bytes, groups, 1, 1,
          profiler_scopes ? "vulkan_sparse_segment_reduce_f32" : nullptr);
      cmdlist->buffer_barrier(allocs[8]);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(151);
    for (std::size_t i : {4u, 7u, 8u, 12u, 11u}) {
      key.push(allocs[i].alloc_id);
      key.push(vulkan_allocation_generation(allocs[i]));
    }
    key.push(static_cast<uint64_t>(capacity));
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    if (!cache.sparse_segment_reduce_command_replay.submit_or_record(
            this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }

  fill_ndarray_fast_u32(row_offsets, 0);
  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline = cache.sparse_csr_finalize_pipeline(device);
    ShaderResourceSet *bindings =
        cache
            .bind_sparse_csr_finalize(this, allocs[6], allocs[10], allocs[9],
                                      allocs[12], allocs[11], capacity, rows)
            .bindings;
    auto record = [allocs, pipeline, bindings, assembly_params, push_bytes,
                   groups, profiler_scopes](Device *, CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, assembly_params.data(), push_bytes,
          groups, 1, 1,
          profiler_scopes ? "vulkan_sparse_csr_finalize_u64" : nullptr);
      cmdlist->buffer_barrier(allocs[9]);
      cmdlist->buffer_barrier(allocs[10]);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(152);
    for (std::size_t i : {6u, 9u, 10u, 12u, 11u}) {
      key.push(allocs[i].alloc_id);
      key.push(vulkan_allocation_generation(allocs[i]));
    }
    for (uint32_t word : assembly_params) {
      key.push(word);
    }
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    if (!cache.sparse_csr_finalize_command_replay.submit_or_record(
            this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }

  vulkan_inclusive_scan_ndarray(row_offsets, 0);

  {
    auto cache_lease = get_sparse_algebra_cache(this, device);
    auto &cache = *cache_lease;
    Pipeline *pipeline =
        cache.sparse_assembly_finalize_control_pipeline(device);
    ShaderResourceSet *bindings =
        cache
            .bind_sparse_assembly_finalize_control(
                this, allocs[12], allocs[11])
            .bindings;
    const std::array<uint32_t, 4> params{
        static_cast<uint32_t>(capacity), 0u, 0u, 0u};
    auto record = [allocs, pipeline, bindings, params, push_bytes,
                   profiler_scopes](Device *, CommandList *cmdlist) {
      dispatch_pipeline_with_push_constants(
          cmdlist, pipeline, bindings, params.data(), push_bytes, 1, 1, 1,
          profiler_scopes ? "vulkan_sparse_assembly_finalize_control"
                          : nullptr);
      cmdlist->buffer_barrier(allocs[11]);
    };
    VulkanCommandReplayKey key;
    key.push(154);
    for (std::size_t i : {12u, 11u}) {
      key.push(allocs[i].alloc_id);
      key.push(vulkan_allocation_generation(allocs[i]));
    }
    key.push(static_cast<uint64_t>(capacity));
    key.push_ptr(pipeline);
    key.push_ptr(bindings);
    if (!cache.sparse_assembly_finalize_control_command_replay.submit_or_record(
            this, device, key, profiler_scopes, record)) {
      enqueue_compute_op_lambda(record, {});
    }
  }
  result.radix_sort_workspace_bytes = vulkan_radix_sort_workspace_bytes();
  result.scan_workspace_bytes = vulkan_scan_workspace_bytes();
  return result;
}

void Program::vulkan_copy_ndarray_prefix(Ndarray *dst,
                                         Ndarray *src,
                                         std::size_t bytes) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan ndarray prefix copy is only available on Vulkan.");
  TI_ERROR_IF(!dst || !src,
              "Vulkan ndarray prefix copy received a null ndarray.");
  const std::size_t dst_bytes =
      dst->get_nelement() * dst->get_element_size();
  const std::size_t src_bytes =
      src->get_nelement() * src->get_element_size();
  TI_ERROR_IF(bytes > dst_bytes || bytes > src_bytes,
              "Vulkan ndarray prefix copy of {} bytes exceeds destination {} "
              "or source {} bytes.",
              bytes, dst_bytes, src_bytes);
  if (bytes == 0 || dst == src) {
    return;
  }
  const Ndarray *resources[] = {dst, src};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const DeviceAllocation dst_alloc = dst->get_device_allocation();
  const DeviceAllocation src_alloc = src->get_device_allocation();
  enqueue_compute_op_lambda(
      [dst_alloc, src_alloc, bytes](Device *, CommandList *cmdlist) {
        cmdlist->buffer_copy(dst_alloc.get_ptr(0), src_alloc.get_ptr(0), bytes);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), bytes);
      },
      {});
  runtime_statistics().record_transfer(RuntimeTransferKind::kDeviceToDevice,
                                       bytes);
}

std::size_t Program::vulkan_csr_spmv(Ndarray *row_offsets,
                                     Ndarray *column_indices,
                                     Ndarray *values,
                                     Ndarray *x,
                                     Ndarray *y,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t nnz) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan native CSR SpMV is only available on Vulkan.");
  TI_ERROR_IF(!row_offsets || !x || !y ||
                  (nnz > 0 && (!column_indices || !values)),
              "Vulkan native CSR SpMV received inconsistent storage for nnz "
              "{}.",
              nnz);
  TI_ERROR_IF(rows == 0 || cols == 0,
              "Vulkan native CSR SpMV requires non-empty dimensions.");
  TI_ERROR_IF(rows > std::numeric_limits<uint32_t>::max() ||
                  cols > std::numeric_limits<uint32_t>::max() ||
                  nnz > std::numeric_limits<uint32_t>::max(),
              "Vulkan native CSR SpMV dimensions exceed UINT32_MAX.");
  const auto scalar_i32 = [](const Ndarray *array) {
    return array->shape.size() == 1 &&
           array->get_element_data_type() == PrimitiveType::i32 &&
           array->get_element_shape().empty() &&
           array->get_element_size() == sizeof(int32_t);
  };
  const auto scalar_f32 = [](const Ndarray *array) {
    return array->shape.size() == 1 &&
           array->get_element_data_type() == PrimitiveType::f32 &&
           array->get_element_shape().empty() &&
           array->get_element_size() == sizeof(float32);
  };
  TI_ERROR_IF(!scalar_i32(row_offsets) ||
                  row_offsets->get_nelement() != rows + 1,
              "Vulkan native CSR SpMV row offsets must contain rows + 1 "
              "scalar i32 entries.");
  TI_ERROR_IF(!scalar_f32(x) || x->get_nelement() != cols ||
                  !scalar_f32(y) || y->get_nelement() != rows,
              "Vulkan native CSR SpMV expects scalar f32 x/y vectors matching "
              "the matrix dimensions.");
  TI_ERROR_IF(x->get_device_allocation() == y->get_device_allocation(),
              "Vulkan native CSR SpMV does not support aliased x/y buffers.");
  if (nnz == 0) {
    fill_ndarray_fast_u32(y, 0);
    return 0;
  }
  TI_ERROR_IF(!scalar_i32(column_indices) ||
                  column_indices->get_nelement() != nnz,
              "Vulkan native CSR SpMV column indices must contain nnz scalar "
              "i32 entries.");
  TI_ERROR_IF(!scalar_f32(values) || values->get_nelement() != nnz,
              "Vulkan native CSR SpMV values must contain nnz scalar f32 "
              "entries.");

  const Ndarray *resources[] = {row_offsets, column_indices, values, x, y};
  retain_ndarrays_for_external_submission(resources,
                                          std::size(resources));
  const auto row_offsets_alloc = row_offsets->get_device_allocation();
  const auto column_indices_alloc = column_indices->get_device_allocation();
  const auto values_alloc = values->get_device_allocation();
  const auto x_alloc = x->get_device_allocation();
  const auto y_alloc = y->get_device_allocation();
  const size_t row_offsets_bytes = (rows + 1) * sizeof(int32_t);
  const size_t column_indices_bytes = nnz * sizeof(int32_t);
  const size_t values_bytes = nnz * sizeof(float32);
  const size_t x_bytes = cols * sizeof(float32);
  const size_t y_bytes = rows * sizeof(float32);

  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native CSR SpMV requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.csr_spmv_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_csr_spmv(this, row_offsets_alloc, row_offsets_bytes,
                         column_indices_alloc, column_indices_bytes,
                         values_alloc, values_bytes, x_alloc, x_bytes, y_alloc,
                         y_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
      static_cast<uint32_t>(nnz), 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (rows + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_spmv =
      [y_alloc, y_bytes, pipeline, bindings, param_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_csr_spmv_f32" : nullptr);
        cmdlist->buffer_barrier(y_alloc.get_ptr(0), y_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(140);
  push_vulkan_command_key_range(command_key, row_offsets_alloc, 0,
                                row_offsets_bytes);
  push_vulkan_command_key_range(command_key, column_indices_alloc, 0,
                                column_indices_bytes);
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, x_alloc, 0, x_bytes);
  push_vulkan_command_key_range(command_key, y_alloc, 0, y_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.csr_spmv_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_spmv)) {
    enqueue_compute_op_lambda(record_spmv, {});
  }
  return 0;
}

std::size_t Program::vulkan_bsr_spmv(Ndarray *row_offsets,
                                     Ndarray *column_indices,
                                     Ndarray *values,
                                     Ndarray *x,
                                     Ndarray *y,
                                     std::size_t block_rows,
                                     std::size_t block_cols,
                                     std::size_t block_nnz,
                                     std::size_t block_size) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan native BSR SpMV is only available on Vulkan.");
  TI_ERROR_IF(!row_offsets || !column_indices || !values || !x || !y,
              "Vulkan native BSR SpMV received a null storage buffer.");
  TI_ERROR_IF(block_rows == 0 || block_cols == 0 || block_nnz == 0,
              "Vulkan native BSR SpMV requires non-empty dimensions and "
              "storage.");
  TI_ERROR_IF(block_size != 2 && block_size != 3 && block_size != 6 &&
                  block_size != 12,
              "Vulkan native BSR SpMV supports block sizes 2, 3, 6, and "
              "12, got {}.",
              block_size);
  TI_ERROR_IF(block_rows > std::numeric_limits<uint32_t>::max() ||
                  block_cols > std::numeric_limits<uint32_t>::max() ||
                  block_nnz > std::numeric_limits<uint32_t>::max(),
              "Vulkan native BSR SpMV block dimensions exceed UINT32_MAX.");
  TI_ERROR_IF(
      block_rows > std::numeric_limits<std::size_t>::max() / block_size ||
          block_cols > std::numeric_limits<std::size_t>::max() / block_size ||
          block_nnz > std::numeric_limits<std::size_t>::max() / block_size /
                          block_size,
      "Vulkan native BSR SpMV scalar dimensions overflow size_t.");
  const std::size_t rows = block_rows * block_size;
  const std::size_t cols = block_cols * block_size;
  const std::size_t value_count = block_nnz * block_size * block_size;
  TI_ERROR_IF(rows > std::numeric_limits<uint32_t>::max() ||
                  cols > std::numeric_limits<uint32_t>::max() ||
                  value_count > std::numeric_limits<uint32_t>::max(),
              "Vulkan native BSR SpMV scalar dimensions exceed UINT32_MAX.");

  const auto scalar_i32 = [](const Ndarray *array) {
    return array->shape.size() == 1 &&
           array->get_element_data_type() == PrimitiveType::i32 &&
           array->get_element_shape().empty() &&
           array->get_element_size() == sizeof(int32_t);
  };
  const auto scalar_f32 = [](const Ndarray *array) {
    return array->shape.size() == 1 &&
           array->get_element_data_type() == PrimitiveType::f32 &&
           array->get_element_shape().empty() &&
           array->get_element_size() == sizeof(float32);
  };
  TI_ERROR_IF(!scalar_i32(row_offsets) ||
                  row_offsets->get_nelement() != block_rows + 1,
              "Vulkan native BSR SpMV row offsets must contain block_rows "
              "+ 1 scalar i32 entries.");
  TI_ERROR_IF(!scalar_i32(column_indices) ||
                  column_indices->get_nelement() != block_nnz,
              "Vulkan native BSR SpMV column indices must contain block_nnz "
              "scalar i32 entries.");
  TI_ERROR_IF(!scalar_f32(values) ||
                  values->get_nelement() != value_count,
              "Vulkan native BSR SpMV values have an invalid shape.");
  TI_ERROR_IF(!scalar_f32(x) || x->get_nelement() != cols ||
                  !scalar_f32(y) || y->get_nelement() != rows,
              "Vulkan native BSR SpMV expects scalar f32 x/y vectors "
              "matching the scalar matrix dimensions.");
  TI_ERROR_IF(x->get_device_allocation() == y->get_device_allocation(),
              "Vulkan native BSR SpMV does not support aliased x/y "
              "buffers.");

  const Ndarray *resources[] = {row_offsets, column_indices, values, x, y};
  retain_ndarrays_for_external_submission(resources,
                                          std::size(resources));
  const auto row_offsets_alloc = row_offsets->get_device_allocation();
  const auto column_indices_alloc = column_indices->get_device_allocation();
  const auto values_alloc = values->get_device_allocation();
  const auto x_alloc = x->get_device_allocation();
  const auto y_alloc = y->get_device_allocation();
  const size_t row_offsets_bytes =
      (block_rows + 1) * sizeof(int32_t);
  const size_t column_indices_bytes = block_nnz * sizeof(int32_t);
  const size_t values_bytes = value_count * sizeof(float32);
  const size_t x_bytes = cols * sizeof(float32);
  const size_t y_bytes = rows * sizeof(float32);

  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native BSR SpMV requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.bsr_spmv_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_bsr_spmv(this, row_offsets_alloc, row_offsets_bytes,
                         column_indices_alloc, column_indices_bytes,
                         values_alloc, values_bytes, x_alloc, x_bytes, y_alloc,
                         y_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(block_rows),
      static_cast<uint32_t>(block_cols),
      static_cast<uint32_t>(block_nnz),
      static_cast<uint32_t>(block_size)};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (rows + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_spmv =
      [y_alloc, y_bytes, pipeline, bindings, param_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_bsr_spmv_f32" : nullptr);
        cmdlist->buffer_barrier(y_alloc.get_ptr(0), y_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(141);
  push_vulkan_command_key_range(command_key, row_offsets_alloc, 0,
                                row_offsets_bytes);
  push_vulkan_command_key_range(command_key, column_indices_alloc, 0,
                                column_indices_bytes);
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, x_alloc, 0, x_bytes);
  push_vulkan_command_key_range(command_key, y_alloc, 0, y_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.bsr_spmv_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_spmv)) {
    enqueue_compute_op_lambda(record_spmv, {});
  }
  return 0;
}

bool Program::vulkan_transform_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_transform_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

namespace {

std::size_t vulkan_transform_value_size(int value_type) {
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "Vulkan native transform received an unsupported value type.");
  return (value_type == 3 || value_type == 4 || value_type == 5)
             ? sizeof(uint64_t)
             : sizeof(uint32_t);
}


void check_vulkan_transform_strided_range(const char *role,
                                          Ndarray *arr,
                                          size_t logical_items,
                                          size_t value_size,
                                          size_t offset,
                                          size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided transform {} stride is smaller than "
              "value size.",
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided transform {} offset/stride must align "
              "to value size.",
              role);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native strided transform {} offset/stride must be "
              "uint32-word aligned.",
              role);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < value_size,
              "Vulkan native strided transform {} buffer is smaller than "
              "value size.",
              role);
  TI_ERROR_IF(offset > buffer_bytes - value_size,
              "Vulkan native strided transform {} offset is out of bounds.",
              role);
  const size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > buffer_bytes,
              "Vulkan native strided transform {} range is out of bounds.",
              role);
}

void check_vulkan_transform_strided_request(Ndarray *src,
                                            Ndarray *dst,
                                            int value_type,
                                            size_t src_offset,
                                            size_t src_stride,
                                            size_t dst_offset,
                                            size_t dst_stride) {
  TI_ERROR_IF(!src || !dst,
              "Vulkan native strided transform received a null ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native strided transform source and destination sizes "
              "differ.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  check_vulkan_transform_strided_range("source", src, src->get_nelement(),
                                       value_size, src_offset, src_stride);
  check_vulkan_transform_strided_range("destination", dst, dst->get_nelement(),
                                       value_size, dst_offset, dst_stride);
}

void check_vulkan_transform_packed_strided_range(const char *role,
                                                 Ndarray *arr,
                                                 size_t logical_items,
                                                 size_t value_size,
                                                 int lane_count,
                                                 size_t offset,
                                                 size_t stride) {
  TI_ERROR_IF(lane_count <= 0,
              "Vulkan native packed strided transform lane count must be "
              "positive.");
  const size_t payload_bytes = static_cast<size_t>(lane_count) * value_size;
  TI_ERROR_IF(stride < payload_bytes,
              "Vulkan native packed strided transform {} stride is smaller "
              "than payload.",
              role);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native packed strided transform {} offset/stride must "
              "align to value size.",
              role);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native packed strided transform {} offset/stride must "
              "be uint32-word aligned.",
              role);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = arr->get_nelement() * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < payload_bytes,
              "Vulkan native packed strided transform {} buffer is smaller "
              "than payload.",
              role);
  TI_ERROR_IF(offset > buffer_bytes - payload_bytes,
              "Vulkan native packed strided transform {} offset is out of "
              "bounds.",
              role);
  const size_t last =
      offset + (logical_items - 1) * stride + payload_bytes;
  TI_ERROR_IF(last > buffer_bytes,
              "Vulkan native packed strided transform {} range is out of "
              "bounds.",
              role);
}

void check_vulkan_transform_packed_strided_request(Ndarray *src,
                                                   Ndarray *dst,
                                                   int value_type,
                                                   int lane_count,
                                                   size_t src_offset,
                                                   size_t src_stride,
                                                   size_t dst_offset,
                                                   size_t dst_stride) {
  TI_ERROR_IF(!src || !dst,
              "Vulkan native packed strided transform received a null "
              "ndarray.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native packed strided transform source and destination "
              "sizes differ.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  check_vulkan_transform_packed_strided_range(
      "source", src, src->get_nelement(), value_size, lane_count, src_offset,
      src_stride);
  check_vulkan_transform_packed_strided_range(
      "destination", dst, dst->get_nelement(), value_size, lane_count,
      dst_offset, dst_stride);
}

void check_vulkan_indexed_copy_strided_request(Ndarray *src,
                                               Ndarray *indices,
                                               Ndarray *dst,
                                               std::size_t item_bytes,
                                               std::size_t src_offset,
                                               std::size_t src_stride,
                                               std::size_t dst_offset,
                                               std::size_t dst_stride,
                                               bool scatter) {
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native strided indexed-copy received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native strided indexed-copy expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "Vulkan native strided indexed-copy expects i32 indices.");
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native strided indexed-copy item size must be a "
              "positive uint32-word multiple.");
  if (scatter) {
    TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
                "Vulkan native strided scatter expects source and indices "
                "sizes to match.");
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
                "Vulkan native strided gather expects indices and destination "
                "sizes to match.");
  }
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < item_bytes,
                "Vulkan native strided indexed-copy {} stride is smaller "
                "than item size.",
                role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "Vulkan native strided indexed-copy {} offset/stride must "
                "be uint32-word aligned.",
                role);
    if (logical_items == 0) {
      return;
    }
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < item_bytes,
                "Vulkan native strided indexed-copy {} buffer is smaller "
                "than item size.",
                role);
    TI_ERROR_IF(offset > bytes - item_bytes,
                "Vulkan native strided indexed-copy {} offset is out of "
                "bounds.",
                role);
    const std::size_t last = offset + (logical_items - 1) * stride + item_bytes;
    TI_ERROR_IF(last > bytes,
                "Vulkan native strided indexed-copy {} range is out of "
                "bounds.",
                role);
  };
  check_range("source", src, src->get_nelement(), src_offset, src_stride);
  check_range("destination", dst, dst->get_nelement(), dst_offset,
              dst_stride);
}

void check_vulkan_indexed_copy_dense_field_request(Program *program,
                                                   SNode *src,
                                                   Ndarray *indices,
                                                   SNode *dst,
                                                   int value_type,
                                                   std::size_t src_n,
                                                   std::size_t dst_n,
                                                   bool scatter) {
  TI_ERROR_IF(!program || !src || !indices || !dst,
              "Vulkan native dense field indexed-copy received a null "
              "argument.");
  TI_ERROR_IF(indices->shape.size() != 1,
              "Vulkan native dense field indexed-copy expects 1D indices.");
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "Vulkan native dense field indexed-copy expects i32 indices.");
  const std::size_t item_bytes = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native dense field indexed-copy item size must be a "
              "positive uint32-word multiple.");
  if (scatter) {
    TI_ERROR_IF(src_n != indices->get_nelement(),
                "Vulkan native dense field scatter expects source and "
                "indices sizes to match.");
  } else {
    TI_ERROR_IF(indices->get_nelement() != dst_n,
                "Vulkan native dense field gather expects indices and "
                "destination sizes to match.");
  }
  const std::size_t src_stride = program->get_dense_field_stride(src, item_bytes);
  const std::size_t dst_stride = program->get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride < item_bytes || dst_stride < item_bytes,
              "Vulkan native dense field indexed-copy received an invalid "
              "field stride.");
  TI_ERROR_IF(src_stride % sizeof(uint32_t) != 0 ||
                  dst_stride % sizeof(uint32_t) != 0,
              "Vulkan native dense field indexed-copy stride must be "
              "uint32-word aligned.");
}

void check_vulkan_indexed_copy_dense_field_indices_field_request(
    Program *program,
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    bool scatter) {
  TI_ERROR_IF(!program || !src || !indices || !dst,
              "Vulkan native dense field indexed-copy received a null "
              "argument.");
  const std::size_t item_bytes = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native dense field indexed-copy item size must be a "
              "positive uint32-word multiple.");
  if (scatter) {
    TI_ERROR_IF(src_n != indices_n,
                "Vulkan native dense field scatter expects source and "
                "indices sizes to match.");
  } else {
    TI_ERROR_IF(indices_n != dst_n,
                "Vulkan native dense field gather expects indices and "
                "destination sizes to match.");
  }
  const std::size_t src_stride = program->get_dense_field_stride(src, item_bytes);
  const std::size_t index_stride =
      program->get_dense_field_stride(indices, sizeof(int32_t));
  const std::size_t dst_stride = program->get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride < item_bytes || dst_stride < item_bytes ||
                  index_stride < sizeof(int32_t),
              "Vulkan native dense field indexed-copy received an invalid "
              "field stride.");
  TI_ERROR_IF(index_stride != sizeof(int32_t),
              "Vulkan native dense field indexed-copy currently requires "
              "contiguous i32 indices when indices are stored in a field.");
  TI_ERROR_IF(src_stride % sizeof(uint32_t) != 0 ||
                  dst_stride % sizeof(uint32_t) != 0,
              "Vulkan native dense field indexed-copy stride must be "
              "uint32-word aligned.");
}


void check_vulkan_reduce_strided_request(Ndarray *values,
                                         Ndarray *output,
                                         int value_type,
                                         std::size_t values_offset,
                                         std::size_t values_stride,
                                         std::size_t output_offset,
                                         std::size_t output_stride,
                                         int op) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native strided reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native strided reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native strided reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native strided reduce output must contain at least one "
              "item.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native strided reduce supports only sum/min/max "
              "operations.");
  auto check_range = [&](const char *role, Ndarray *arr,
                         std::size_t logical_items, std::size_t offset,
                         std::size_t stride) {
    TI_ERROR_IF(stride < value_size,
                "Vulkan native strided reduce {} stride is smaller than "
                "value size.",
                role);
    TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
                "Vulkan native strided reduce {} offset/stride must align "
                "to value size.",
                role);
    const std::size_t bytes = arr->get_nelement() * arr->get_element_size();
    TI_ERROR_IF(bytes < value_size,
                "Vulkan native strided reduce {} buffer is smaller than "
                "value size.",
                role);
    TI_ERROR_IF(offset > bytes - value_size,
                "Vulkan native strided reduce {} offset is out of bounds.",
                role);
    const std::size_t last =
        offset + (logical_items - 1) * stride + value_size;
    TI_ERROR_IF(last > bytes,
                "Vulkan native strided reduce {} range is out of bounds.",
                role);
    TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                    stride % sizeof(uint32_t) != 0,
                "Vulkan native strided reduce {} offset/stride must be "
                "uint32-word aligned.",
                role);
  };
  check_range("source", values, values->get_nelement(), values_offset,
              values_stride);
  check_range("destination", output, 1, output_offset, output_stride);
}

void check_vulkan_scan_member_request(Ndarray *data,
                                      int value_type,
                                      std::size_t offset,
                                      std::size_t stride) {
  TI_ERROR_IF(!data, "Vulkan native strided scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native strided scan expects a 1D ndarray.");
  const std::size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native strided scan received an unsupported value type.");
  TI_ERROR_IF(stride < value_size,
              "Vulkan native strided scan source stride is smaller than value "
              "size.");
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native strided scan source offset/stride must align to "
              "value size.");
  const std::size_t n = data->get_nelement();
  if (n == 0) {
    return;
  }
  const std::size_t src_bytes = n * data->get_element_size();
  TI_ERROR_IF(src_bytes < value_size,
              "Vulkan native strided scan source buffer is smaller than "
              "value size.");
  TI_ERROR_IF(offset > src_bytes - value_size,
              "Vulkan native strided scan source offset is out of bounds.");
  const std::size_t last = offset + (n - 1) * stride + value_size;
  TI_ERROR_IF(last > src_bytes,
              "Vulkan native strided scan source range is out of bounds.");
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 || stride % sizeof(uint32_t) != 0,
              "Vulkan native strided scan source offset/stride must be "
              "uint32-word aligned.");
}



void check_vulkan_strided_range(const char *op_name,
                                const char *arg_name,
                                Ndarray *arr,
                                size_t logical_items,
                                size_t value_size,
                                size_t offset,
                                size_t stride) {
  TI_ERROR_IF(stride < value_size,
              "{} {} stride is smaller than value size.", op_name, arg_name);
  TI_ERROR_IF(offset % value_size != 0 || stride % value_size != 0,
              "{} {} offset/stride must align to value size.", op_name,
              arg_name);
  TI_ERROR_IF(offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "{} {} offset/stride must be uint32-word aligned.", op_name,
              arg_name);
  if (logical_items == 0) {
    return;
  }
  const size_t buffer_bytes = logical_items * arr->get_element_size();
  TI_ERROR_IF(buffer_bytes < value_size,
              "{} {} buffer is smaller than value size.", op_name, arg_name);
  TI_ERROR_IF(offset > buffer_bytes - value_size,
              "{} {} offset is out of bounds.", op_name, arg_name);
  const size_t last = offset + (logical_items - 1) * stride + value_size;
  TI_ERROR_IF(last > buffer_bytes, "{} {} range is out of bounds.", op_name,
              arg_name);
}

void check_vulkan_scatter_add_strided_request(Ndarray *src,
                                              Ndarray *indices,
                                              Ndarray *dst,
                                              int value_type,
                                              size_t src_offset,
                                              size_t src_stride,
                                              size_t dst_offset,
                                              size_t dst_stride) {
  const char *op_name = "Vulkan native strided scatter-add";
  TI_ERROR_IF(!src || !indices || !dst, "{} received a null ndarray.",
              op_name);
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "{} expects 1D ndarrays.", op_name);
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "{} source and indices sizes differ.", op_name);
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "{} received an unsupported value type.",
              op_name);
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "{} expects i32 indices.", op_name);
  check_vulkan_strided_range(op_name, "source", src, src->get_nelement(),
                             value_size, src_offset, src_stride);
  check_vulkan_strided_range(op_name, "destination", dst, dst->get_nelement(),
                             value_size, dst_offset, dst_stride);
}

void check_vulkan_grouped_reduce_strided_keys_request(Ndarray *keys,
                                                      Ndarray *values,
                                                      Ndarray *output,
                                                      int value_type,
                                                      size_t keys_offset,
                                                      size_t keys_stride,
                                                      size_t values_offset,
                                                      size_t values_stride,
                                                      size_t output_offset,
                                                      size_t output_stride,
                                                      int op) {
  const char *op_name = "Vulkan native strided grouped reduce";
  TI_ERROR_IF(!keys || !values || !output, "{} received a null ndarray.",
              op_name);
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "{} expects 1D ndarrays.", op_name);
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "{} keys and values sizes differ.", op_name);
  TI_ERROR_IF(output->get_nelement() == 0,
              "{} output must contain at least one group.", op_name);
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "{} received an unsupported value type.",
              op_name);
  TI_ERROR_IF(op != 0, "{} currently supports only sum.", op_name);
  check_vulkan_strided_range(op_name, "keys", keys, keys->get_nelement(),
                             sizeof(int32_t), keys_offset, keys_stride);
  check_vulkan_strided_range(op_name, "values", values, values->get_nelement(),
                             value_size, values_offset, values_stride);
  check_vulkan_strided_range(op_name, "output", output, output->get_nelement(),
                             value_size, output_offset, output_stride);
}


std::size_t vulkan_reduce_storage_impl(Program *program,
                                       DeviceAllocation values_alloc,
                                       DeviceAllocation output_alloc,
                                       std::size_t n,
                                       std::size_t values_element_size,
                                       std::size_t output_element_size,
                                       int value_type,
                                       int op,
                                       std::size_t offset,
                                       std::size_t stride,
                                       std::size_t output_offset,
                                       std::size_t output_stride,
                                       bool member_source,
                                       bool member_destination) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native reduce is only available on Vulkan.");
  TI_ERROR_IF(values_alloc.device == nullptr || output_alloc.device == nullptr,
              "Vulkan native reduce received null storage.");
  TI_ERROR_IF(n == 0,
              "Vulkan native reduce expects at least one input item.");
  TI_ERROR_IF(value_type < 0 || value_type > 5,
              "Vulkan native reduce received an unsupported value type.");
  TI_ERROR_IF(!program->vulkan_reduce_value_type_available(value_type),
              "Vulkan native reduce dtype is not supported by this device.");
  const size_t element_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(op < 0 || op > 2,
              "Vulkan native reduce supports only sum/min/max operations.");
  if (!member_source) {
    TI_ERROR_IF(values_element_size != element_size,
                "Vulkan native reduce dtype does not match value type.");
    stride = element_size;
  }
  if (!member_destination) {
    TI_ERROR_IF(output_element_size != element_size,
                "Vulkan native reduce output dtype does not match value type.");
    output_stride = element_size;
  }
  auto check_strided = [&](const char *role, std::size_t role_offset,
                           std::size_t role_stride) {
    TI_ERROR_IF(role_stride < element_size,
                "Vulkan native reduce {} stride is smaller than value size.",
                role);
    TI_ERROR_IF(role_offset % element_size != 0 ||
                    role_stride % element_size != 0,
                "Vulkan native reduce {} offset/stride must align to value "
                "size.",
                role);
    TI_ERROR_IF(role_offset % sizeof(uint32_t) != 0 ||
                    role_stride % sizeof(uint32_t) != 0,
                "Vulkan native reduce {} offset/stride must be uint32-word "
                "aligned.",
                role);
  };
  if (member_source) {
    check_strided("source", offset, stride);
  }
  if (member_destination) {
    check_strided("destination", output_offset, output_stride);
  }

  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native reduce requires a compute device.");
  auto cache_lease = get_reduce_cache(program, device);
  auto &cache = *cache_lease;
  const bool use_i32_sum_atomic =
      value_type == 0 && op == 0 && !member_source && !member_destination &&
      get_environ_config("TI_VULKAN_REDUCE_I32_SUM_ATOMIC", 1) != 0;
  if (use_i32_sum_atomic) {
    Pipeline *pipeline = cache.i32_sum_atomic_pipeline(device);
    const size_t value_bytes = n * element_size;
    const size_t output_bytes = sizeof(int32_t);
    ShaderResourceSet *bindings =
        cache
            .bind_i32_sum_atomic_resource_set(program, values_alloc, offset,
                                              value_bytes, output_alloc,
                                              output_offset, output_bytes)
            .bindings;
    const int items_per_group_config = get_environ_config(
        "TI_VULKAN_REDUCE_I32_SUM_ATOMIC_ITEMS_PER_GROUP", 16384);
    const size_t items_per_group =
        static_cast<size_t>(std::max(256, items_per_group_config));
    constexpr size_t kMaxGroups = 65535;
    const uint32_t groups = static_cast<uint32_t>(
        std::min(kMaxGroups, (n + items_per_group - 1) / items_per_group));
    const bool profiler_scopes = program->profiler != nullptr;
    auto record_reduce_atomic =
        [values_alloc, output_alloc, offset, output_offset, value_bytes,
         output_bytes, pipeline, bindings, groups, profiler_scopes](
            Device * /*op_device*/, CommandList *cmdlist) {
          cmdlist->buffer_fill(output_alloc.get_ptr(output_offset),
                               output_bytes, 0);
          cmdlist->buffer_barrier(output_alloc);
          dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                            profiler_scopes ? "vulkan_reduce_i32_sum_atomic"
                                            : nullptr);
          cmdlist->buffer_barrier(output_alloc.get_ptr(output_offset),
                                  output_bytes);
        };
    VulkanCommandReplayKey command_key;
    command_key.push(1);
    command_key.push(values_alloc.alloc_id);
    command_key.push(vulkan_allocation_generation(values_alloc));
    command_key.push(offset);
    command_key.push(static_cast<uint64_t>(value_bytes));
    command_key.push(output_alloc.alloc_id);
    command_key.push(vulkan_allocation_generation(output_alloc));
    command_key.push(output_offset);
    command_key.push(static_cast<uint64_t>(output_bytes));
    command_key.push(groups);
    command_key.push_ptr(pipeline);
    command_key.push_ptr(bindings);
    if (!cache.reduce_i32_sum_atomic_command_replay.submit_or_record(
            program, device, command_key, profiler_scopes,
            record_reduce_atomic)) {
      program->enqueue_compute_op_lambda(record_reduce_atomic, {});
    }
    return cache.allocated_bytes();
  }
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_REDUCE_SINGLE_SHARED_MAX_N", 4096);
  const bool use_single_shared =
      single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);
  cache.ensure_op_pipelines(device, value_type, op, member_source,
                            use_single_shared);

  const size_t value_bytes =
      member_source ? strided_binding_bytes(n, element_size, offset, stride)
                    : n * element_size;
  const size_t values_binding_offset = member_source ? 0 : offset;
  const size_t output_bytes = element_size;

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (!use_single_shared) {
    num_chunks = (n + kReducePrivateChunkSize - 1) / kReducePrivateChunkSize;
    partial_bytes = num_chunks * element_size;
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      program->synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }
  if (member_source) {
    cache.ensure_params();
  }
  DeviceAllocation partial_alloc = cache.partial;
  DeviceAllocation params_alloc = cache.params;
  auto &pipeline_set = cache.pipeline_set(value_type);
  Pipeline *private_pipeline =
      member_source ? pipeline_set.private_strided_pipelines[op].get()
                    : pipeline_set.private_pipelines[op].get();
  Pipeline *final_pipeline = pipeline_set.final_pipelines[op].get();
  Pipeline *single_pipeline =
      member_source ? pipeline_set.single_strided_pipelines[op].get()
                    : pipeline_set.single_pipelines[op].get();
  constexpr size_t params_bytes = 3 * sizeof(uint32_t);
  ShaderResourceSet *single_bindings = nullptr;
  ShaderResourceSet *private_bindings = nullptr;
  ShaderResourceSet *final_bindings = nullptr;
  if (use_single_shared) {
    single_bindings =
        member_source
            ? cache
                  .bind_single_strided_resource_set(
                      program, values_alloc, values_binding_offset,
                      value_bytes, output_alloc, output_offset, output_bytes,
                      params_alloc, params_bytes)
                  .bindings
            : cache
                  .bind_single_resource_set(
                      program, values_alloc, values_binding_offset,
                      value_bytes, output_alloc, output_offset, output_bytes)
                  .bindings;
  } else {
    private_bindings =
        member_source
            ? cache
                  .bind_private_strided_resource_set(
                      program, values_alloc, values_binding_offset,
                      value_bytes, partial_alloc, partial_bytes, params_alloc,
                      params_bytes)
                  .bindings
            : cache
                  .bind_private_resource_set(program, values_alloc,
                                             values_binding_offset, value_bytes,
                                             partial_alloc, partial_bytes)
                  .bindings;
    final_bindings =
        cache
            .bind_final_resource_set(program, partial_alloc, partial_bytes,
                                     output_alloc, output_offset, output_bytes)
            .bindings;
  }
  const bool profiler_scopes = program->profiler != nullptr;
  std::array<uint32_t, 3> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(offset / sizeof(uint32_t)),
      static_cast<uint32_t>(stride / sizeof(uint32_t)),
  };
  auto record_reduce_tree =
      [values_alloc, output_alloc, partial_alloc, params_alloc,
       values_binding_offset, value_bytes, output_bytes, partial_bytes,
       private_pipeline, final_pipeline, single_pipeline, num_chunks,
       use_single_shared, member_source, output_offset, param_words,
       profiler_scopes, params_bytes, single_bindings, private_bindings,
       final_bindings](Device * /*op_device*/, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        auto bind_params = [&](ShaderResourceSet *bindings) {
          if (!member_source) {
            return;
          }
          for (uint32_t i = 0; i < param_words.size(); ++i) {
            cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                                 sizeof(uint32_t), param_words[i]);
          }
          cmdlist->buffer_barrier(params_alloc);
        };
        if (use_single_shared) {
          ShaderResourceSet *bindings = single_bindings;
          bind_params(bindings);
          dispatch_pipeline(cmdlist, single_pipeline, bindings, 1, 1, 1,
                            scope_name(member_source
                                           ? "vulkan_reduce_single_strided"
                                           : "vulkan_reduce_single"));
          cmdlist->buffer_barrier(output_alloc.get_ptr(output_offset),
                                  output_bytes);
          return;
        }
        {
          ShaderResourceSet *bindings = private_bindings;
          bind_params(bindings);
          dispatch_pipeline(cmdlist, private_pipeline, bindings,
                            static_cast<uint32_t>(num_chunks), 1, 1,
                            scope_name(member_source
                                           ? "vulkan_reduce_private_strided"
                                           : "vulkan_reduce_private"));
          cmdlist->buffer_barrier(partial_alloc);
        }
        {
          ShaderResourceSet *bindings = final_bindings;
          dispatch_pipeline(cmdlist, final_pipeline, bindings, 1, 1, 1,
                            scope_name("vulkan_reduce_final"));
          cmdlist->buffer_barrier(output_alloc.get_ptr(output_offset),
                                  output_bytes);
        }
      };
  VulkanCommandReplayKey command_key;
  command_key.push(2);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(static_cast<uint64_t>(op));
  command_key.push(use_single_shared ? 1 : 0);
  command_key.push(member_source ? 1 : 0);
  command_key.push(member_destination ? 1 : 0);
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(values_binding_offset);
  command_key.push(static_cast<uint64_t>(value_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(output_offset);
  command_key.push(static_cast<uint64_t>(output_bytes));
  command_key.push(partial_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(partial_alloc));
  command_key.push(static_cast<uint64_t>(partial_bytes));
  command_key.push(params_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(params_alloc));
  command_key.push(static_cast<uint64_t>(params_bytes));
  command_key.push(static_cast<uint64_t>(num_chunks));
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(private_pipeline);
  command_key.push_ptr(final_pipeline);
  command_key.push_ptr(single_pipeline);
  command_key.push_ptr(single_bindings);
  command_key.push_ptr(private_bindings);
  command_key.push_ptr(final_bindings);
  if (!cache.reduce_tree_command_replay.submit_or_record(
          program, device, command_key, profiler_scopes, record_reduce_tree)) {
    program->enqueue_compute_op_lambda(record_reduce_tree, {});
  }
  return cache.cached_bytes;
}

std::size_t vulkan_reduce_ndarray_impl(Program *program,
                                       Ndarray *values,
                                       Ndarray *output,
                                       int value_type,
                                       int op,
                                       std::size_t offset,
                                       std::size_t stride,
                                       std::size_t output_offset,
                                       std::size_t output_stride,
                                       bool member_source,
                                       bool member_destination) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native reduce output must contain at least one item.");
  if (member_source || member_destination) {
    check_vulkan_reduce_strided_request(values, output, value_type, offset,
                                        stride, output_offset, output_stride,
                                        op);
  } else {
    const size_t element_size = vulkan_transform_value_size(value_type);
    TI_ERROR_IF(values->get_element_size() != element_size ||
                    output->get_element_size() != element_size,
                "Vulkan native reduce dtype does not match value type.");
  }
  return vulkan_reduce_storage_impl(
      program, values->ndarray_alloc_, output->ndarray_alloc_,
      values->get_nelement(), values->get_element_size(),
      output->get_element_size(), value_type, op, offset, stride,
      output_offset, output_stride, member_source, member_destination);
}

std::size_t vulkan_check_count_storage_impl(Program *program,
                                            DeviceAllocation values_alloc,
                                            DeviceAllocation output_alloc,
                                            std::size_t n,
                                            std::size_t values_element_size,
                                            int value_type,
                                            int check_op,
                                            int lower,
                                            int upper,
                                            std::size_t offset,
                                            std::size_t stride) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native check_count is only available on Vulkan.");
  TI_ERROR_IF(values_alloc.device == nullptr || output_alloc.device == nullptr,
              "Vulkan native check_count received null storage.");
  TI_ERROR_IF(n == 0,
              "Vulkan native check_count expects at least one input item.");
  TI_ERROR_IF(!program->vulkan_check_count_value_type_available(value_type),
              "Vulkan native check_count received an unsupported value type.");
  TI_ERROR_IF(check_op < 0 || check_op > 5,
              "Vulkan native check_count received an unsupported check op.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(values_element_size < value_size,
              "Vulkan native check_count value storage is smaller than dtype.");
  TI_ERROR_IF(stride < value_size || offset % value_size != 0 ||
                  stride % value_size != 0,
              "Vulkan native check_count received invalid offset/stride.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native check_count currently supports at most "
              "UINT32_MAX input items.");
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native check_count requires a compute device.");
  auto cache_lease = get_check_count_cache(program, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.pipeline_for(device, value_type);
  const size_t values_bytes = strided_binding_bytes(n, value_size, offset, stride);
  const size_t output_bytes = sizeof(int32_t);
  ShaderResourceSet *bindings =
      cache
          .bind_resource_set(program, values_alloc, 0, values_bytes,
                             output_alloc, 0, output_bytes)
          .bindings;
  const int items_per_group_config =
      get_environ_config("TI_VULKAN_CHECK_COUNT_ITEMS_PER_GROUP", 16384);
  const size_t items_per_group =
      static_cast<size_t>(std::max(256, items_per_group_config));
  constexpr size_t kMaxGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(
      std::min(kMaxGroups, (n + items_per_group - 1) / items_per_group));
  const std::array<uint32_t, 8> param_words{
      static_cast<uint32_t>(check_op),
      static_cast<uint32_t>(lower),
      static_cast<uint32_t>(upper),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(offset / value_size),
      static_cast<uint32_t>(stride / value_size),
      0u,
      0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = program->profiler != nullptr;
  auto record_check_count =
      [output_alloc, output_bytes, pipeline, bindings, param_words, push_bytes,
       groups, value_type, profiler_scopes](Device * /*op_device*/,
                                            CommandList *cmdlist) {
        cmdlist->buffer_fill(output_alloc.get_ptr(0), output_bytes, 0);
        cmdlist->buffer_barrier(output_alloc);
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_check_count_f32"
                       : value_type == 2
                       ? "vulkan_check_count_u32"
                       : value_type == 3
                       ? "vulkan_check_count_u64"
                       : value_type == 4
                       ? "vulkan_check_count_i64"
                       : value_type == 5
                       ? "vulkan_check_count_f64"
                       : "vulkan_check_count_i32")
                : nullptr);
        cmdlist->buffer_barrier(output_alloc.get_ptr(0), output_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(90);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(static_cast<uint64_t>(check_op));
  command_key.push(static_cast<uint64_t>(static_cast<int64_t>(lower)));
  command_key.push(static_cast<uint64_t>(static_cast<int64_t>(upper)));
  command_key.push(static_cast<uint64_t>(n));
  command_key.push(static_cast<uint64_t>(offset));
  command_key.push(static_cast<uint64_t>(stride));
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, output_alloc, 0, output_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.command_replay.submit_or_record(program, device, command_key,
                                             profiler_scopes,
                                             record_check_count)) {
    program->enqueue_compute_op_lambda(record_check_count, {});
  }
  cache.cached_bytes = 0;
  return cache.cached_bytes;
}

std::size_t vulkan_check_count_ndarray_impl(Program *program,
                                            Ndarray *values,
                                            Ndarray *output,
                                            int value_type,
                                            int check_op,
                                            int lower,
                                            int upper,
                                            std::size_t offset,
                                            std::size_t stride,
                                            bool member_source) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native check_count received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native check_count expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native check_count expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native check_count output must contain at least one "
              "item.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native check_count received an unsupported value type.");
  if (!member_source) {
    TI_ERROR_IF(values->get_element_size() != value_size,
              "Vulkan native check_count dtype does not match value type.");
    offset = 0;
    stride = value_size;
  } else {
    check_vulkan_strided_range("Vulkan native check_count", "source", values,
                               values->get_nelement(), value_size, offset,
                               stride);
  }
  TI_ERROR_IF(output->get_element_size() != sizeof(int32_t),
              "Vulkan native check_count output must be i32.");
  const size_t n = values->get_nelement();
  return vulkan_check_count_storage_impl(
      program, values->ndarray_alloc_, output->ndarray_alloc_, n,
      values->get_element_size(), value_type, check_op, lower, upper, offset,
      stride);
}

std::size_t vulkan_metric_reduce_storage_impl(Program *program,
                                              DeviceAllocation values_alloc,
                                              DeviceAllocation other_alloc,
                                              DeviceAllocation output_alloc,
                                              std::size_t n,
                                              int value_type,
                                              int metric_op,
                                              std::size_t values_offset,
                                              std::size_t values_stride,
                                              std::size_t other_offset,
                                              std::size_t other_stride) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native metric_reduce is only available on Vulkan.");
  TI_ERROR_IF(values_alloc.device == nullptr || other_alloc.device == nullptr ||
                  output_alloc.device == nullptr,
              "Vulkan native metric_reduce received null storage.");
  TI_ERROR_IF(n == 0,
              "Vulkan native metric_reduce expects at least one input item.");
  TI_ERROR_IF(!program->vulkan_metric_reduce_value_type_available(value_type),
              "Vulkan native metric_reduce currently supports only f32.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "Vulkan native metric_reduce received an unsupported metric op.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native metric_reduce received an unsupported value "
              "type.");
  TI_ERROR_IF(values_stride < value_size || other_stride < value_size ||
                  values_offset % value_size != 0 ||
                  values_stride % value_size != 0 ||
                  other_offset % value_size != 0 ||
                  other_stride % value_size != 0,
              "Vulkan native metric_reduce received invalid offset/stride.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native metric_reduce currently supports at most "
              "UINT32_MAX input items.");
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native metric_reduce requires a compute device.");
  auto cache_lease = get_metric_reduce_cache(program, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.pipeline_for(device, value_type);
  const size_t values_bytes =
      strided_binding_bytes(n, value_size, values_offset, values_stride);
  const size_t other_bytes =
      strided_binding_bytes(n, value_size, other_offset, other_stride);
  const size_t output_bytes = value_size;
  ShaderResourceSet *bindings =
      cache
          .bind_resource_set(program, values_alloc, 0, values_bytes,
                             other_alloc, 0, other_bytes, output_alloc, 0,
                             output_bytes)
          .bindings;
  const int items_per_group_config =
      get_environ_config("TI_VULKAN_METRIC_REDUCE_ITEMS_PER_GROUP", 16384);
  const size_t items_per_group =
      static_cast<size_t>(std::max(256, items_per_group_config));
  constexpr size_t kMaxGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(
      std::min(kMaxGroups, (n + items_per_group - 1) / items_per_group));
  const std::array<uint32_t, 8> param_words{
      static_cast<uint32_t>(metric_op),
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(values_offset / value_size),
      static_cast<uint32_t>(values_stride / value_size),
      static_cast<uint32_t>(other_offset / value_size),
      static_cast<uint32_t>(other_stride / value_size),
      0u,
      0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = program->profiler != nullptr;
  auto record_metric_reduce =
      [output_alloc, output_bytes, pipeline, bindings, param_words, push_bytes,
       groups, metric_op, profiler_scopes](Device * /*op_device*/,
                                           CommandList *cmdlist) {
        cmdlist->buffer_fill(output_alloc.get_ptr(0), output_bytes, 0);
        cmdlist->buffer_barrier(output_alloc);
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes
                ? (metric_op == 1 ? "vulkan_max_abs_delta_f32"
                                  : "vulkan_max_abs_f32")
                : nullptr);
        cmdlist->buffer_barrier(output_alloc.get_ptr(0), output_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(91);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(static_cast<uint64_t>(metric_op));
  command_key.push(static_cast<uint64_t>(n));
  command_key.push(static_cast<uint64_t>(values_offset));
  command_key.push(static_cast<uint64_t>(values_stride));
  command_key.push(static_cast<uint64_t>(other_offset));
  command_key.push(static_cast<uint64_t>(other_stride));
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, other_alloc, 0, other_bytes);
  push_vulkan_command_key_range(command_key, output_alloc, 0, output_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.command_replay.submit_or_record(program, device, command_key,
                                             profiler_scopes,
                                             record_metric_reduce)) {
    program->enqueue_compute_op_lambda(record_metric_reduce, {});
  }
  cache.cached_bytes = 0;
  return cache.cached_bytes;
}

std::size_t vulkan_metric_reduce_ndarray_impl(Program *program,
                                              Ndarray *values,
                                              Ndarray *other,
                                              Ndarray *output,
                                              int value_type,
                                              int metric_op,
                                              std::size_t values_offset,
                                              std::size_t values_stride,
                                              std::size_t other_offset,
                                              std::size_t other_stride,
                                              bool member_source) {
  TI_ERROR_IF(!values || !output,
              "Vulkan native metric_reduce received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() == 0,
              "Vulkan native metric_reduce expects at least one input item.");
  TI_ERROR_IF(output->get_nelement() < 1,
              "Vulkan native metric_reduce output must contain at least one "
              "item.");
  TI_ERROR_IF(!program->vulkan_metric_reduce_value_type_available(value_type),
              "Vulkan native metric_reduce currently supports only f32.");
  TI_ERROR_IF(metric_op < 0 || metric_op > 1,
              "Vulkan native metric_reduce received an unsupported metric op.");
  TI_ERROR_IF(metric_op == 1 && !other,
              "Vulkan native max_abs_delta received a null rhs ndarray.");
  if (!other) {
    other = values;
    other_offset = values_offset;
    other_stride = values_stride;
  }
  TI_ERROR_IF(other->shape.size() != 1,
              "Vulkan native metric_reduce rhs must be a 1D ndarray.");
  TI_ERROR_IF(other->get_nelement() != values->get_nelement(),
              "Vulkan native metric_reduce inputs must have the same length.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native metric_reduce received an unsupported value "
              "type.");
  if (!member_source) {
    TI_ERROR_IF(values->get_element_size() != value_size ||
                    other->get_element_size() != value_size,
                "Vulkan native metric_reduce dtype does not match value "
                "type.");
    values_offset = 0;
    values_stride = value_size;
    other_offset = 0;
    other_stride = value_size;
  } else {
    check_vulkan_strided_range("Vulkan native metric_reduce", "source", values,
                               values->get_nelement(), value_size,
                               values_offset, values_stride);
    check_vulkan_strided_range("Vulkan native metric_reduce", "rhs", other,
                               other->get_nelement(), value_size,
                               other_offset, other_stride);
  }
  TI_ERROR_IF(output->get_element_size() != value_size,
              "Vulkan native metric_reduce output dtype does not match value "
              "type.");
  const size_t n = values->get_nelement();
  return vulkan_metric_reduce_storage_impl(
      program, values->ndarray_alloc_, other->ndarray_alloc_,
      output->ndarray_alloc_, n, value_type, metric_op, values_offset,
      values_stride, other_offset, other_stride);
}

std::size_t vulkan_transform_affine_storage_impl(
    Program *program,
    DeviceAllocation src_alloc,
    DeviceAllocation dst_alloc,
    std::size_t n,
    std::size_t src_element_size,
    std::size_t dst_element_size,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias,
    bool member_source,
    bool member_destination,
    bool trusted = false) {
  if (!trusted) {
    TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
                "Vulkan native transform is only available on Vulkan.");
    TI_ERROR_IF(src_alloc.device == nullptr || dst_alloc.device == nullptr,
                "Vulkan native transform received null storage.");
    TI_ERROR_IF(lane_count <= 0,
                "Vulkan native transform lane count must be positive.");
  }
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  const std::size_t payload_size =
      static_cast<std::size_t>(lane_count) * value_size;
  if (!member_source) {
    if (!trusted) {
      TI_ERROR_IF(src_element_size != value_size,
                  "Vulkan native transform dtype does not match value type.");
    }
    src_stride = payload_size;
  }
  if (!member_destination) {
    if (!trusted) {
      TI_ERROR_IF(dst_element_size != value_size,
                  "Vulkan native transform destination dtype does not match "
                  "value type.");
    }
    dst_stride = payload_size;
  }
  auto check_strided = [&](const char *role, std::size_t role_offset,
                           std::size_t role_stride) {
    TI_ERROR_IF(role_stride < payload_size,
                "Vulkan native transform {} stride is smaller than payload.",
                role);
    TI_ERROR_IF(role_offset % value_size != 0 ||
                    role_stride % value_size != 0,
                "Vulkan native transform {} offset/stride must align to value "
                "size.",
                role);
    TI_ERROR_IF(role_offset % sizeof(uint32_t) != 0 ||
                    role_stride % sizeof(uint32_t) != 0,
                "Vulkan native transform {} offset/stride must be "
                "uint32-word aligned.",
                role);
  };
  if (!trusted && member_source) {
    check_strided("source", src_offset, src_stride);
  }
  if (!trusted && member_destination) {
    check_strided("destination", dst_offset, dst_stride);
  }
  if (!trusted) {
    TI_ERROR_IF(!program->vulkan_transform_value_type_available(value_type),
                "Vulkan native transform value type is not supported by this "
                "device.");
    TI_ERROR_IF(
        n > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
        "Vulkan native transform currently supports at most UINT32_MAX items.");
  }

  const size_t scalar_count = n * static_cast<size_t>(lane_count);
  if (!trusted) {
    TI_ERROR_IF(scalar_count >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native transform scalar item count exceeds UINT32_MAX.");
  }
  if (n == 0) {
    return 0;
  }
  Device *device = program->get_compute_device();
  if (!trusted) {
    TI_ERROR_IF(!device, "Vulkan native transform requires a compute device.");
  }
  auto cache_lease = get_transform_cache(program, device);
  auto &cache = *cache_lease;

  const bool use_dense_i32_affine =
      (value_type == 0 || value_type == 2) && lane_count == 1 &&
      !member_source && !member_destination &&
      get_environ_config("TI_VULKAN_TRANSFORM_DENSE_I32_AFFINE", 1) != 0;
  if (use_dense_i32_affine) {
    uint32_t scale_word = 0;
    uint32_t bias_word = 0;
    if (value_type == 0) {
      scale_word = static_cast<uint32_t>(static_cast<int32_t>(scale));
      bias_word = static_cast<uint32_t>(static_cast<int32_t>(bias));
    } else {
      scale_word = static_cast<uint32_t>(scale);
      bias_word = static_cast<uint32_t>(bias);
    }
    ShaderResourceSet *bindings = cache.cached_dense_i32_affine_resource_set();
    Pipeline *pipeline = cache.dense_i32_pipeline(device);
    std::array<uint32_t, 3> push_words{scale_word, bias_word,
                                       static_cast<uint32_t>(n)};
    const size_t src_bytes = n * value_size;
    const size_t dst_bytes = n * value_size;
    const uint32_t push_bytes =
        static_cast<uint32_t>(push_words.size() * sizeof(uint32_t));
    const uint32_t groups =
        static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
    const bool profiler_scopes = program->profiler != nullptr;
    cache.dense_i32_affine_replay.rw_buffer(bindings, 0, src_alloc,
                                            src_offset, src_bytes);
    cache.dense_i32_affine_replay.rw_buffer(bindings, 1, dst_alloc,
                                            dst_offset, dst_bytes);
    auto record_transform_dense =
        [src_alloc, dst_alloc, bindings, pipeline, src_offset, dst_offset,
         src_bytes, dst_bytes, push_words, push_bytes, groups, profiler_scopes](
            Device * /*op_device*/, CommandList *cmdlist) {
          dispatch_pipeline_with_push_constants(
              cmdlist, pipeline, bindings, push_words.data(), push_bytes,
              groups, 1, 1,
              profiler_scopes ? "vulkan_transform_i32_affine_dense" : nullptr);
          cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), dst_bytes);
        };
    VulkanCommandReplayKey command_key;
    command_key.push(30);
    command_key.push(static_cast<uint64_t>(value_type));
    command_key.push(src_alloc.alloc_id);
    command_key.push(vulkan_allocation_generation(src_alloc));
    command_key.push(src_offset);
    command_key.push(static_cast<uint64_t>(src_bytes));
    command_key.push(dst_alloc.alloc_id);
    command_key.push(vulkan_allocation_generation(dst_alloc));
    command_key.push(dst_offset);
    command_key.push(static_cast<uint64_t>(dst_bytes));
    command_key.push(groups);
    for (uint32_t word : push_words) {
      command_key.push(word);
    }
    command_key.push_ptr(pipeline);
    command_key.push_ptr(bindings);
    if (!cache.dense_i32_affine_command_replay.submit_or_record(
            program, device, command_key, profiler_scopes,
            record_transform_dense)) {
      program->enqueue_compute_op_lambda(record_transform_dense, {});
    }
    return cache.cached_bytes;
  }

  std::array<uint32_t, 10> param_words{0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  if (value_type == 0) {
    param_words[0] = static_cast<uint32_t>(static_cast<int32_t>(scale));
    param_words[1] = static_cast<uint32_t>(static_cast<int32_t>(bias));
  } else if (value_type == 2) {
    param_words[0] = static_cast<uint32_t>(scale);
    param_words[1] = static_cast<uint32_t>(bias);
  } else if (value_type == 1) {
    float scale_f32 = static_cast<float>(scale);
    float bias_f32 = static_cast<float>(bias);
    std::memcpy(&param_words[0], &scale_f32, sizeof(param_words[0]));
    std::memcpy(&param_words[1], &bias_f32, sizeof(param_words[1]));
  } else {
    uint64_t scale_u64 = 0;
    uint64_t bias_u64 = 0;
    if (value_type == 4) {
      scale_u64 = static_cast<uint64_t>(static_cast<int64_t>(scale));
      bias_u64 = static_cast<uint64_t>(static_cast<int64_t>(bias));
    } else if (value_type == 5) {
      std::memcpy(&scale_u64, &scale, sizeof(scale_u64));
      std::memcpy(&bias_u64, &bias, sizeof(bias_u64));
    } else {
      scale_u64 = static_cast<uint64_t>(scale);
      bias_u64 = static_cast<uint64_t>(bias);
    }
    param_words[0] = static_cast<uint32_t>(scale_u64);
    param_words[1] = static_cast<uint32_t>(scale_u64 >> 32);
    param_words[2] = static_cast<uint32_t>(bias_u64);
    param_words[3] = static_cast<uint32_t>(bias_u64 >> 32);
  }
  param_words[4] = static_cast<uint32_t>(n);
  const uint32_t packed_stride_words =
      static_cast<uint32_t>(payload_size / sizeof(uint32_t));
  param_words[5] =
      member_source ? static_cast<uint32_t>(src_offset / sizeof(uint32_t)) : 0;
  param_words[6] = member_source
                       ? static_cast<uint32_t>(src_stride / sizeof(uint32_t))
                       : packed_stride_words;
  param_words[7] = member_destination
                       ? static_cast<uint32_t>(dst_offset / sizeof(uint32_t))
                       : 0;
  param_words[8] = member_destination
                       ? static_cast<uint32_t>(dst_stride / sizeof(uint32_t))
                       : packed_stride_words;
  param_words[9] = static_cast<uint32_t>(lane_count);

  ShaderResourceSet *bindings = cache.cached_affine_resource_set();
  const bool has_float64 =
      program->get_device_caps().get(DeviceCapability::spirv_has_float64) != 0;
  Pipeline *pipeline = cache.pipeline_for(device, value_type, has_float64);
  const size_t src_bytes =
      member_source ? strided_binding_bytes(n, payload_size, src_offset,
                                            src_stride)
                    : scalar_count * value_size;
  const size_t src_binding_offset = member_source ? 0 : src_offset;
  const size_t dst_bytes =
      member_destination ? strided_binding_bytes(n, payload_size, dst_offset,
                                                 dst_stride)
                         : scalar_count * value_size;
  const size_t dst_binding_offset = member_destination ? 0 : dst_offset;
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const uint32_t groups =
      static_cast<uint32_t>((scalar_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = program->profiler != nullptr;
  cache.affine_replay.rw_buffer(bindings, 0, src_alloc, src_binding_offset,
                                src_bytes);
  cache.affine_replay.rw_buffer(bindings, 1, dst_alloc, dst_binding_offset,
                                dst_bytes);

  auto record_transform_affine =
      [dst_alloc, bindings, pipeline, dst_binding_offset, dst_bytes,
       param_words, push_bytes, groups, profiler_scopes](
          Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
            1, 1, profiler_scopes ? "vulkan_transform_affine" : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_binding_offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(31);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(static_cast<uint64_t>(lane_count));
  command_key.push(member_source ? 1 : 0);
  command_key.push(member_destination ? 1 : 0);
  command_key.push(src_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(src_alloc));
  command_key.push(src_binding_offset);
  command_key.push(static_cast<uint64_t>(src_bytes));
  command_key.push(dst_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(dst_alloc));
  command_key.push(dst_binding_offset);
  command_key.push(static_cast<uint64_t>(dst_bytes));
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.affine_command_replay.submit_or_record(
          program, device, command_key, profiler_scopes,
          record_transform_affine)) {
    program->enqueue_compute_op_lambda(record_transform_affine, {});
  }
  return cache.cached_bytes;
}

std::size_t vulkan_transform_affine_ndarray_impl(Program *program,
                                                 Ndarray *src,
                                                 Ndarray *dst,
                                                 int value_type,
                                                 int lane_count,
                                                 std::size_t src_offset,
                                                 std::size_t src_stride,
                                                 std::size_t dst_offset,
                                                  std::size_t dst_stride,
                                                  double scale,
                                                  double bias,
                                                  bool member_source,
                                                  bool member_destination,
                                                  bool trusted = false) {
  if (!trusted) {
    TI_ERROR_IF(!src || !dst, "Vulkan native transform received null ndarray.");
    TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
                "Vulkan native transform source and destination sizes differ.");
    if (member_source || member_destination) {
      if (lane_count == 1) {
        check_vulkan_transform_strided_request(src, dst, value_type, src_offset,
                                               src_stride, dst_offset,
                                               dst_stride);
      } else {
        check_vulkan_transform_packed_strided_request(
            src, dst, value_type, lane_count, src_offset, src_stride, dst_offset,
            dst_stride);
      }
    } else {
      const std::size_t value_size = vulkan_transform_value_size(value_type);
      TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
                  "Vulkan native transform source and destination dtypes "
                  "differ.");
      TI_ERROR_IF(src->get_element_size() != value_size,
                  "Vulkan native transform dtype does not match value type.");
    }
  }
  return vulkan_transform_affine_storage_impl(
      program, src->ndarray_alloc_, dst->ndarray_alloc_, src->get_nelement(),
      src->get_element_size(), dst->get_element_size(), value_type, lane_count,
      src_offset, src_stride, dst_offset, dst_stride, scale, bias,
      member_source, member_destination, trusted);
}

std::size_t vulkan_transform_indexed_affine_ndarray_impl(Program *program,
                                                          Ndarray *src,
                                                          Ndarray *indices,
                                                          Ndarray *dst,
                                                          int value_type,
                                                          double scale,
                                                          double bias) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native indexed transform is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native indexed transform received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native indexed transform currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native indexed transform source and destination dtypes "
              "differ.");
  const std::size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size != sizeof(uint32_t),
              "Vulkan native indexed transform currently supports 32-bit "
              "i32/u32/f32 values.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  dst->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native indexed transform expects 32-bit values and i32 "
              "indices.");
  TI_ERROR_IF(value_type != 0 && value_type != 1 && value_type != 2,
              "Vulkan native indexed transform only supports i32/f32/u32.");
  TI_ERROR_IF(!program->vulkan_transform_value_type_available(value_type),
              "Vulkan native indexed transform value type is not supported by "
              "this device.");
  const std::size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
                  src->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
                  dst->get_nelement() >
                      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native indexed transform currently supports at most "
              "UINT32_MAX items.");

  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  TI_ERROR_IF(src_alloc.device == nullptr || indices_alloc.device == nullptr ||
                  dst_alloc.device == nullptr,
              "Vulkan native indexed transform received null storage.");
  const auto same_allocation = [](const DeviceAllocation &a,
                                  const DeviceAllocation &b) {
    return a.device == b.device && a.alloc_id == b.alloc_id;
  };
  TI_ERROR_IF(same_allocation(src_alloc, dst_alloc) ||
                  same_allocation(indices_alloc, dst_alloc),
              "Vulkan native indexed transform requires a non-aliased "
              "destination.");

  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native indexed transform requires a compute device.");
  auto cache_lease = get_transform_cache(program, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_indexed_affine_resource_set();
  Pipeline *pipeline = cache.indexed_pipeline_for(device, value_type);

  std::array<uint32_t, 5> push_words{0, 0, static_cast<uint32_t>(n),
                                     static_cast<uint32_t>(src->get_nelement()),
                                     static_cast<uint32_t>(dst->get_nelement())};
  if (value_type == 1) {
    float scale_f32 = static_cast<float>(scale);
    float bias_f32 = static_cast<float>(bias);
    std::memcpy(&push_words[0], &scale_f32, sizeof(push_words[0]));
    std::memcpy(&push_words[1], &bias_f32, sizeof(push_words[1]));
  } else if (value_type == 0) {
    push_words[0] = static_cast<uint32_t>(static_cast<int32_t>(scale));
    push_words[1] = static_cast<uint32_t>(static_cast<int32_t>(bias));
  } else {
    push_words[0] = static_cast<uint32_t>(scale);
    push_words[1] = static_cast<uint32_t>(bias);
  }

  const size_t src_bytes = src->get_nelement() * value_size;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * value_size;
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t push_bytes =
      static_cast<uint32_t>(push_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = program->profiler != nullptr;

  cache.indexed_affine_replay.rw_buffer(bindings, 0, src_alloc, 0, src_bytes);
  cache.indexed_affine_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                        indices_bytes);
  cache.indexed_affine_replay.rw_buffer(bindings, 2, dst_alloc, 0, dst_bytes);
  auto record_transform_indexed =
      [dst_alloc, bindings, pipeline, dst_bytes, push_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, push_words.data(), push_bytes, groups,
            1, 1,
            profiler_scopes ? "vulkan_transform_indexed_affine" : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(33);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  for (uint32_t word : push_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.indexed_affine_command_replay.submit_or_record(
          program, device, command_key, profiler_scopes,
          record_transform_indexed)) {
    program->enqueue_compute_op_lambda(record_transform_indexed, {});
  }
  return cache.cached_bytes;
}

std::size_t vulkan_add_merge_storage_impl(Program *program,
                                          DeviceAllocation src_alloc,
                                          DeviceAllocation dst_alloc,
                                          size_t n,
                                          size_t src_element_size,
                                          size_t dst_element_size,
                                          int value_type,
                                          size_t src_offset,
                                          size_t src_stride,
                                          size_t dst_offset,
                                          size_t dst_stride) {
  TI_ERROR_IF(program->compile_config().arch != Arch::vulkan,
              "Vulkan native add-merge is only available on Vulkan.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native add-merge received an unsupported value type.");
  TI_ERROR_IF(src_element_size < value_size || dst_element_size < value_size,
              "Vulkan native add-merge element size is smaller than value "
              "size.");
  TI_ERROR_IF(src_offset % value_size != 0 || src_stride % value_size != 0 ||
                  dst_offset % value_size != 0 ||
                  dst_stride % value_size != 0,
              "Vulkan native add-merge offsets and strides must align to "
              "value size.");
  if (n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native add-merge currently supports at most UINT32_MAX "
              "items.");
  TI_ERROR_IF((src_stride != 0 && src_stride < value_size) ||
                  dst_stride < value_size,
              "Vulkan native add-merge stride is smaller than value size.");
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native add-merge requires a compute device.");
  auto cache_lease = get_add_merge_cache(program, device);
  auto &cache = *cache_lease;
  const uint32_t src_offset_items =
      static_cast<uint32_t>(src_offset / value_size);
  const uint32_t src_stride_items =
      static_cast<uint32_t>(src_stride / value_size);
  const uint32_t dst_offset_items =
      static_cast<uint32_t>(dst_offset / value_size);
  const uint32_t dst_stride_items =
      static_cast<uint32_t>(dst_stride / value_size);
  const std::array<uint32_t, 6> param_words{
      static_cast<uint32_t>(n), src_offset_items, src_stride_items,
      dst_offset_items, dst_stride_items, 0};
  const bool has_float64 =
      program->get_device_caps().get(DeviceCapability::spirv_has_float64) != 0;
  Pipeline *pipeline = cache.pipeline_for(device, value_type, has_float64);
  ShaderResourceSet *bindings = cache.cached_resource_set();
  const size_t src_bytes =
      strided_binding_bytes(n, value_size, src_offset, src_stride);
  const size_t dst_bytes =
      strided_binding_bytes(n, value_size, dst_offset, dst_stride);
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = program->profiler != nullptr;
  cache.binding_replay.rw_buffer(bindings, 0, src_alloc, 0, src_bytes);
  cache.binding_replay.rw_buffer(bindings, 1, dst_alloc, 0, dst_bytes);
  auto record_add_merge =
      [dst_alloc, bindings, pipeline, dst_bytes, param_words, push_bytes,
       groups, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1, profiler_scopes ? "vulkan_add_merge" : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(32);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(src_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(src_alloc));
  command_key.push(static_cast<uint64_t>(src_bytes));
  command_key.push(dst_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(dst_alloc));
  command_key.push(static_cast<uint64_t>(dst_bytes));
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.command_replay.submit_or_record(program, device, command_key,
                                             profiler_scopes,
                                             record_add_merge)) {
    program->enqueue_compute_op_lambda(record_add_merge, {});
  }
  return 0;
}

std::size_t vulkan_add_merge_ndarray_impl(Program *program,
                                          Ndarray *src,
                                          Ndarray *dst,
                                          int value_type,
                                          std::size_t src_offset,
                                          std::size_t src_stride,
                                          std::size_t dst_offset,
                                          std::size_t dst_stride) {
  TI_ERROR_IF(!src || !dst, "Vulkan native add-merge received null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1,
              "Vulkan native add-merge expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native add-merge source and destination sizes differ.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(src_stride < value_size || dst_stride < value_size,
              "Vulkan native add-merge received an invalid stride.");
  return vulkan_add_merge_storage_impl(
      program, src->ndarray_alloc_, dst->ndarray_alloc_, src->get_nelement(),
      src->get_element_size(), dst->get_element_size(), value_type, src_offset,
      src_stride, dst_offset, dst_stride);
}

}  // namespace

namespace {

void check_vulkan_sparse_f32_vector(const char *role,
                                    Ndarray *array,
                                    std::size_t n) {
  TI_ERROR_IF(!array, "Vulkan sparse algebra received a null {} vector.",
              role);
  TI_ERROR_IF(array->shape.size() != 1 ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  !array->get_element_shape().empty() ||
                  array->get_element_size() != sizeof(float32) ||
                  array->get_nelement() != n,
              "Vulkan sparse algebra {} must contain exactly {} scalar f32 "
              "entries.",
              role, n);
}

void check_vulkan_sparse_f32_output(Ndarray *output) {
  TI_ERROR_IF(!output, "Vulkan sparse algebra received a null output.");
  TI_ERROR_IF(output->shape.size() != 1 ||
                  output->get_element_data_type() != PrimitiveType::f32 ||
                  !output->get_element_shape().empty() ||
                  output->get_element_size() != sizeof(float32) ||
                  output->get_nelement() != 1,
              "Vulkan sparse algebra reduction output must contain exactly "
              "one scalar f32 entry.");
}

void check_vulkan_sparse_i32_status(Ndarray *status) {
  TI_ERROR_IF(!status, "Vulkan sparse algebra received a null status.");
  TI_ERROR_IF(status->shape.size() != 1 ||
                  status->get_element_data_type() != PrimitiveType::i32 ||
                  !status->get_element_shape().empty() ||
                  status->get_element_size() != sizeof(int32_t) ||
                  status->get_nelement() != 1,
              "Vulkan sparse algebra status must contain exactly one "
              "scalar i32 entry.");
}

void check_vulkan_sparse_i32_vector(const char *role,
                                    Ndarray *array,
                                    std::size_t n) {
  TI_ERROR_IF(!array, "Vulkan sparse algebra received a null {} vector.",
              role);
  TI_ERROR_IF(array->shape.size() != 1 ||
                  array->get_element_data_type() != PrimitiveType::i32 ||
                  !array->get_element_shape().empty() ||
                  array->get_element_size() != sizeof(int32_t) ||
                  array->get_nelement() != n,
              "Vulkan sparse algebra {} must contain exactly {} scalar i32 "
              "entries.",
              role, n);
}

uint32_t vulkan_sparse_f32_word(float value) {
  uint32_t word = 0;
  static_assert(sizeof(word) == sizeof(value));
  std::memcpy(&word, &value, sizeof(word));
  return word;
}

}  // namespace

std::size_t Program::vulkan_sparse_axpy(Ndarray *x,
                                        Ndarray *y,
                                        std::size_t n,
                                        float alpha) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse axpy is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse axpy length must be in [1, UINT32_MAX].");
  check_vulkan_sparse_f32_vector("x", x, n);
  check_vulkan_sparse_f32_vector("y", y, n);
  TI_ERROR_IF(x->get_device_allocation() == y->get_device_allocation(),
              "Vulkan sparse axpy does not support aliased x/y buffers.");

  const Ndarray *resources[] = {x, y};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const auto x_alloc = x->get_device_allocation();
  const auto y_alloc = y->get_device_allocation();
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse axpy requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_axpy_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_axpy(this, x_alloc, vector_bytes, y_alloc,
                             vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), vulkan_sparse_f32_word(alpha), 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_axpy =
      [y_alloc, vector_bytes, pipeline, bindings, param_words, push_bytes,
       groups, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_axpy_f32" : nullptr);
        cmdlist->buffer_barrier(y_alloc.get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(141);
  push_vulkan_command_key_range(command_key, x_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(command_key, y_alloc, 0, vector_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_axpy_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_axpy)) {
    enqueue_compute_op_lambda(record_axpy, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_diagonal_refresh(
    Ndarray *values,
    Ndarray *diagonal_offsets,
    Ndarray *staging_inverse,
    Ndarray *status,
    std::size_t rows,
    std::size_t nnz) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse diagonal refresh is only available on "
              "Vulkan.");
  TI_ERROR_IF(rows == 0 || nnz == 0 ||
                  rows > std::numeric_limits<uint32_t>::max() ||
                  nnz > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse diagonal refresh geometry must fit in "
              "non-zero uint32 ranges.");
  check_vulkan_sparse_f32_vector("values", values, nnz);
  check_vulkan_sparse_i32_vector("diagonal offsets", diagonal_offsets,
                                 rows);
  check_vulkan_sparse_f32_vector("staging inverse", staging_inverse,
                                 rows);
  check_vulkan_sparse_i32_status(status);
  const auto values_alloc = values->get_device_allocation();
  const auto offsets_alloc = diagonal_offsets->get_device_allocation();
  const auto staging_alloc = staging_inverse->get_device_allocation();
  const auto status_alloc = status->get_device_allocation();
  TI_ERROR_IF(values_alloc == offsets_alloc || values_alloc == staging_alloc ||
                  values_alloc == status_alloc ||
                  offsets_alloc == staging_alloc ||
                  offsets_alloc == status_alloc ||
                  staging_alloc == status_alloc,
              "Vulkan sparse diagonal refresh resources must not alias.");

  const Ndarray *resources[] = {values, diagonal_offsets, staging_inverse,
                                status};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t values_bytes = nnz * sizeof(float32);
  const size_t offsets_bytes = rows * sizeof(int32_t);
  const size_t inverse_bytes = rows * sizeof(float32);
  const size_t status_bytes = sizeof(int32_t);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse diagonal refresh requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_diagonal_refresh_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_diagonal_refresh(
              this, values_alloc, values_bytes, offsets_alloc,
              offsets_bytes, staging_alloc, inverse_bytes, status_alloc,
              status_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(rows), static_cast<uint32_t>(nnz), 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (rows + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_refresh =
      [staging_alloc, status_alloc, inverse_bytes, status_bytes, pipeline,
       bindings, param_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_diagonal_refresh_f32"
                            : nullptr);
        cmdlist->buffer_barrier(staging_alloc.get_ptr(0), inverse_bytes);
        cmdlist->buffer_barrier(status_alloc.get_ptr(0), status_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(148);
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, offsets_alloc, 0,
                                offsets_bytes);
  push_vulkan_command_key_range(command_key, staging_alloc, 0,
                                inverse_bytes);
  push_vulkan_command_key_range(command_key, status_alloc, 0,
                                status_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_diagonal_refresh_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_refresh)) {
    enqueue_compute_op_lambda(record_refresh, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_diagonal_apply(
    Ndarray *inverse_diagonal,
    Ndarray *input,
    Ndarray *output,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse diagonal apply is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse diagonal apply length must be in [1, "
              "UINT32_MAX].");
  check_vulkan_sparse_f32_vector("inverse diagonal", inverse_diagonal, n);
  check_vulkan_sparse_f32_vector("input", input, n);
  check_vulkan_sparse_f32_vector("output", output, n);
  const auto inverse_alloc = inverse_diagonal->get_device_allocation();
  const auto input_alloc = input->get_device_allocation();
  const auto output_alloc = output->get_device_allocation();
  TI_ERROR_IF(inverse_alloc == input_alloc || inverse_alloc == output_alloc,
              "Vulkan sparse diagonal apply inverse storage must not alias "
              "an input or output vector.");

  const Ndarray *resources[] = {inverse_diagonal, input, output};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse diagonal apply requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_diagonal_apply_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_diagonal_apply(
              this, inverse_alloc, vector_bytes, input_alloc, vector_bytes,
              output_alloc, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_apply =
      [output_alloc, vector_bytes, pipeline, bindings, param_words,
       push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_diagonal_apply_f32" : nullptr);
        cmdlist->buffer_barrier(output_alloc.get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(142);
  push_vulkan_command_key_range(command_key, inverse_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(command_key, input_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(command_key, output_alloc, 0, vector_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_diagonal_apply_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_apply)) {
    enqueue_compute_op_lambda(record_apply, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_block_cholesky_refresh(
    Ndarray *values,
    Ndarray *diagonal_block_offsets,
    Ndarray *staging_factors,
    Ndarray *status,
    std::size_t block_rows,
    std::size_t block_nnz,
    std::size_t block_size) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse block Cholesky refresh is only available on "
              "Vulkan.");
  TI_ERROR_IF(block_rows == 0 || block_nnz == 0 ||
                  block_rows > std::numeric_limits<uint32_t>::max() ||
                  block_nnz > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse block Cholesky refresh geometry must fit in "
              "non-zero uint32 ranges.");
  TI_ERROR_IF(block_size != 2 && block_size != 3 && block_size != 6 &&
                  block_size != 12,
              "Vulkan sparse block Cholesky refresh supports block sizes "
              "2, 3, 6, and 12, got {}.",
              block_size);
  TI_ERROR_IF(block_nnz >
                      std::numeric_limits<std::size_t>::max() /
                          block_size / block_size ||
                  block_rows >
                      std::numeric_limits<std::size_t>::max() /
                          block_size / block_size,
              "Vulkan sparse block Cholesky refresh dimensions overflow "
              "size_t.");
  const std::size_t value_count = block_nnz * block_size * block_size;
  const std::size_t factor_count = block_rows * block_size * block_size;
  check_vulkan_sparse_f32_vector("values", values, value_count);
  check_vulkan_sparse_i32_vector("diagonal block offsets",
                                 diagonal_block_offsets, block_rows);
  check_vulkan_sparse_f32_vector("staging factors", staging_factors,
                                 factor_count);
  check_vulkan_sparse_i32_status(status);
  const auto values_alloc = values->get_device_allocation();
  const auto offsets_alloc =
      diagonal_block_offsets->get_device_allocation();
  const auto staging_alloc = staging_factors->get_device_allocation();
  const auto status_alloc = status->get_device_allocation();
  TI_ERROR_IF(values_alloc == offsets_alloc || values_alloc == staging_alloc ||
                  values_alloc == status_alloc ||
                  offsets_alloc == staging_alloc ||
                  offsets_alloc == status_alloc ||
                  staging_alloc == status_alloc,
              "Vulkan sparse block Cholesky refresh resources must not "
              "alias.");

  const Ndarray *resources[] = {values, diagonal_block_offsets,
                                staging_factors, status};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t values_bytes = value_count * sizeof(float32);
  const size_t offsets_bytes = block_rows * sizeof(int32_t);
  const size_t factor_bytes = factor_count * sizeof(float32);
  const size_t status_bytes = sizeof(int32_t);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse block Cholesky refresh requires a compute "
              "device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_block_cholesky_refresh_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_block_cholesky_refresh(
              this, values_alloc, values_bytes, offsets_alloc,
              offsets_bytes, staging_alloc, factor_bytes, status_alloc,
              status_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(block_rows),
      static_cast<uint32_t>(block_nnz),
      static_cast<uint32_t>(block_size), 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups,
      (block_rows + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_refresh =
      [staging_alloc, status_alloc, factor_bytes, status_bytes, pipeline,
       bindings, param_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes
                ? "vulkan_sparse_block_cholesky_refresh_f32"
                : nullptr);
        cmdlist->buffer_barrier(staging_alloc.get_ptr(0), factor_bytes);
        cmdlist->buffer_barrier(status_alloc.get_ptr(0), status_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(149);
  push_vulkan_command_key_range(command_key, values_alloc, 0, values_bytes);
  push_vulkan_command_key_range(command_key, offsets_alloc, 0,
                                offsets_bytes);
  push_vulkan_command_key_range(command_key, staging_alloc, 0,
                                factor_bytes);
  push_vulkan_command_key_range(command_key, status_alloc, 0,
                                status_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_block_cholesky_refresh_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_refresh)) {
    enqueue_compute_op_lambda(record_refresh, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_block_diagonal_apply(
    Ndarray *factor_blocks,
    Ndarray *input,
    Ndarray *output,
    std::size_t block_rows,
    std::size_t block_size) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(
      !vulkan_sparse_algebra_available(),
      "Vulkan sparse block diagonal apply is only available on Vulkan.");
  TI_ERROR_IF(block_rows == 0 ||
                  block_rows > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse block diagonal apply block_rows must be in "
              "[1, UINT32_MAX].");
  TI_ERROR_IF(block_size != 2 && block_size != 3 && block_size != 6 &&
                  block_size != 12,
              "Vulkan sparse block diagonal apply supports block sizes 2, "
              "3, 6, and 12, got {}.",
              block_size);
  TI_ERROR_IF(
      block_rows > std::numeric_limits<std::size_t>::max() / block_size ||
          block_rows * block_size >
              std::numeric_limits<std::size_t>::max() / block_size,
      "Vulkan sparse block diagonal apply dimensions overflow size_t.");
  const std::size_t rows = block_rows * block_size;
  const std::size_t factor_count = rows * block_size;
  TI_ERROR_IF(rows > std::numeric_limits<uint32_t>::max() ||
                  factor_count > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse block diagonal apply scalar dimensions exceed "
              "UINT32_MAX.");
  check_vulkan_sparse_f32_vector("Cholesky factors", factor_blocks,
                                 factor_count);
  check_vulkan_sparse_f32_vector("input", input, rows);
  check_vulkan_sparse_f32_vector("output", output, rows);
  const auto factor_alloc = factor_blocks->get_device_allocation();
  const auto input_alloc = input->get_device_allocation();
  const auto output_alloc = output->get_device_allocation();
  TI_ERROR_IF(factor_alloc == input_alloc || factor_alloc == output_alloc,
              "Vulkan sparse block Cholesky storage must not alias "
              "an input or output vector.");

  const Ndarray *resources[] = {factor_blocks, input, output};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t factor_bytes = factor_count * sizeof(float32);
  const size_t vector_bytes = rows * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse block diagonal apply requires a compute "
              "device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_block_cholesky_apply_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_block_cholesky_apply(
              this, factor_alloc, factor_bytes, input_alloc, vector_bytes,
              output_alloc, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(block_rows),
      static_cast<uint32_t>(block_size), 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups,
      (block_rows + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_apply =
      [output_alloc, vector_bytes, pipeline, bindings, param_words,
       push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes
                ? "vulkan_sparse_block_cholesky_apply_f32"
                : nullptr);
        cmdlist->buffer_barrier(output_alloc.get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(143);
  push_vulkan_command_key_range(command_key, factor_alloc, 0, factor_bytes);
  push_vulkan_command_key_range(command_key, input_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(command_key, output_alloc, 0, vector_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_block_cholesky_apply_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_apply)) {
    enqueue_compute_op_lambda(record_apply, {});
  }
  return cache.allocated_bytes();
}

namespace {

std::size_t vulkan_sparse_dot_impl(Program *program,
                                   Ndarray *x,
                                   Ndarray *y,
                                   Ndarray *output,
                                   std::size_t n,
                                   bool take_sqrt) {
  TI_ERROR_IF(!program->vulkan_sparse_algebra_available(),
              "Vulkan sparse dot/norm is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse dot/norm length must be in [1, UINT32_MAX].");
  check_vulkan_sparse_f32_vector("x", x, n);
  check_vulkan_sparse_f32_vector("y", y, n);
  check_vulkan_sparse_f32_output(output);
  const auto x_alloc = x->get_device_allocation();
  const auto y_alloc = y->get_device_allocation();
  const auto output_alloc = output->get_device_allocation();
  TI_ERROR_IF(output_alloc == x_alloc || output_alloc == y_alloc,
              "Vulkan sparse dot/norm output must not alias an input "
              "buffer.");

  const Ndarray *resources[] = {x, y, output};
  program->retain_ndarrays_for_external_submission(resources,
                                                    std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  constexpr size_t kTargetItemsPerGroup = 1024;
  constexpr size_t kMaxDispatchGroups = 65535;
  const size_t group_count = std::min(
      kMaxDispatchGroups,
      (n + kTargetItemsPerGroup - 1) / kTargetItemsPerGroup);
  const uint32_t groups = static_cast<uint32_t>(group_count);
  const size_t partial_bytes = group_count * sizeof(float32);
  Device *device = program->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse dot/norm requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(program, device);
  auto &cache = *cache_lease;
  if (cache.has_workspace_allocs() &&
      cache.needs_partial_realloc(partial_bytes)) {
    program->synchronize();
  }
  cache.ensure_partial(partial_bytes);
  const auto partial_alloc = cache.partial;
  Pipeline *partial_pipeline = cache.sparse_dot_partial_pipeline(device);
  ShaderResourceSet *partial_bindings =
      cache
          .bind_sparse_dot_partial(program, x_alloc, vector_bytes, y_alloc,
                                   vector_bytes, partial_alloc, partial_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = program->profiler != nullptr;
  auto record_dot_partial =
      [partial_alloc, partial_bytes, partial_pipeline, partial_bindings,
       param_words, push_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, partial_pipeline, partial_bindings, param_words.data(),
            push_bytes, groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_dot_partial_f32" : nullptr);
        cmdlist->buffer_barrier(partial_alloc.get_ptr(0), partial_bytes);
      };
  VulkanCommandReplayKey partial_key;
  partial_key.push(142);
  push_vulkan_command_key_range(partial_key, x_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(partial_key, y_alloc, 0, vector_bytes);
  push_vulkan_command_key_range(partial_key, partial_alloc, 0,
                                partial_bytes);
  for (uint32_t word : param_words) {
    partial_key.push(word);
  }
  partial_key.push(groups);
  partial_key.push_ptr(partial_pipeline);
  partial_key.push_ptr(partial_bindings);
  if (!cache.sparse_dot_partial_command_replay.submit_or_record(
          program, device, partial_key, profiler_scopes,
          record_dot_partial)) {
    program->enqueue_compute_op_lambda(record_dot_partial, {});
  }

  const size_t reduce_workspace = vulkan_reduce_storage_impl(
      program, partial_alloc, output_alloc, group_count, sizeof(float32),
      sizeof(float32), 1, 0, 0, sizeof(float32), 0, sizeof(float32), false,
      false);
  if (take_sqrt) {
    constexpr size_t kScalarBytes = sizeof(float32);
    Pipeline *sqrt_pipeline = cache.sparse_sqrt_scalar_pipeline(device);
    ShaderResourceSet *sqrt_bindings =
        cache.bind_sparse_sqrt_scalar(program, output_alloc, kScalarBytes)
            .bindings;
    auto record_sqrt = [output_alloc, sqrt_pipeline, sqrt_bindings,
                        profiler_scopes](Device * /*op_device*/,
                                         CommandList *cmdlist) {
      dispatch_pipeline(cmdlist, sqrt_pipeline, sqrt_bindings, 1, 1, 1,
                        profiler_scopes
                            ? "vulkan_sparse_sqrt_scalar_f32"
                            : nullptr);
      cmdlist->buffer_barrier(output_alloc.get_ptr(0), sizeof(float32));
    };
    VulkanCommandReplayKey sqrt_key;
    sqrt_key.push(143);
    push_vulkan_command_key_range(sqrt_key, output_alloc, 0, kScalarBytes);
    sqrt_key.push_ptr(sqrt_pipeline);
    sqrt_key.push_ptr(sqrt_bindings);
    if (!cache.sparse_sqrt_scalar_command_replay.submit_or_record(
            program, device, sqrt_key, profiler_scopes, record_sqrt)) {
      program->enqueue_compute_op_lambda(record_sqrt, {});
    }
  }
  return cache.allocated_bytes() + reduce_workspace;
}

}  // namespace

std::size_t Program::vulkan_sparse_dot(Ndarray *x,
                                       Ndarray *y,
                                       Ndarray *output,
                                       std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  return vulkan_sparse_dot_impl(this, x, y, output, n, false);
}

std::size_t Program::vulkan_sparse_norm(Ndarray *x,
                                        Ndarray *output,
                                        std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  return vulkan_sparse_dot_impl(this, x, x, output, n, true);
}

std::size_t Program::vulkan_sparse_scalar_divide(Ndarray *numerator,
                                                 Ndarray *denominator,
                                                 Ndarray *quotient,
                                                 Ndarray *status) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse scalar divide is only available on Vulkan.");
  check_vulkan_sparse_f32_output(numerator);
  check_vulkan_sparse_f32_output(denominator);
  check_vulkan_sparse_f32_output(quotient);
  check_vulkan_sparse_i32_status(status);
  const std::array<DeviceAllocation, 4> allocs{
      numerator->get_device_allocation(), denominator->get_device_allocation(),
      quotient->get_device_allocation(), status->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse scalar divide buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {numerator, denominator, quotient, status};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  constexpr size_t kScalarBytes = sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse scalar divide requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_scalar_divide_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_scalar_divide(this, allocs[0], allocs[1], allocs[2],
                                     allocs[3], kScalarBytes)
          .bindings;
  const bool profiler_scopes = profiler != nullptr;
  auto record_divide =
      [allocs, pipeline, bindings,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, 1, 1, 1,
                          profiler_scopes
                              ? "vulkan_sparse_scalar_divide_f32"
                              : nullptr);
        cmdlist->buffer_barrier(allocs[2].get_ptr(0), sizeof(float32));
        cmdlist->buffer_barrier(allocs[3].get_ptr(0), sizeof(int32_t));
      };
  VulkanCommandReplayKey command_key;
  command_key.push(144);
  push_vulkan_command_key_range(command_key, allocs[0], 0, kScalarBytes);
  push_vulkan_command_key_range(command_key, allocs[1], 0, kScalarBytes);
  push_vulkan_command_key_range(command_key, allocs[2], 0, kScalarBytes);
  push_vulkan_command_key_range(command_key, allocs[3], 0, sizeof(int32_t));
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_scalar_divide_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_divide)) {
    enqueue_compute_op_lambda(record_divide, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_cg_update(Ndarray *direction,
                                             Ndarray *applied_direction,
                                             Ndarray *alpha,
                                             Ndarray *solution,
                                             Ndarray *residual,
                                             std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse CG update is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse CG update length must be in [1, UINT32_MAX].");
  check_vulkan_sparse_f32_vector("direction", direction, n);
  check_vulkan_sparse_f32_vector("applied direction", applied_direction,
                                 n);
  check_vulkan_sparse_f32_output(alpha);
  check_vulkan_sparse_f32_vector("solution", solution, n);
  check_vulkan_sparse_f32_vector("residual", residual, n);
  const std::array<DeviceAllocation, 5> allocs{
      direction->get_device_allocation(),
      applied_direction->get_device_allocation(),
      alpha->get_device_allocation(), solution->get_device_allocation(),
      residual->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse CG update buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {direction, applied_direction, alpha, solution,
                                residual};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse CG update requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_cg_update_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_cg_update(this, allocs[0], allocs[1], allocs[2],
                                 allocs[3], allocs[4], vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_update =
      [allocs, vector_bytes, pipeline, bindings, param_words, push_bytes,
       groups, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_cg_update_f32" : nullptr);
        cmdlist->buffer_barrier(allocs[3].get_ptr(0), vector_bytes);
        cmdlist->buffer_barrier(allocs[4].get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(145);
  push_vulkan_command_key_range(command_key, allocs[0], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[1], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[2], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[3], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[4], 0, vector_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_cg_update_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_update)) {
    enqueue_compute_op_lambda(record_update, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_cg_direction(Ndarray *residual,
                                                Ndarray *beta,
                                                Ndarray *direction,
                                                std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse CG direction is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse CG direction length must be in [1, UINT32_MAX].");
  check_vulkan_sparse_f32_vector("residual", residual, n);
  check_vulkan_sparse_f32_output(beta);
  check_vulkan_sparse_f32_vector("direction", direction, n);
  const std::array<DeviceAllocation, 3> allocs{
      residual->get_device_allocation(), beta->get_device_allocation(),
      direction->get_device_allocation()};
  TI_ERROR_IF(allocs[0] == allocs[1] || allocs[0] == allocs[2] ||
                  allocs[1] == allocs[2],
              "Vulkan sparse CG direction buffers must not alias.");
  const Ndarray *resources[] = {residual, beta, direction};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse CG direction requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_cg_direction_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_cg_direction(this, allocs[0], allocs[1], allocs[2],
                                    vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_direction =
      [allocs, vector_bytes, pipeline, bindings, param_words, push_bytes,
       groups, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_cg_direction_f32" : nullptr);
        cmdlist->buffer_barrier(allocs[2].get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(146);
  push_vulkan_command_key_range(command_key, allocs[0], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[1], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[2], 0, vector_bytes);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_cg_direction_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_direction)) {
    enqueue_compute_op_lambda(record_direction, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_minres_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t stage,
    bool limit_reached,
    bool has_preconditioner,
    bool stop_on_estimate) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse MINRES is only available on Vulkan.");
  TI_ERROR_IF(stage > 4 || !std::isfinite(absolute_tolerance) ||
                  !std::isfinite(relative_tolerance) ||
                  absolute_tolerance < 0.0f || relative_tolerance < 0.0f ||
                  (absolute_tolerance == 0.0f &&
                   relative_tolerance == 0.0f),
              "Vulkan sparse MINRES scalar stage received invalid "
              "controls.");
  check_vulkan_sparse_f32_output(initial_residual_squared);
  check_vulkan_sparse_f32_output(rhs_squared);
  check_vulkan_sparse_f32_output(dot);
  check_vulkan_sparse_i32_vector("MINRES state", state, 33);
  const std::array<DeviceAllocation, 4> allocs{
      initial_residual_squared->get_device_allocation(),
      rhs_squared->get_device_allocation(), dot->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse MINRES scalar buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {initial_residual_squared, rhs_squared, dot,
                                state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse MINRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_minres_scalar_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_minres_scalar(this, allocs[0], allocs[1], allocs[2],
                                     allocs[3])
          .bindings;
  const uint32_t flags = (limit_reached ? 1u : 0u) |
                         (has_preconditioner ? 2u : 0u) |
                         (stop_on_estimate ? 4u : 0u);
  const std::array<uint32_t, 4> param_words{
      vulkan_sparse_f32_word(absolute_tolerance),
      vulkan_sparse_f32_word(relative_tolerance), stage, flags};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;
  auto record_scalar =
      [allocs, pipeline, bindings, param_words, push_bytes,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, 1,
            1, 1,
            profiler_scopes ? "vulkan_sparse_minres_scalar_f32" : nullptr);
        cmdlist->buffer_barrier(allocs[3].get_ptr(0),
                                33 * sizeof(uint32_t));
      };
  VulkanCommandReplayKey command_key;
  command_key.push(150);
  push_vulkan_command_key_range(command_key, allocs[0], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[1], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[2], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[3], 0,
                                33 * sizeof(uint32_t));
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_minres_scalar_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_scalar)) {
    enqueue_compute_op_lambda(record_scalar, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_minres_vector_state(
    Ndarray *source,
    Ndarray *destination,
    Ndarray *state,
    std::size_t n,
    std::uint32_t coefficient,
    bool add) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse MINRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max() ||
                  coefficient >= 25,
              "Vulkan sparse MINRES vector stage received invalid "
              "geometry.");
  check_vulkan_sparse_f32_vector("MINRES source", source, n);
  check_vulkan_sparse_f32_vector("MINRES destination", destination, n);
  check_vulkan_sparse_i32_vector("MINRES state", state, 33);
  const std::array<DeviceAllocation, 3> allocs{
      source->get_device_allocation(), destination->get_device_allocation(),
      state->get_device_allocation()};
  TI_ERROR_IF(allocs[0] == allocs[1] || allocs[0] == allocs[2] ||
                  allocs[1] == allocs[2],
              "Vulkan sparse MINRES vector buffers must not alias.");
  const Ndarray *resources[] = {source, destination, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse MINRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_minres_vector_state_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_minres_vector_state(this, allocs[0], allocs[1],
                                            allocs[2], vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), coefficient, add ? 1u : 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_vector =
      [allocs, vector_bytes, pipeline, bindings, param_words, push_bytes,
       groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_minres_vector_state_f32"
                            : nullptr);
        cmdlist->buffer_barrier(allocs[1].get_ptr(0), vector_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(151);
  push_vulkan_command_key_range(command_key, allocs[0], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[1], 0, vector_bytes);
  push_vulkan_command_key_range(command_key, allocs[2], 0,
                                33 * sizeof(uint32_t));
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_minres_vector_state_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_vector)) {
    enqueue_compute_op_lambda(record_vector, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_minres_commit(
    Ndarray *v,
    Ndarray *r1,
    Ndarray *r2,
    Ndarray *lanczos_residual,
    Ndarray *w_older,
    Ndarray *w_old,
    Ndarray *w,
    Ndarray *solution,
    Ndarray *state,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse MINRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse MINRES commit received invalid geometry.");
  Ndarray *vectors[] = {v, r1, r2, lanczos_residual, w_older,
                        w_old, w, solution};
  const char *roles[] = {"v", "r1", "r2", "Lanczos residual",
                         "older direction", "old direction", "direction",
                         "solution"};
  for (std::size_t i = 0; i < std::size(vectors); ++i) {
    check_vulkan_sparse_f32_vector(roles[i], vectors[i], n);
  }
  check_vulkan_sparse_i32_vector("MINRES state", state, 33);
  const std::array<DeviceAllocation, 9> allocs{
      v->get_device_allocation(),
      r1->get_device_allocation(),
      r2->get_device_allocation(),
      lanczos_residual->get_device_allocation(),
      w_older->get_device_allocation(),
      w_old->get_device_allocation(),
      w->get_device_allocation(),
      solution->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse MINRES commit buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {v,       r1,    r2,       lanczos_residual,
                                w_older, w_old, w,        solution,
                                state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse MINRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_minres_commit_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_minres_commit(this, allocs, vector_bytes).bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record_commit =
      [allocs, vector_bytes, pipeline, bindings, param_words, push_bytes,
       groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes,
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_minres_commit_f32" : nullptr);
        for (std::size_t index : {1u, 2u, 4u, 5u, 6u, 7u}) {
          cmdlist->buffer_barrier(allocs[index].get_ptr(0), vector_bytes);
        }
      };
  VulkanCommandReplayKey command_key;
  command_key.push(152);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        command_key, allocs[index], 0,
        index == 8 ? 33 * sizeof(uint32_t) : vector_bytes);
  }
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_minres_commit_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_commit)) {
    enqueue_compute_op_lambda(record_commit, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_bicgstab_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot0,
    Ndarray *dot1,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t stage,
    bool limit_reached) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse BiCGSTAB is only available on Vulkan.");
  TI_ERROR_IF(stage > 6 || !std::isfinite(absolute_tolerance) ||
                  !std::isfinite(relative_tolerance) ||
                  absolute_tolerance < 0.0f || relative_tolerance < 0.0f ||
                  (absolute_tolerance == 0.0f &&
                   relative_tolerance == 0.0f),
              "Vulkan sparse BiCGSTAB scalar stage received invalid "
              "controls.");
  check_vulkan_sparse_f32_output(initial_residual_squared);
  check_vulkan_sparse_f32_output(rhs_squared);
  check_vulkan_sparse_f32_output(dot0);
  check_vulkan_sparse_f32_output(dot1);
  check_vulkan_sparse_i32_vector("BiCGSTAB state", state, 24);
  const std::array<DeviceAllocation, 5> allocs{
      initial_residual_squared->get_device_allocation(),
      rhs_squared->get_device_allocation(), dot0->get_device_allocation(),
      dot1->get_device_allocation(), state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse BiCGSTAB scalar buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {initial_residual_squared, rhs_squared, dot0,
                                dot1, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse BiCGSTAB requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_bicgstab_scalar_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_bicgstab_scalar(this, allocs).bindings;
  const std::array<uint32_t, 4> param_words{
      vulkan_sparse_f32_word(absolute_tolerance),
      vulkan_sparse_f32_word(relative_tolerance), stage,
      limit_reached ? 1u : 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, pipeline, bindings, param_words, push_bytes,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, param_words.data(), push_bytes, 1, 1, 1,
        profiler_scopes ? "vulkan_sparse_bicgstab_scalar_f32" : nullptr);
    cmdlist->buffer_barrier(allocs[4].get_ptr(0),
                            24 * sizeof(uint32_t));
  };
  VulkanCommandReplayKey key;
  key.push(153);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        key, allocs[index], 0,
        index == 4 ? 24 * sizeof(uint32_t) : sizeof(float32));
  }
  for (uint32_t word : param_words) {
    key.push(word);
  }
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_bicgstab_scalar_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_bicgstab_direction(
    Ndarray *residual,
    Ndarray *direction,
    Ndarray *operator_direction,
    Ndarray *state,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse BiCGSTAB is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse BiCGSTAB direction received invalid "
              "geometry.");
  check_vulkan_sparse_f32_vector("BiCGSTAB residual", residual, n);
  check_vulkan_sparse_f32_vector("BiCGSTAB direction", direction, n);
  check_vulkan_sparse_f32_vector(
      "BiCGSTAB operator direction", operator_direction, n);
  check_vulkan_sparse_i32_vector("BiCGSTAB state", state, 24);
  const std::array<DeviceAllocation, 4> allocs{
      residual->get_device_allocation(), direction->get_device_allocation(),
      operator_direction->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse BiCGSTAB direction buffers must not "
                  "alias.");
    }
  }
  const Ndarray *resources[] = {residual, direction, operator_direction,
                                state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse BiCGSTAB requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_bicgstab_direction_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_bicgstab_direction(this, allocs, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, vector_bytes, pipeline, bindings, param_words,
                 push_bytes, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
        1, 1,
        profiler_scopes ? "vulkan_sparse_bicgstab_direction_f32" : nullptr);
    cmdlist->buffer_barrier(allocs[1].get_ptr(0), vector_bytes);
  };
  VulkanCommandReplayKey key;
  key.push(154);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        key, allocs[index], 0,
        index == 3 ? 24 * sizeof(uint32_t) : vector_bytes);
  }
  key.push(static_cast<uint32_t>(n));
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_bicgstab_direction_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_bicgstab_intermediate(
    Ndarray *residual,
    Ndarray *operator_direction,
    Ndarray *intermediate,
    Ndarray *state,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse BiCGSTAB is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse BiCGSTAB intermediate received invalid "
              "geometry.");
  check_vulkan_sparse_f32_vector("BiCGSTAB residual", residual, n);
  check_vulkan_sparse_f32_vector(
      "BiCGSTAB operator direction", operator_direction, n);
  check_vulkan_sparse_f32_vector(
      "BiCGSTAB intermediate residual", intermediate, n);
  check_vulkan_sparse_i32_vector("BiCGSTAB state", state, 24);
  const std::array<DeviceAllocation, 4> allocs{
      residual->get_device_allocation(),
      operator_direction->get_device_allocation(),
      intermediate->get_device_allocation(), state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse BiCGSTAB intermediate buffers must not "
                  "alias.");
    }
  }
  const Ndarray *resources[] = {residual, operator_direction, intermediate,
                                state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse BiCGSTAB requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_bicgstab_intermediate_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_bicgstab_intermediate(this, allocs, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, vector_bytes, pipeline, bindings, param_words,
                 push_bytes, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
        1, 1,
        profiler_scopes ? "vulkan_sparse_bicgstab_intermediate_f32"
                        : nullptr);
    cmdlist->buffer_barrier(allocs[2].get_ptr(0), vector_bytes);
  };
  VulkanCommandReplayKey key;
  key.push(155);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        key, allocs[index], 0,
        index == 3 ? 24 * sizeof(uint32_t) : vector_bytes);
  }
  key.push(static_cast<uint32_t>(n));
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_bicgstab_intermediate_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_bicgstab_commit(
    Ndarray *solution_direction,
    Ndarray *solution_intermediate,
    Ndarray *intermediate,
    Ndarray *operator_intermediate,
    Ndarray *solution,
    Ndarray *residual,
    Ndarray *state,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse BiCGSTAB is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse BiCGSTAB commit received invalid geometry.");
  Ndarray *vectors[] = {solution_direction, solution_intermediate,
                        intermediate, operator_intermediate, solution,
                        residual};
  const char *roles[] = {"solution direction", "solution intermediate",
                         "intermediate residual", "operator intermediate",
                         "solution", "residual"};
  for (std::size_t i = 0; i < std::size(vectors); ++i) {
    check_vulkan_sparse_f32_vector(roles[i], vectors[i], n);
  }
  check_vulkan_sparse_i32_vector("BiCGSTAB state", state, 24);
  const std::array<DeviceAllocation, 7> allocs{
      solution_direction->get_device_allocation(),
      solution_intermediate->get_device_allocation(),
      intermediate->get_device_allocation(),
      operator_intermediate->get_device_allocation(),
      solution->get_device_allocation(), residual->get_device_allocation(),
      state->get_device_allocation()};
  TI_ERROR_IF(allocs[4] == allocs[5] || allocs[4] == allocs[6] ||
                  allocs[5] == allocs[6],
              "Vulkan sparse BiCGSTAB commit output/state buffers must not "
              "alias.");
  for (std::size_t index = 0; index < 4; ++index) {
    TI_ERROR_IF(allocs[index] == allocs[4] ||
                    allocs[index] == allocs[5] ||
                    allocs[index] == allocs[6],
                "Vulkan sparse BiCGSTAB commit inputs must not alias its "
                "output/state buffers.");
  }
  const Ndarray *resources[] = {
      solution_direction, solution_intermediate, intermediate,
      operator_intermediate, solution, residual, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse BiCGSTAB requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_bicgstab_commit_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_bicgstab_commit(this, allocs, vector_bytes).bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, vector_bytes, pipeline, bindings, param_words,
                 push_bytes, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
        1, 1,
        profiler_scopes ? "vulkan_sparse_bicgstab_commit_f32" : nullptr);
    cmdlist->buffer_barrier(allocs[4].get_ptr(0), vector_bytes);
    cmdlist->buffer_barrier(allocs[5].get_ptr(0), vector_bytes);
  };
  VulkanCommandReplayKey key;
  key.push(156);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        key, allocs[index], 0,
        index == 6 ? 24 * sizeof(uint32_t) : vector_bytes);
  }
  key.push(static_cast<uint32_t>(n));
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_bicgstab_commit_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_bicgstab_reconcile(
    Ndarray *true_residual,
    Ndarray *residual,
    Ndarray *shadow_residual,
    Ndarray *direction,
    Ndarray *operator_direction,
    Ndarray *solution,
    Ndarray *state,
    std::size_t n) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse BiCGSTAB is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse BiCGSTAB reconcile received invalid "
              "geometry.");
  Ndarray *vectors[] = {true_residual, residual, shadow_residual, direction,
                        operator_direction, solution};
  const char *roles[] = {"true residual", "residual", "shadow residual",
                         "direction", "operator direction", "solution"};
  for (std::size_t i = 0; i < std::size(vectors); ++i) {
    check_vulkan_sparse_f32_vector(roles[i], vectors[i], n);
  }
  check_vulkan_sparse_i32_vector("BiCGSTAB state", state, 24);
  const std::array<DeviceAllocation, 7> allocs{
      true_residual->get_device_allocation(),
      residual->get_device_allocation(),
      shadow_residual->get_device_allocation(),
      direction->get_device_allocation(),
      operator_direction->get_device_allocation(),
      solution->get_device_allocation(), state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse BiCGSTAB reconcile buffers must not "
                  "alias.");
    }
  }
  const Ndarray *resources[] = {true_residual, residual, shadow_residual,
                                direction, operator_direction, solution,
                                state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse BiCGSTAB requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_bicgstab_reconcile_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_bicgstab_reconcile(this, allocs, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> param_words{
      static_cast<uint32_t>(n), 0u, 0u, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, vector_bytes, pipeline, bindings, param_words,
                 push_bytes, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
        1, 1,
        profiler_scopes ? "vulkan_sparse_bicgstab_reconcile_f32"
                        : nullptr);
    for (std::size_t index = 1; index <= 5; ++index) {
      cmdlist->buffer_barrier(allocs[index].get_ptr(0), vector_bytes);
    }
  };
  VulkanCommandReplayKey key;
  key.push(157);
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(
        key, allocs[index], 0,
        index == 6 ? 24 * sizeof(uint32_t) : vector_bytes);
  }
  key.push(static_cast<uint32_t>(n));
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_bicgstab_reconcile_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_gmres_multi_dot(
    Ndarray *basis,
    Ndarray *work,
    Ndarray *partials,
    Ndarray *projection,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t basis_count,
    std::size_t group_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse GMRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max() ||
                  basis_stride < n ||
                  basis_stride > std::numeric_limits<uint32_t>::max() ||
                  basis_count == 0 || basis_count > 32 ||
                  group_count == 0 || group_count > 65535,
              "Vulkan sparse GMRES multi-dot received invalid geometry.");
  TI_ERROR_IF(!basis || basis->shape.size() != 1 ||
                  basis->get_element_data_type() != PrimitiveType::f32 ||
                  !basis->get_element_shape().empty() ||
                  basis->get_element_size() != sizeof(float32) ||
                  basis->get_nelement() < basis_count * basis_stride,
              "Vulkan sparse GMRES basis storage is too small.");
  check_vulkan_sparse_f32_vector("GMRES work", work, n);
  TI_ERROR_IF(!partials || partials->shape.size() != 1 ||
                  partials->get_element_data_type() != PrimitiveType::f32 ||
                  !partials->get_element_shape().empty() ||
                  partials->get_element_size() != sizeof(float32) ||
                  partials->get_nelement() <
                      basis_count * group_count,
              "Vulkan sparse GMRES multi-dot partial storage is too small.");
  TI_ERROR_IF(!projection || projection->shape.size() != 1 ||
                  projection->get_element_data_type() != PrimitiveType::f32 ||
                  !projection->get_element_shape().empty() ||
                  projection->get_element_size() != sizeof(float32) ||
                  projection->get_nelement() < basis_count,
              "Vulkan sparse GMRES projection storage is too small.");
  check_vulkan_sparse_i32_vector("GMRES state", state, 32);
  const std::array<DeviceAllocation, 5> all_allocs{
      basis->get_device_allocation(), work->get_device_allocation(),
      partials->get_device_allocation(), projection->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < all_allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < all_allocs.size(); ++j) {
      TI_ERROR_IF(all_allocs[i] == all_allocs[j],
                  "Vulkan sparse GMRES multi-dot buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {basis, work, partials, projection, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t basis_bytes =
      basis->get_nelement() * sizeof(float32);
  const size_t vector_bytes = n * sizeof(float32);
  const size_t partial_bytes =
      basis_count * group_count * sizeof(float32);
  const size_t projection_bytes = basis_count * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse GMRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *partial_pipeline =
      cache.sparse_gmres_multi_dot_partial_pipeline(device);
  const std::array<DeviceAllocation, 4> partial_allocs{
      all_allocs[0], all_allocs[1], all_allocs[2], all_allocs[4]};
  ShaderResourceSet *partial_bindings =
      cache
          .bind_sparse_gmres_multi_dot_partial(
              this, partial_allocs, basis_bytes, vector_bytes, partial_bytes)
          .bindings;
  const std::array<uint32_t, 4> partial_params{
      static_cast<uint32_t>(n), static_cast<uint32_t>(basis_stride),
      static_cast<uint32_t>(basis_count),
      static_cast<uint32_t>(group_count)};
  const bool profiler_scopes = profiler != nullptr;
  auto record_partial =
      [partial_allocs, partial_bytes, partial_pipeline, partial_bindings,
       partial_params,
       profiler_scopes](Device *, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, partial_pipeline, partial_bindings,
            partial_params.data(), sizeof(partial_params),
            partial_params[3], 1, 1,
            profiler_scopes
                ? "vulkan_sparse_gmres_multi_dot_partial_f32"
                : nullptr);
        cmdlist->buffer_barrier(partial_allocs[2].get_ptr(0),
                                partial_bytes);
      };
  VulkanCommandReplayKey partial_key;
  partial_key.push(160);
  push_vulkan_command_key_range(partial_key, partial_allocs[0], 0,
                                basis_bytes);
  push_vulkan_command_key_range(partial_key, partial_allocs[1], 0,
                                vector_bytes);
  push_vulkan_command_key_range(partial_key, partial_allocs[2], 0,
                                partial_bytes);
  push_vulkan_command_key_range(partial_key, partial_allocs[3], 0,
                                32 * sizeof(uint32_t));
  for (uint32_t word : partial_params) {
    partial_key.push(word);
  }
  partial_key.push_ptr(partial_pipeline);
  partial_key.push_ptr(partial_bindings);
  if (!cache.sparse_gmres_multi_dot_partial_command_replay.submit_or_record(
          this, device, partial_key, profiler_scopes, record_partial)) {
    enqueue_compute_op_lambda(record_partial, {});
  }

  Pipeline *final_pipeline =
      cache.sparse_gmres_multi_dot_final_pipeline(device);
  const std::array<DeviceAllocation, 3> final_allocs{
      all_allocs[2], all_allocs[3], all_allocs[4]};
  ShaderResourceSet *final_bindings =
      cache
          .bind_sparse_gmres_multi_dot_final(
              this, final_allocs, partial_bytes, projection_bytes)
          .bindings;
  const std::array<uint32_t, 4> final_params{
      static_cast<uint32_t>(group_count),
      static_cast<uint32_t>(basis_count), 0u, 0u};
  auto record_final =
      [final_allocs, projection_bytes, final_pipeline, final_bindings,
       final_params,
       profiler_scopes](Device *, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, final_pipeline, final_bindings, final_params.data(),
            sizeof(final_params), final_params[1], 1, 1,
            profiler_scopes
                ? "vulkan_sparse_gmres_multi_dot_final_f32"
                : nullptr);
        cmdlist->buffer_barrier(final_allocs[1].get_ptr(0),
                                projection_bytes);
      };
  VulkanCommandReplayKey final_key;
  final_key.push(161);
  push_vulkan_command_key_range(final_key, final_allocs[0], 0,
                                partial_bytes);
  push_vulkan_command_key_range(final_key, final_allocs[1], 0,
                                projection_bytes);
  push_vulkan_command_key_range(final_key, final_allocs[2], 0,
                                32 * sizeof(uint32_t));
  for (uint32_t word : final_params) {
    final_key.push(word);
  }
  final_key.push_ptr(final_pipeline);
  final_key.push_ptr(final_bindings);
  if (!cache.sparse_gmres_multi_dot_final_command_replay.submit_or_record(
          this, device, final_key, profiler_scopes, record_final)) {
    enqueue_compute_op_lambda(record_final, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_gmres_projection(
    Ndarray *basis,
    Ndarray *work,
    Ndarray *projection,
    Ndarray *hessenberg,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t restart,
    std::size_t step,
    std::size_t pass) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse GMRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max() ||
                  basis_stride < n ||
                  basis_stride > std::numeric_limits<uint32_t>::max() ||
                  (restart != 8 && restart != 16 && restart != 32) ||
                  step >= restart || pass > 1,
              "Vulkan sparse GMRES projection received invalid controls.");
  check_vulkan_sparse_f32_vector(
      "GMRES basis", basis, (restart + 1) * basis_stride);
  check_vulkan_sparse_f32_vector("GMRES work", work, n);
  check_vulkan_sparse_f32_vector("GMRES projection", projection, restart);
  check_vulkan_sparse_f32_vector(
      "GMRES Hessenberg storage", hessenberg, restart * (restart + 1));
  check_vulkan_sparse_i32_vector("GMRES state", state, 32);
  const std::array<DeviceAllocation, 5> allocs{
      basis->get_device_allocation(), work->get_device_allocation(),
      projection->get_device_allocation(), hessenberg->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse GMRES projection buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {basis, work, projection, hessenberg, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t basis_bytes =
      (restart + 1) * basis_stride * sizeof(float32);
  const size_t vector_bytes = n * sizeof(float32);
  const size_t projection_bytes = restart * sizeof(float32);
  const size_t hessenberg_bytes =
      restart * (restart + 1) * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse GMRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_gmres_projection_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_gmres_projection(
              this, allocs, basis_bytes, vector_bytes, projection_bytes,
              hessenberg_bytes)
          .bindings;
  const std::array<uint32_t, 5> params{
      static_cast<uint32_t>(n), static_cast<uint32_t>(basis_stride),
      static_cast<uint32_t>(restart), static_cast<uint32_t>(step),
      static_cast<uint32_t>(pass)};
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record =
      [allocs, vector_bytes, hessenberg_bytes, pipeline, bindings, params,
       groups, profiler_scopes](Device *, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, params.data(), sizeof(params),
            groups, 1, 1,
            profiler_scopes ? "vulkan_sparse_gmres_projection_f32"
                            : nullptr);
        cmdlist->buffer_barrier(allocs[1].get_ptr(0), vector_bytes);
        cmdlist->buffer_barrier(allocs[3].get_ptr(0), hessenberg_bytes);
      };
  VulkanCommandReplayKey key;
  key.push(162);
  push_vulkan_command_key_range(key, allocs[0], 0, basis_bytes);
  push_vulkan_command_key_range(key, allocs[1], 0, vector_bytes);
  push_vulkan_command_key_range(key, allocs[2], 0, projection_bytes);
  push_vulkan_command_key_range(key, allocs[3], 0, hessenberg_bytes);
  push_vulkan_command_key_range(key, allocs[4], 0,
                                32 * sizeof(uint32_t));
  for (uint32_t word : params) {
    key.push(word);
  }
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_gmres_projection_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_gmres_basis(
    Ndarray *source,
    Ndarray *basis,
    Ndarray *current,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t row,
    std::size_t mode) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse GMRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max() ||
                  basis_stride < n ||
                  basis_stride > std::numeric_limits<uint32_t>::max() ||
                  row > 32 || mode > 2,
              "Vulkan sparse GMRES basis update received invalid controls.");
  check_vulkan_sparse_f32_vector("GMRES basis source", source, n);
  check_vulkan_sparse_f32_vector("GMRES current basis vector", current, n);
  TI_ERROR_IF(!basis || basis->shape.size() != 1 ||
                  basis->get_element_data_type() != PrimitiveType::f32 ||
                  !basis->get_element_shape().empty() ||
                  basis->get_element_size() != sizeof(float32) ||
                  basis->get_nelement() < (row + 1) * basis_stride,
              "Vulkan sparse GMRES basis storage is too small.");
  check_vulkan_sparse_i32_vector("GMRES state", state, 32);
  const std::array<DeviceAllocation, 4> allocs{
      source->get_device_allocation(), basis->get_device_allocation(),
      current->get_device_allocation(), state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse GMRES basis buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {source, basis, current, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t vector_bytes = n * sizeof(float32);
  const size_t basis_bytes = basis->get_nelement() * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse GMRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_gmres_basis_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_gmres_basis(
               this, allocs, vector_bytes, basis_bytes)
          .bindings;
  const std::array<uint32_t, 4> params{
      static_cast<uint32_t>(n), static_cast<uint32_t>(basis_stride),
      static_cast<uint32_t>(row), static_cast<uint32_t>(mode)};
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, basis_bytes, vector_bytes, pipeline, bindings,
                 params, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, params.data(), sizeof(params), groups,
        1, 1, profiler_scopes ? "vulkan_sparse_gmres_basis_f32" : nullptr);
    cmdlist->buffer_barrier(allocs[1].get_ptr(0), basis_bytes);
    cmdlist->buffer_barrier(allocs[2].get_ptr(0), vector_bytes);
  };
  VulkanCommandReplayKey key;
  key.push(163);
  push_vulkan_command_key_range(key, allocs[0], 0, vector_bytes);
  push_vulkan_command_key_range(key, allocs[1], 0, basis_bytes);
  push_vulkan_command_key_range(key, allocs[2], 0,
                                vector_bytes);
  push_vulkan_command_key_range(key, allocs[3], 0,
                                32 * sizeof(uint32_t));
  for (uint32_t word : params) {
    key.push(word);
  }
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_gmres_basis_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_gmres_combine(
    Ndarray *basis,
    Ndarray *coefficients,
    Ndarray *update,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse GMRES is only available on Vulkan.");
  TI_ERROR_IF(n == 0 || n > std::numeric_limits<uint32_t>::max() ||
                  basis_stride < n ||
                  basis_stride > std::numeric_limits<uint32_t>::max(),
              "Vulkan sparse GMRES basis combination received invalid "
              "geometry.");
  const std::size_t restart = coefficients->get_nelement();
  TI_ERROR_IF(restart != 8 && restart != 16 && restart != 32,
              "Vulkan sparse GMRES coefficients require restart 8, 16, or "
              "32.");
  TI_ERROR_IF(!basis || basis->shape.size() != 1 ||
                  basis->get_element_data_type() != PrimitiveType::f32 ||
                  !basis->get_element_shape().empty() ||
                  basis->get_element_size() != sizeof(float32) ||
                  basis->get_nelement() < restart * basis_stride,
              "Vulkan GMRES/FGMRES basis storage is too small.");
  check_vulkan_sparse_f32_vector(
      "GMRES coefficients", coefficients, restart);
  check_vulkan_sparse_f32_vector("GMRES update", update, n);
  check_vulkan_sparse_i32_vector("GMRES state", state, 32);
  const std::array<DeviceAllocation, 4> allocs{
      basis->get_device_allocation(), coefficients->get_device_allocation(),
      update->get_device_allocation(), state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse GMRES combination buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {basis, coefficients, update, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t basis_bytes =
      basis->get_nelement() * sizeof(float32);
  const size_t coefficient_bytes = restart * sizeof(float32);
  const size_t vector_bytes = n * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse GMRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_gmres_combine_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_gmres_combine(
              this, allocs, basis_bytes, coefficient_bytes, vector_bytes)
          .bindings;
  const std::array<uint32_t, 4> params{
      static_cast<uint32_t>(n), static_cast<uint32_t>(basis_stride), 0u, 0u};
  constexpr std::size_t kMaxDispatchGroups = 65535;
  const uint32_t groups = static_cast<uint32_t>(std::min(
      kMaxDispatchGroups, (n + kBlockSize - 1) / kBlockSize));
  const bool profiler_scopes = profiler != nullptr;
  auto record = [allocs, vector_bytes, pipeline, bindings, params, groups,
                 profiler_scopes](Device *, CommandList *cmdlist) {
    dispatch_pipeline_with_push_constants(
        cmdlist, pipeline, bindings, params.data(), sizeof(params), groups,
        1, 1,
        profiler_scopes ? "vulkan_sparse_gmres_combine_f32" : nullptr);
    cmdlist->buffer_barrier(allocs[2].get_ptr(0), vector_bytes);
  };
  VulkanCommandReplayKey key;
  key.push(164);
  push_vulkan_command_key_range(key, allocs[0], 0, basis_bytes);
  push_vulkan_command_key_range(key, allocs[1], 0, coefficient_bytes);
  push_vulkan_command_key_range(key, allocs[2], 0, vector_bytes);
  push_vulkan_command_key_range(key, allocs[3], 0,
                                32 * sizeof(uint32_t));
  for (uint32_t word : params) {
    key.push(word);
  }
  key.push(groups);
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_gmres_combine_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_gmres_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot0,
    Ndarray *dot1,
    Ndarray *hessenberg,
    Ndarray *cosines,
    Ndarray *sines,
    Ndarray *g,
    Ndarray *coefficients,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::size_t restart,
    std::size_t max_iterations,
    std::size_t stage,
    std::size_t step,
    bool limit_reached) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse GMRES is only available on Vulkan.");
  TI_ERROR_IF((restart != 8 && restart != 16 && restart != 32) ||
                  max_iterations > std::numeric_limits<uint32_t>::max() ||
                  stage > 4 || step >= restart ||
                  !std::isfinite(absolute_tolerance) ||
                  !std::isfinite(relative_tolerance) ||
                  absolute_tolerance < 0.0f || relative_tolerance < 0.0f ||
                  (absolute_tolerance == 0.0f &&
                   relative_tolerance == 0.0f),
              "Vulkan sparse GMRES scalar stage received invalid "
              "controls.");
  check_vulkan_sparse_f32_output(initial_residual_squared);
  check_vulkan_sparse_f32_output(rhs_squared);
  check_vulkan_sparse_f32_output(dot0);
  check_vulkan_sparse_f32_output(dot1);
  check_vulkan_sparse_f32_vector(
      "GMRES Hessenberg storage", hessenberg, restart * (restart + 1));
  check_vulkan_sparse_f32_vector("GMRES cosines", cosines, restart);
  check_vulkan_sparse_f32_vector("GMRES sines", sines, restart);
  check_vulkan_sparse_f32_vector("GMRES least-squares rhs", g, restart + 1);
  check_vulkan_sparse_f32_vector(
      "GMRES coefficients", coefficients, restart);
  check_vulkan_sparse_i32_vector("GMRES state", state, 32);
  const std::array<DeviceAllocation, 10> allocs{
      initial_residual_squared->get_device_allocation(),
      rhs_squared->get_device_allocation(), dot0->get_device_allocation(),
      dot1->get_device_allocation(), hessenberg->get_device_allocation(),
      cosines->get_device_allocation(), sines->get_device_allocation(),
      g->get_device_allocation(), coefficients->get_device_allocation(),
      state->get_device_allocation()};
  for (std::size_t i = 0; i < allocs.size(); ++i) {
    for (std::size_t j = i + 1; j < allocs.size(); ++j) {
      TI_ERROR_IF(allocs[i] == allocs[j],
                  "Vulkan sparse GMRES scalar buffers must not alias.");
    }
  }
  const Ndarray *resources[] = {initial_residual_squared, rhs_squared, dot0,
                                dot1, hessenberg, cosines, sines, g,
                                coefficients, state};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  const size_t hessenberg_bytes =
      restart * (restart + 1) * sizeof(float32);
  const size_t restart_bytes = restart * sizeof(float32);
  const size_t g_bytes = (restart + 1) * sizeof(float32);
  Device *device = get_compute_device();
  TI_ERROR_IF(!device, "Vulkan sparse GMRES requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_gmres_scalar_pipeline(device);
  ShaderResourceSet *bindings =
      cache
          .bind_sparse_gmres_scalar(
              this, allocs, hessenberg_bytes, restart_bytes, g_bytes)
          .bindings;
  const std::array<uint32_t, 7> params{
      vulkan_sparse_f32_word(absolute_tolerance),
      vulkan_sparse_f32_word(relative_tolerance),
      static_cast<uint32_t>(restart),
      static_cast<uint32_t>(max_iterations),
      static_cast<uint32_t>(stage), static_cast<uint32_t>(step),
      limit_reached ? 1u : 0u};
  const bool profiler_scopes = profiler != nullptr;
  auto record =
      [allocs, hessenberg_bytes, restart_bytes, g_bytes, pipeline, bindings,
       params, profiler_scopes](Device *, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, params.data(), sizeof(params), 1, 1,
            1, profiler_scopes ? "vulkan_sparse_gmres_scalar_f32"
                               : nullptr);
        cmdlist->buffer_barrier(allocs[4].get_ptr(0), hessenberg_bytes);
        cmdlist->buffer_barrier(allocs[5].get_ptr(0), restart_bytes);
        cmdlist->buffer_barrier(allocs[6].get_ptr(0), restart_bytes);
        cmdlist->buffer_barrier(allocs[7].get_ptr(0), g_bytes);
        cmdlist->buffer_barrier(allocs[8].get_ptr(0), restart_bytes);
        cmdlist->buffer_barrier(allocs[9].get_ptr(0),
                                32 * sizeof(uint32_t));
      };
  VulkanCommandReplayKey key;
  key.push(165);
  const std::array<size_t, 10> bytes{
      sizeof(float32), sizeof(float32), sizeof(float32), sizeof(float32),
      hessenberg_bytes, restart_bytes, restart_bytes, g_bytes,
      restart_bytes, 32 * sizeof(uint32_t)};
  for (std::size_t index = 0; index < allocs.size(); ++index) {
    push_vulkan_command_key_range(key, allocs[index], 0, bytes[index]);
  }
  for (uint32_t word : params) {
    key.push(word);
  }
  key.push_ptr(pipeline);
  key.push_ptr(bindings);
  if (!cache.sparse_gmres_scalar_command_replay.submit_or_record(
          this, device, key, profiler_scopes, record)) {
    enqueue_compute_op_lambda(record, {});
  }
  return cache.allocated_bytes();
}

std::size_t Program::vulkan_sparse_convergence(
    Ndarray *residual_squared,
    Ndarray *status,
    Ndarray *completed_iterations,
    Ndarray *rhs_squared,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t iteration) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse convergence is only available on Vulkan.");
  TI_ERROR_IF(!std::isfinite(absolute_tolerance) ||
                  !std::isfinite(relative_tolerance) ||
                  absolute_tolerance < 0.0f || relative_tolerance < 0.0f ||
                  (absolute_tolerance == 0.0f &&
                   relative_tolerance == 0.0f),
              "Vulkan sparse convergence requires finite non-negative atol "
              "and rtol with at least one positive tolerance.");
  check_vulkan_sparse_f32_output(residual_squared);
  check_vulkan_sparse_f32_output(rhs_squared);
  check_vulkan_sparse_i32_status(status);
  check_vulkan_sparse_i32_status(completed_iterations);
  const std::array<DeviceAllocation, 4> allocs{
      residual_squared->get_device_allocation(),
      status->get_device_allocation(),
      completed_iterations->get_device_allocation(),
      rhs_squared->get_device_allocation()};
  TI_ERROR_IF(allocs[0] == allocs[1] || allocs[0] == allocs[2] ||
                  allocs[0] == allocs[3] || allocs[1] == allocs[2] ||
                  allocs[1] == allocs[3] || allocs[2] == allocs[3],
              "Vulkan sparse convergence buffers must not alias.");
  const Ndarray *resources[] = {residual_squared, status,
                                completed_iterations, rhs_squared};
  retain_ndarrays_for_external_submission(resources, std::size(resources));
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse convergence requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  auto &cache = *cache_lease;
  Pipeline *pipeline = cache.sparse_convergence_pipeline(device);
  ShaderResourceSet *bindings =
      cache.bind_sparse_convergence(this, allocs[0], allocs[1], allocs[2],
                                    allocs[3])
          .bindings;
  const std::array<uint32_t, 4> param_words{
      vulkan_sparse_f32_word(absolute_tolerance),
      vulkan_sparse_f32_word(relative_tolerance), iteration, 0u};
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;
  auto record_convergence =
      [allocs, pipeline, bindings, param_words, push_bytes,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, 1, 1,
            1, profiler_scopes ? "vulkan_sparse_convergence_f32" : nullptr);
        cmdlist->buffer_barrier(allocs[1].get_ptr(0), sizeof(int32_t));
        cmdlist->buffer_barrier(allocs[2].get_ptr(0), sizeof(int32_t));
      };
  VulkanCommandReplayKey command_key;
  command_key.push(147);
  push_vulkan_command_key_range(command_key, allocs[0], 0, sizeof(float32));
  push_vulkan_command_key_range(command_key, allocs[1], 0, sizeof(int32_t));
  push_vulkan_command_key_range(command_key, allocs[2], 0, sizeof(int32_t));
  push_vulkan_command_key_range(command_key, allocs[3], 0, sizeof(float32));
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.sparse_convergence_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_convergence)) {
    enqueue_compute_op_lambda(record_convergence, {});
  }
  return cache.allocated_bytes();
}

bool Program::vulkan_add_merge_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_add_merge_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 4) {
    return true;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

std::size_t Program::vulkan_add_merge_ndarray(Ndarray *src,
                                              Ndarray *dst,
                                              int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_add_merge_value_type_available(value_type),
              "Vulkan native add-merge does not support the requested value "
              "type.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(!src || !dst, "Vulkan native add-merge received null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || dst->shape.size() != 1,
              "Vulkan native add-merge expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != dst->get_nelement(),
              "Vulkan native add-merge source and destination sizes differ.");
  const size_t src_element_size = src->get_element_size();
  const size_t dst_element_size = dst->get_element_size();
  TI_ERROR_IF(src_element_size != dst_element_size ||
                  src_element_size < value_size ||
                  src_element_size % value_size != 0,
              "Vulkan native add-merge payload does not match value type.");
  const size_t scalar_items =
      src->get_nelement() * (src_element_size / value_size);
  return vulkan_add_merge_storage_impl(
      this, src->ndarray_alloc_, dst->ndarray_alloc_, scalar_items, value_size,
      value_size, value_type, 0, value_size, 0, value_size);
}

std::size_t Program::vulkan_add_merge_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_add_merge_value_type_available(value_type),
              "Vulkan native strided add-merge does not support the requested "
              "value type.");
  return vulkan_add_merge_ndarray_impl(this, src, dst, value_type, src_offset,
                                       src_stride, dst_offset, dst_stride);
}

std::size_t Program::vulkan_add_merge_dense_field(Ndarray *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field add-merge is only available on "
              "Vulkan.");
  TI_ERROR_IF(!src || !dst,
              "Vulkan native dense field add-merge received a null input.");
  TI_ERROR_IF(!vulkan_add_merge_value_type_available(value_type),
              "Vulkan native dense field add-merge does not support the "
              "requested value type.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(src->shape.size() != 1 || src->get_nelement() != n ||
                  src->get_element_size() != value_size,
              "Vulkan native dense field add-merge source shape or dtype "
              "mismatch.");
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  return vulkan_add_merge_storage_impl(
      this, src->ndarray_alloc_, dst_alloc, n, value_size, value_size,
      value_type, 0, value_size, dst_ptr.offset, dst_stride);
}

std::size_t Program::vulkan_add_merge_dense_field_packed(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field add-merge is only available "
              "on Vulkan.");
  TI_ERROR_IF(!src || !dst,
              "Vulkan native packed dense field add-merge received a null "
              "field.");
  TI_ERROR_IF(!vulkan_add_merge_value_type_available(value_type),
              "Vulkan native packed dense field add-merge does not support "
              "the requested value type.");
  TI_ERROR_IF(lane_count <= 0,
              "Vulkan native packed dense field add-merge received an invalid "
              "lane count.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  const size_t lanes = static_cast<size_t>(lane_count);
  TI_ERROR_IF(n > std::numeric_limits<size_t>::max() / lanes,
              "Vulkan native packed dense field add-merge received an "
              "oversized request.");
  const size_t scalar_items = n * lanes;
  if (scalar_items == 0) {
    return 0;
  }
  const size_t expected_stride = lanes * value_size;
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, value_size);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(src_stride != expected_stride || dst_stride != expected_stride,
              "Vulkan native packed dense field add-merge expects packed "
              "contiguous dense MatrixField gradients.");
  DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  return vulkan_add_merge_storage_impl(
      this, src_alloc, dst_alloc, scalar_items, value_size, value_size,
      value_type, src_ptr.offset, value_size, dst_ptr.offset, value_size);
}

std::size_t Program::vulkan_add_scalar_field_to_dense_field(SNode *src,
                                                            SNode *dst,
                                                            int value_type,
                                                            std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scalar-to-dense add is only available on "
              "Vulkan.");
  TI_ERROR_IF(!src || !dst,
              "Vulkan native scalar-to-dense add received a null field.");
  TI_ERROR_IF(!vulkan_add_merge_value_type_available(value_type),
              "Vulkan native scalar-to-dense add does not support the "
              "requested value type.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  return vulkan_add_merge_storage_impl(
      this, src_alloc, dst_alloc, n, value_size, value_size, value_type,
      src_ptr.offset, 0, dst_ptr.offset, dst_stride);
}

bool Program::vulkan_indexed_copy_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scatter_add_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_scatter_add_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type == 0 || value_type == 2) {
    return true;
  }
  if (value_type == 1) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_atomic_float_add) != 0;
  }
  if (value_type == 3 || value_type == 4) {
    const auto caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  if (value_type == 5) {
    const auto caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_float64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_float64_add) != 0;
  }
  return false;
}

bool Program::vulkan_bucket_builder_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_grouped_reduce_available() const {
  return compile_config().arch == Arch::vulkan;
}

bool Program::vulkan_grouped_reduce_value_type_available(int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type >= 0 && value_type <= 2) {
    return true;
  }
  if (value_type == 3 || value_type == 4) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_int64) != 0;
  }
  if (value_type == 5) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_float64) != 0;
  }
  return false;
}

bool Program::vulkan_grouped_reduce_atomic_value_type_available(
    int value_type) const {
  if (compile_config().arch != Arch::vulkan) {
    return false;
  }
  if (value_type == 0 || value_type == 2) {
    return true;
  }
  if (value_type == 1) {
    return const_cast<Program *>(this)
               ->get_device_caps()
               .get(DeviceCapability::spirv_has_atomic_float_add) != 0;
  }
  if (value_type == 3 || value_type == 4) {
    const auto caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_int64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_int64) != 0;
  }
  if (value_type == 5) {
    const auto caps = const_cast<Program *>(this)->get_device_caps();
    return caps.get(DeviceCapability::spirv_has_float64) != 0 &&
           caps.get(DeviceCapability::spirv_has_atomic_float64_add) != 0;
  }
  return false;
}

std::size_t Program::vulkan_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                              Ndarray *values,
                                                              Ndarray *output,
                                                              int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_grouped_reduce_atomic_ndarray(keys, values, output, 0, op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                          Ndarray *values,
                                                          Ndarray *output,
                                                          int value_type,
                                                          int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native grouped reduce is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !output,
              "Vulkan native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1,
              "Vulkan native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "Vulkan native grouped reduce output must contain at least one group.");
  TI_ERROR_IF(!vulkan_grouped_reduce_atomic_value_type_available(value_type),
              "Vulkan native grouped reduce does not support the requested "
              "value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "Unsupported Vulkan grouped reduce value type.");
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != value_size ||
                  output->get_element_size() != value_size,
              "Vulkan native grouped reduce dtype does not match the "
              "requested value type.");
  TI_ERROR_IF(op != 0, "Vulkan native grouped reduce currently supports only sum.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native grouped reduce input is too large for u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native grouped reduce requires a compute device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const size_t input_bytes = n * sizeof(int32_t);
  const size_t values_bytes = n * value_size;
  const size_t output_bytes = num_groups * value_size;
  Pipeline *zero_pipeline = cache.grouped_reduce_zero_pipeline(value_type);
  Pipeline *atomic_pipeline = cache.grouped_reduce_atomic_pipeline(value_type);
  TI_ERROR_IF(!zero_pipeline || !atomic_pipeline,
              "Vulkan native grouped reduce could not find a pipeline for the "
              "requested value type.");
  ShaderResourceSet *zero_bindings =
      cache
          .bind_grouped_reduce_zero_resource_set(
              this, value_type, output_alloc, 0, output_bytes)
          .bindings;
  ShaderResourceSet *atomic_bindings =
      cache
          .bind_grouped_reduce_atomic_resource_set(
              this, value_type, keys_alloc, 0, input_bytes, values_alloc, 0,
              values_bytes, output_alloc, 0, output_bytes)
          .bindings;
  const uint32_t zero_groups =
      static_cast<uint32_t>((num_groups + kBlockSize - 1) / kBlockSize);
  const uint32_t reduce_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  auto record_grouped_reduce_atomic =
      [keys_alloc, values_alloc, output_alloc, input_bytes, output_bytes,
       zero_pipeline, atomic_pipeline, zero_bindings, atomic_bindings,
       zero_groups, reduce_groups, value_type, values_bytes,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, zero_pipeline, zero_bindings, zero_groups, 1,
                          1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_zero_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_zero_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_zero_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_zero_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_zero_f64"
                                     : "vulkan_grouped_reduce_zero_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        if (reduce_groups == 0) {
          return;
        }
        dispatch_pipeline(cmdlist, atomic_pipeline, atomic_bindings,
                          reduce_groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_atomic_sum_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_atomic_sum_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_atomic_sum_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_atomic_sum_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_atomic_sum_f64"
                                     : "vulkan_grouped_reduce_atomic_sum_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(20);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(keys_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(keys_alloc));
  command_key.push(static_cast<uint64_t>(input_bytes));
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(static_cast<uint64_t>(values_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(static_cast<uint64_t>(output_bytes));
  command_key.push(zero_groups);
  command_key.push(reduce_groups);
  command_key.push_ptr(zero_pipeline);
  command_key.push_ptr(atomic_pipeline);
  command_key.push_ptr(zero_bindings);
  command_key.push_ptr(atomic_bindings);
  if (!cache.grouped_reduce_atomic_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_grouped_reduce_atomic)) {
    enqueue_compute_op_lambda(record_grouped_reduce_atomic, {});
  }
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_dense_field(
    SNode *keys,
    SNode *values,
    SNode *output,
    int value_type,
    std::size_t n,
    std::size_t num_groups,
    int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field grouped reduce is only available on "
              "Vulkan.");
  TI_ERROR_IF(!keys || !values || !output,
              "Vulkan native dense field grouped reduce received a null field.");
  TI_ERROR_IF(num_groups == 0,
              "Vulkan native dense field grouped reduce output must contain at "
              "least one group.");
  TI_ERROR_IF(!vulkan_grouped_reduce_atomic_value_type_available(value_type),
              "Vulkan native dense field grouped reduce does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native dense field grouped reduce received an "
              "unsupported value type.");
  DevicePtr keys_ptr = get_dense_field_device_ptr(keys);
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr output_ptr = get_dense_field_device_ptr(output);
  const size_t keys_stride = get_dense_field_stride(keys, sizeof(int32_t));
  const size_t values_stride = get_dense_field_stride(values, value_size);
  const size_t output_stride = get_dense_field_stride(output, value_size);
  TI_ERROR_IF(keys_stride != sizeof(int32_t) ||
                  values_stride != value_size ||
                  output_stride != value_size,
              "Vulkan native dense field grouped reduce requires contiguous "
              "keys, values, and output fields.");
  TI_ERROR_IF(op != 0,
              "Vulkan native dense field grouped reduce currently supports "
              "only sum.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field grouped reduce input is too large for "
              "u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(device == nullptr,
              "Vulkan native dense field grouped reduce requires a compute "
              "device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation keys_alloc{keys_ptr.device, keys_ptr.alloc_id};
  const DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  const DeviceAllocation output_alloc{output_ptr.device, output_ptr.alloc_id};
  const size_t input_bytes = n * sizeof(int32_t);
  const size_t values_bytes = n * value_size;
  const size_t output_bytes = num_groups * value_size;
  Pipeline *zero_pipeline = cache.grouped_reduce_zero_pipeline(value_type);
  Pipeline *atomic_pipeline = cache.grouped_reduce_atomic_pipeline(value_type);
  TI_ERROR_IF(!zero_pipeline || !atomic_pipeline,
              "Vulkan native dense field grouped reduce could not find a "
              "pipeline for the requested value type.");
  ShaderResourceSet *zero_bindings =
      cache
          .bind_grouped_reduce_zero_resource_set(
              this, value_type, output_alloc, output_ptr.offset, output_bytes)
          .bindings;
  ShaderResourceSet *atomic_bindings =
      cache
          .bind_grouped_reduce_atomic_resource_set(
              this, value_type, keys_alloc, keys_ptr.offset, input_bytes,
              values_alloc, values_ptr.offset, values_bytes, output_alloc,
              output_ptr.offset, output_bytes)
          .bindings;
  const uint32_t zero_groups =
      static_cast<uint32_t>((num_groups + kBlockSize - 1) / kBlockSize);
  const uint32_t reduce_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  auto record_grouped_reduce_atomic =
      [keys_alloc, values_alloc, output_alloc, keys_ptr, values_ptr, output_ptr,
       input_bytes, output_bytes, zero_pipeline, atomic_pipeline,
       zero_bindings, atomic_bindings, zero_groups, reduce_groups, value_type,
       values_bytes, profiler_scopes](Device * /*op_device*/,
                                      CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, zero_pipeline, zero_bindings, zero_groups, 1,
                          1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_zero_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_zero_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_zero_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_zero_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_zero_f64"
                                     : "vulkan_grouped_reduce_zero_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        if (reduce_groups == 0) {
          return;
        }
        dispatch_pipeline(cmdlist, atomic_pipeline, atomic_bindings,
                          reduce_groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_grouped_reduce_atomic_sum_f32"
                                     : value_type == 2
                                     ? "vulkan_grouped_reduce_atomic_sum_u32"
                                     : value_type == 3
                                     ? "vulkan_grouped_reduce_atomic_sum_u64"
                                     : value_type == 4
                                     ? "vulkan_grouped_reduce_atomic_sum_i64"
                                     : value_type == 5
                                     ? "vulkan_grouped_reduce_atomic_sum_f64"
                                     : "vulkan_grouped_reduce_atomic_sum_i32")
                              : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(21);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(keys_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(keys_alloc));
  command_key.push(keys_ptr.offset);
  command_key.push(static_cast<uint64_t>(input_bytes));
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(values_ptr.offset);
  command_key.push(static_cast<uint64_t>(values_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(output_ptr.offset);
  command_key.push(static_cast<uint64_t>(output_bytes));
  command_key.push(zero_groups);
  command_key.push(reduce_groups);
  command_key.push_ptr(zero_pipeline);
  command_key.push_ptr(atomic_pipeline);
  command_key.push_ptr(zero_bindings);
  command_key.push_ptr(atomic_bindings);
  if (!cache.grouped_reduce_atomic_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_grouped_reduce_atomic)) {
    enqueue_compute_op_lambda(record_grouped_reduce_atomic, {});
  }
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_grouped_reduce_atomic_strided_ndarray(
      keys, values, output, value_type, offset, stride, 0,
      vulkan_scan_value_type_size(value_type), op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_grouped_reduce_atomic_strided_keys_ndarray(
      keys, values, output, value_type, 0, sizeof(int32_t), values_offset,
      values_stride, output_offset, output_stride, op);
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided grouped reduce is only available on "
              "Vulkan.");
  check_vulkan_grouped_reduce_strided_keys_request(
      keys, values, output, value_type, keys_offset, keys_stride,
      values_offset, values_stride, output_offset, output_stride, op);
  TI_ERROR_IF(!vulkan_grouped_reduce_atomic_value_type_available(value_type),
              "Vulkan native strided grouped reduce does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan strided grouped reduce value type.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided grouped reduce input is too large for "
              "u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(device == nullptr,
              "Vulkan native strided grouped reduce requires a compute "
              "device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_grouped_reduce_zero_strided_pipeline(value_type);
  cache.ensure_grouped_reduce_atomic_strided_pipeline(value_type);
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const size_t keys_bytes = n * keys->get_element_size();
  const size_t values_bytes = n * values->get_element_size();
  const size_t output_bytes = num_groups * output->get_element_size();
  const std::array<uint32_t, 3> zero_param_words{
      static_cast<uint32_t>(num_groups),
      static_cast<uint32_t>(output_offset / value_size),
      static_cast<uint32_t>(output_stride / value_size),
  };
  const std::array<uint32_t, 8> reduce_param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(keys_offset / sizeof(int32_t)),
      static_cast<uint32_t>(keys_stride / sizeof(int32_t)),
      static_cast<uint32_t>(values_offset / value_size),
      static_cast<uint32_t>(values_stride / value_size),
      static_cast<uint32_t>(num_groups),
      static_cast<uint32_t>(output_offset / value_size),
      static_cast<uint32_t>(output_stride / value_size),
  };
  const size_t zero_params_bytes = zero_param_words.size() * sizeof(uint32_t);
  const size_t reduce_params_bytes =
      reduce_param_words.size() * sizeof(uint32_t);
  Pipeline *zero_pipeline = cache.grouped_reduce_zero_strided_pipeline(value_type);
  Pipeline *atomic_pipeline =
      cache.grouped_reduce_atomic_strided_pipeline(value_type);
  TI_ERROR_IF(!zero_pipeline || !atomic_pipeline,
              "Vulkan native strided grouped reduce could not find a pipeline "
              "for the requested value type.");
  ShaderResourceSet *zero_bindings =
      cache
          .bind_grouped_reduce_zero_strided_resource_set(
              this, value_type, output_alloc, output_bytes)
          .bindings;
  ShaderResourceSet *atomic_bindings =
      cache
          .bind_grouped_reduce_atomic_strided_resource_set(
              this, value_type, keys_alloc, keys_bytes, values_alloc,
              values_bytes, output_alloc, output_bytes)
          .bindings;
  const uint32_t zero_groups =
      static_cast<uint32_t>((num_groups + kBlockSize - 1) / kBlockSize);
  const uint32_t reduce_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  auto record_grouped_reduce_atomic_strided =
      [keys_alloc, values_alloc, output_alloc, keys_bytes, values_bytes,
       output_bytes, zero_params_bytes, reduce_params_bytes,
       zero_param_words, reduce_param_words, zero_pipeline, atomic_pipeline,
       zero_bindings, atomic_bindings, zero_groups,
       reduce_groups, value_type, profiler_scopes](Device * /*op_device*/,
                                                   CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, zero_pipeline, zero_bindings, zero_param_words.data(),
            static_cast<uint32_t>(zero_params_bytes), zero_groups, 1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_grouped_reduce_zero_f32_strided"
                       : value_type == 2
                       ? "vulkan_grouped_reduce_zero_u32_strided"
                       : value_type == 3
                       ? "vulkan_grouped_reduce_zero_u64_strided"
                       : value_type == 4
                       ? "vulkan_grouped_reduce_zero_i64_strided"
                       : value_type == 5
                       ? "vulkan_grouped_reduce_zero_f64_strided"
                       : "vulkan_grouped_reduce_zero_i32_strided")
                : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        if (reduce_groups == 0) {
          return;
        }
        dispatch_pipeline_with_push_constants(
            cmdlist, atomic_pipeline, atomic_bindings, reduce_param_words.data(),
            static_cast<uint32_t>(reduce_params_bytes), reduce_groups, 1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_grouped_reduce_atomic_sum_f32_strided"
                       : value_type == 2
                       ? "vulkan_grouped_reduce_atomic_sum_u32_strided"
                       : value_type == 3
                       ? "vulkan_grouped_reduce_atomic_sum_u64_strided"
                       : value_type == 4
                       ? "vulkan_grouped_reduce_atomic_sum_i64_strided"
                       : value_type == 5
                       ? "vulkan_grouped_reduce_atomic_sum_f64_strided"
                       : "vulkan_grouped_reduce_atomic_sum_i32_strided")
                : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(22);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(keys_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(keys_alloc));
  command_key.push(static_cast<uint64_t>(keys_bytes));
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(static_cast<uint64_t>(values_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(static_cast<uint64_t>(output_bytes));
  command_key.push(static_cast<uint64_t>(zero_params_bytes));
  command_key.push(static_cast<uint64_t>(reduce_params_bytes));
  command_key.push(zero_groups);
  command_key.push(reduce_groups);
  for (uint32_t word : zero_param_words) {
    command_key.push(word);
  }
  for (uint32_t word : reduce_param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(zero_pipeline);
  command_key.push_ptr(atomic_pipeline);
  command_key.push_ptr(zero_bindings);
  command_key.push_ptr(atomic_bindings);
  if (!cache.grouped_reduce_atomic_strided_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_grouped_reduce_atomic_strided)) {
    enqueue_compute_op_lambda(record_grouped_reduce_atomic_strided, {});
  }
  return cache.partial_capacity;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scan is only available on Vulkan.");
  TI_ERROR_IF(!data, "Vulkan native scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native scan expects a 1D ndarray.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native scan received an unsupported value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(data->get_element_size() != value_size,
              "Vulkan native scan dtype does not match the requested value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scan requires a compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr);
}

std::size_t Program::vulkan_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                           int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native reverse scan is only available on Vulkan.");
  TI_ERROR_IF(!data, "Vulkan native reverse scan received null ndarray.");
  TI_ERROR_IF(data->shape.size() != 1,
              "Vulkan native reverse scan expects a 1D ndarray.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native reverse scan received an unsupported value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(data->get_element_size() != value_size,
              "Vulkan native reverse scan dtype does not match the requested "
              "value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native reverse scan requires a compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr, false, 0, 0, 0, true);
}

std::size_t Program::vulkan_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scan is only available on Vulkan.");
  check_vulkan_scan_member_request(data, value_type, offset, stride);
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native strided scan received an unsupported value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native strided scan requires a compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr, true, offset, stride);
}

std::size_t Program::vulkan_inclusive_reverse_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided reverse scan is only available on Vulkan.");
  check_vulkan_scan_member_request(data, value_type, offset, stride);
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native strided reverse scan received an unsupported "
              "value type.");

  const size_t n = data->get_nelement();
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native strided reverse scan requires a compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  return enqueue_vulkan_scan(this, cache, data->ndarray_alloc_, n, value_type,
                             profiler != nullptr, true, offset, stride, 0,
                             true);
}

std::size_t Program::vulkan_inclusive_scan_dense_field(SNode *data,
                                                       int value_type,
                                                       std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field scan is only available on Vulkan.");
  TI_ERROR_IF(!data, "Vulkan native dense field scan received null field.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native dense field scan received an unsupported value "
              "type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  DevicePtr ptr = get_dense_field_device_ptr(data);
  const size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "Vulkan native dense field scan source stride is smaller than "
              "value size.");
  TI_ERROR_IF(ptr.offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native dense field scan source offset/stride must align "
              "to value size.");
  TI_ERROR_IF(ptr.offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native dense field scan source offset/stride must be "
              "uint32-word aligned.");
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native dense field scan requires a compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  DeviceAllocation alloc{ptr.device, ptr.alloc_id};
  if (stride == value_size) {
    return enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                               profiler != nullptr, false, 0, value_size,
                               ptr.offset);
  }
  return enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                             profiler != nullptr, true, ptr.offset, stride);
}

std::size_t Program::vulkan_inclusive_reverse_scan_dense_field(SNode *data,
                                                               int value_type,
                                                               std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field reverse scan is only available on "
              "Vulkan.");
  TI_ERROR_IF(!data,
              "Vulkan native dense field reverse scan received null field.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native dense field reverse scan received an unsupported "
              "value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  DevicePtr ptr = get_dense_field_device_ptr(data);
  const size_t stride = get_dense_field_stride(data, value_size);
  TI_ERROR_IF(stride < value_size,
              "Vulkan native dense field reverse scan source stride is "
              "smaller than value size.");
  TI_ERROR_IF(ptr.offset % value_size != 0 || stride % value_size != 0,
              "Vulkan native dense field reverse scan source offset/stride "
              "must align to value size.");
  TI_ERROR_IF(ptr.offset % sizeof(uint32_t) != 0 ||
                  stride % sizeof(uint32_t) != 0,
              "Vulkan native dense field reverse scan source offset/stride "
              "must be uint32-word aligned.");
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field reverse scan requires a compute "
              "device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  DeviceAllocation alloc{ptr.device, ptr.alloc_id};
  if (stride == value_size) {
    return enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                               profiler != nullptr, false, 0, value_size,
                               ptr.offset, true);
  }
  return enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                             profiler != nullptr, true, ptr.offset, stride, 0,
                             true);
}

std::size_t Program::vulkan_inclusive_scan_dense_field_packed(SNode *data,
                                                              int value_type,
                                                              std::size_t n,
                                                              int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field scan is only available on "
              "Vulkan.");
  TI_ERROR_IF(!data,
              "Vulkan native packed dense field scan received null field.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native packed dense field scan received an unsupported "
              "value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  const size_t item_bytes =
      value_size * static_cast<size_t>(std::max(1, lane_count));
  TI_ERROR_IF(lane_count <= 0,
              "Vulkan native packed dense field scan received an invalid lane "
              "count.");
  DevicePtr ptr = get_dense_field_device_ptr(data);
  TI_ERROR_IF(get_dense_field_stride(data, value_size) != item_bytes,
              "Vulkan native packed dense field scan expects a packed "
              "contiguous MatrixField layout.");
  TI_ERROR_IF(ptr.offset % value_size != 0 || item_bytes % value_size != 0,
              "Vulkan native packed dense field scan source offset/stride "
              "must align to value size.");
  TI_ERROR_IF(ptr.offset % sizeof(uint32_t) != 0 ||
                  item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native packed dense field scan source offset/stride "
              "must be uint32-word aligned.");
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field scan requires a compute "
              "device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  DeviceAllocation alloc{ptr.device, ptr.alloc_id};
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    temp_bytes = std::max(
        temp_bytes,
        enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                            profiler != nullptr, true,
                            ptr.offset + static_cast<size_t>(lane) * value_size,
                            item_bytes));
  }
  return temp_bytes;
}

std::size_t Program::vulkan_inclusive_reverse_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field reverse scan is only "
              "available on Vulkan.");
  TI_ERROR_IF(!data,
              "Vulkan native packed dense field reverse scan received null "
              "field.");
  TI_ERROR_IF(!vulkan_scan_value_type_available(value_type),
              "Vulkan native packed dense field reverse scan received an "
              "unsupported value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  const size_t item_bytes =
      value_size * static_cast<size_t>(std::max(1, lane_count));
  TI_ERROR_IF(lane_count <= 0,
              "Vulkan native packed dense field reverse scan received an "
              "invalid lane count.");
  DevicePtr ptr = get_dense_field_device_ptr(data);
  TI_ERROR_IF(get_dense_field_stride(data, value_size) != item_bytes,
              "Vulkan native packed dense field reverse scan expects a packed "
              "contiguous MatrixField layout.");
  TI_ERROR_IF(ptr.offset % value_size != 0 || item_bytes % value_size != 0,
              "Vulkan native packed dense field reverse scan source "
              "offset/stride must align to value size.");
  TI_ERROR_IF(ptr.offset % sizeof(uint32_t) != 0 ||
                  item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native packed dense field reverse scan source "
              "offset/stride must be uint32-word aligned.");
  if (n <= 1) {
    return 0;
  }

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field reverse scan requires a "
              "compute device.");
  auto cache_lease = get_scan_cache(this, device);
  auto &cache = *cache_lease;
  DeviceAllocation alloc{ptr.device, ptr.alloc_id};
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    temp_bytes = std::max(
        temp_bytes,
        enqueue_vulkan_scan(this, cache, alloc, n, value_type,
                            profiler != nullptr, true,
                            ptr.offset + static_cast<size_t>(lane) * value_size,
                            item_bytes, 0, true));
  }
  return temp_bytes;
}

std::size_t Program::vulkan_compact_ndarray(Ndarray *values,
                                            Ndarray *flags,
                                            Ndarray *output,
                                            Ndarray *count,
                                            int value_type) {
  return vulkan_compact_ndarray_bounded(values, flags, output, count,
                                        value_type, nullptr, 0);
}

std::size_t Program::vulkan_compact_ndarray_bounded(
    Ndarray *values,
    Ndarray *flags,
    Ndarray *output,
    Ndarray *count,
    int value_type,
    Ndarray *dispatch_packet,
    int block_dim) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native compact is only available on Vulkan.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "Vulkan native compact received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || flags->shape.size() != 1 ||
                  output->shape.size() != 1 || count->shape.size() != 1,
              "Vulkan native compact expects 1D ndarrays.");
  TI_ERROR_IF(values->get_nelement() != flags->get_nelement(),
              "Vulkan native compact values and flags must have the same "
              "length.");
  TI_ERROR_IF(output->get_nelement() < values->get_nelement(),
              "Vulkan native compact output must have at least input length.");
  TI_ERROR_IF(count->get_nelement() < 1,
              "Vulkan native compact count must contain at least one item.");
  TI_ERROR_IF(dispatch_packet != nullptr &&
                  (dispatch_packet->shape.size() != 1 ||
                   dispatch_packet->get_nelement() < 4 ||
                   dispatch_packet->get_element_size() != sizeof(uint32_t)),
              "Vulkan bounded compact dispatch packet must contain at least "
              "four u32 words.");
  TI_ERROR_IF(dispatch_packet != nullptr &&
                  (block_dim <= 0 || block_dim > 1024),
              "Vulkan bounded compact block dimension must be in [1, 1024].");
  const size_t expected_value_bytes =
      (value_type == 0 || value_type == 1 || value_type == 2)
          ? sizeof(uint32_t)
          : (value_type == 3 || value_type == 4 || value_type == 5)
                ? sizeof(uint64_t)
                : 0;
  TI_ERROR_IF(expected_value_bytes == 0,
              "Vulkan native compact received an unsupported value type.");
  const size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  flags->get_element_size() != sizeof(int32_t) ||
                  count->get_element_size() != sizeof(int32_t),
              "Vulkan native compact received mismatched value/flag/count "
              "dtypes or a non-4-byte-aligned payload.");

  const size_t n = values->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native compact word count exceeds UINT32_MAX.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native compact requires a compute device.");
  auto cache_lease = get_compact_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_compact_pipelines(device);
  const size_t prefix_bytes = n * sizeof(int32_t);
  if (cache.has_workspace_allocs() &&
      cache.needs_prefix_realloc(prefix_bytes)) {
    synchronize();
  }
  cache.ensure_prefix(prefix_bytes);
  const size_t value_total_bytes = n * item_bytes;

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation flags_alloc = flags->ndarray_alloc_;
  DeviceAllocation output_alloc = output->ndarray_alloc_;
  DeviceAllocation count_alloc = count->ndarray_alloc_;
  DeviceAllocation packet_alloc =
      dispatch_packet ? dispatch_packet->ndarray_alloc_ : kDeviceNullAllocation;
  DeviceAllocation prefix_alloc = cache.prefix;
  Pipeline *flags_pipeline = cache.compact_i32_flags.get();
  Pipeline *scatter_pipeline =
      dispatch_packet ? cache.compact_i32_scatter_packet.get()
                      : cache.compact_i32_scatter.get();
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t flag_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t word_groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const int compact_fuse_max_n_config =
      get_environ_config("TI_VULKAN_COMPACT_FUSE_MAX_N", 4096);
  const bool use_fused_recording =
      compact_fuse_max_n_config > 0 &&
      n <= static_cast<size_t>(compact_fuse_max_n_config);
  ShaderResourceSet *flags_resource_set =
      cache
          .bind_flags_resource_set(this, flags_alloc, 0, prefix_alloc,
                                   prefix_bytes)
          .bindings;
  ShaderResourceSet *scatter_resource_set = nullptr;
  if (dispatch_packet) {
    scatter_resource_set =
        cache
            .bind_scatter_packet_resource_set(
                this, values_alloc, 0, value_total_bytes, flags_alloc, 0,
                prefix_bytes, prefix_alloc, output_alloc, 0, count_alloc, 0,
                packet_alloc, 0)
            .bindings;
  } else {
    scatter_resource_set =
        cache
            .bind_scatter_resource_set(this, values_alloc, 0,
                                       value_total_bytes, flags_alloc, 0,
                                       prefix_bytes, prefix_alloc, output_alloc,
                                       0, count_alloc, 0)
            .bindings;
  }

  if (use_fused_recording) {
    auto scan_plan = prepare_vulkan_i32_scan(this, cache.scan, prefix_alloc, n);
    cache.cached_bytes = cache.allocated_bytes();
    auto record_compact_fused =
        [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, flag_groups,
         values_alloc, output_alloc, count_alloc, scatter_pipeline, scan_plan,
         value_total_bytes, word_groups, flags_resource_set,
         scatter_resource_set, packet_alloc,
         profiler_scopes](Device *op_device, CommandList *cmdlist) {
          {
            ShaderResourceSet *bindings = flags_resource_set;
            dispatch_pipeline(cmdlist, flags_pipeline, bindings,
                              flag_groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_flags"
                                              : nullptr);
            cmdlist->buffer_barrier(prefix_alloc);
          }
          record_vulkan_i32_scan(op_device, cmdlist, scan_plan,
                                 profiler_scopes);
          {
            ShaderResourceSet *bindings = scatter_resource_set;
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings,
                              word_groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_scatter"
                                              : nullptr);
          }
          cmdlist->buffer_barrier(output_alloc);
          cmdlist->buffer_barrier(count_alloc);
          if (packet_alloc != kDeviceNullAllocation) {
            cmdlist->buffer_barrier(packet_alloc);
          }
        };
    VulkanCommandReplayKey command_key = make_vulkan_compact_fused_command_key(
        false, value_type, values_alloc, 0, value_total_bytes, flags_alloc, 0,
        output_alloc, 0, count_alloc, 0, prefix_alloc, prefix_bytes,
        flag_groups, word_groups, flags_pipeline, scatter_pipeline,
        flags_resource_set, scatter_resource_set, scan_plan);
    if (!cache.ndarray_fused_command_replay.submit_or_record(
            this, device, command_key, profiler_scopes,
            record_compact_fused)) {
      enqueue_compute_op_lambda(record_compact_fused, {});
    }
    return cache.cached_bytes;
  }

  auto record_compact_flags =
      [flags_alloc, prefix_alloc, prefix_bytes, flags_pipeline, flag_groups,
       flags_resource_set,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        ShaderResourceSet *bindings = flags_resource_set;
        dispatch_pipeline(cmdlist, flags_pipeline, bindings, flag_groups,
                          1, 1,
                          profiler_scopes ? "vulkan_compact_i32_flags"
                                          : nullptr);
        cmdlist->buffer_barrier(prefix_alloc);
      };
  VulkanCommandReplayKey flags_command_key;
  flags_command_key.push(70);
  push_vulkan_command_key_range(flags_command_key, flags_alloc, 0,
                                prefix_bytes);
  push_vulkan_command_key_range(flags_command_key, prefix_alloc, 0,
                                prefix_bytes);
  flags_command_key.push(flag_groups);
  flags_command_key.push_ptr(flags_pipeline);
  flags_command_key.push_ptr(flags_resource_set);
  if (!cache.ndarray_flags_command_replay.submit_or_record(
          this, device, flags_command_key, profiler_scopes,
          record_compact_flags)) {
    enqueue_compute_op_lambda(record_compact_flags, {});
  }

  enqueue_vulkan_i32_scan(this, cache.scan, prefix_alloc, n, profiler_scopes);
  cache.cached_bytes = cache.allocated_bytes();

  auto record_compact_scatter =
      [values_alloc, flags_alloc, prefix_alloc, output_alloc, count_alloc,
       prefix_bytes, value_total_bytes, scatter_pipeline, word_groups,
       scatter_resource_set, packet_alloc,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        ShaderResourceSet *bindings = scatter_resource_set;
        dispatch_pipeline(cmdlist, scatter_pipeline, bindings,
                          word_groups, 1, 1,
                          profiler_scopes ? "vulkan_compact_i32_scatter"
                                          : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        cmdlist->buffer_barrier(count_alloc);
        if (packet_alloc != kDeviceNullAllocation) {
          cmdlist->buffer_barrier(packet_alloc);
        }
      };
  VulkanCommandReplayKey scatter_command_key;
  scatter_command_key.push(71);
  scatter_command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(scatter_command_key, values_alloc, 0,
                                value_total_bytes);
  push_vulkan_command_key_range(scatter_command_key, flags_alloc, 0,
                                prefix_bytes);
  push_vulkan_command_key_range(scatter_command_key, prefix_alloc, 0,
                                prefix_bytes);
  push_vulkan_command_key_range(scatter_command_key, output_alloc, 0,
                                value_total_bytes);
  push_vulkan_command_key_range(scatter_command_key, count_alloc, 0,
                                sizeof(int32_t));
  if (packet_alloc != kDeviceNullAllocation) {
    push_vulkan_command_key_range(scatter_command_key, packet_alloc, 0,
                                  4 * sizeof(uint32_t));
  }
  scatter_command_key.push(word_groups);
  scatter_command_key.push_ptr(scatter_pipeline);
  scatter_command_key.push_ptr(scatter_resource_set);
  if (!cache.ndarray_scatter_command_replay.submit_or_record(
          this, device, scatter_command_key, profiler_scopes,
          record_compact_scatter)) {
    enqueue_compute_op_lambda(record_compact_scatter, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_compact_dense_field(SNode *values,
                                                SNode *flags,
                                                SNode *output,
                                                SNode *count,
                                                int value_type,
                                                std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field compact is only available on Vulkan.");
  TI_ERROR_IF(!values || !flags || !output || !count,
              "Vulkan native dense field compact received null field.");
  const size_t expected_value_bytes =
      (value_type == 0 || value_type == 1 || value_type == 2)
          ? sizeof(uint32_t)
          : (value_type == 3 || value_type == 4 || value_type == 5)
                ? sizeof(uint64_t)
                : 0;
  TI_ERROR_IF(expected_value_bytes == 0,
              "Vulkan native dense field compact received an unsupported "
              "value type.");
  const size_t item_bytes = expected_value_bytes;
  TI_ERROR_IF(item_bytes % sizeof(uint32_t) != 0,
              "Vulkan native dense field compact received a non-4-byte-"
              "aligned payload.");
  const size_t values_stride = get_dense_field_stride(values, item_bytes);
  const size_t flags_stride = get_dense_field_stride(flags, sizeof(int32_t));
  const size_t output_stride = get_dense_field_stride(output, item_bytes);
  const size_t count_stride = get_dense_field_stride(count, sizeof(int32_t));
  TI_ERROR_IF(values_stride != item_bytes || output_stride != item_bytes ||
                  flags_stride != sizeof(int32_t) ||
                  count_stride < sizeof(int32_t),
              "Vulkan native dense field compact requires contiguous values, "
              "flags, and output fields.");
  return vulkan_compact_ranges(
      get_dense_field_device_ptr(values), get_dense_field_device_ptr(flags),
      get_dense_field_device_ptr(output), get_dense_field_device_ptr(count),
      item_bytes, n, false);
}

std::size_t Program::vulkan_compact_ranges(DevicePtr values_ptr,
                                         DevicePtr flags_ptr,
                                         DevicePtr output_ptr,
                                         DevicePtr count_ptr,
                                         std::size_t item_bytes,
                                         std::size_t n,
                                         bool reuse_prefix) {
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field compact word count exceeds "
              "UINT32_MAX.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native dense field compact requires a compute device.");
  auto cache_lease = get_compact_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_compact_pipelines(device);
  const size_t prefix_bytes = n * sizeof(int32_t);
  if (cache.has_workspace_allocs() &&
      cache.needs_prefix_realloc(prefix_bytes)) {
    synchronize();
  }
  cache.ensure_prefix(prefix_bytes);
  const size_t value_total_bytes = n * item_bytes;

  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  DeviceAllocation flags_alloc{flags_ptr.device, flags_ptr.alloc_id};
  DeviceAllocation output_alloc{output_ptr.device, output_ptr.alloc_id};
  DeviceAllocation count_alloc{count_ptr.device, count_ptr.alloc_id};
  const size_t values_offset = values_ptr.offset;
  const size_t flags_offset = flags_ptr.offset;
  const size_t output_offset = output_ptr.offset;
  const size_t count_offset = count_ptr.offset;
  DeviceAllocation prefix_alloc = cache.prefix;
  Pipeline *flags_pipeline = cache.compact_i32_flags.get();
  Pipeline *scatter_pipeline = cache.compact_i32_scatter.get();
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t flag_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t word_groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const int compact_fuse_max_n_config =
      get_environ_config("TI_VULKAN_COMPACT_FUSE_MAX_N", 4096);
  const bool use_fused_recording =
      compact_fuse_max_n_config > 0 &&
      n <= static_cast<size_t>(compact_fuse_max_n_config);
  ShaderResourceSet *flags_resource_set =
      cache
          .bind_flags_resource_set(this, flags_alloc, flags_offset,
                                   prefix_alloc, prefix_bytes)
          .bindings;
  ShaderResourceSet *scatter_resource_set =
      cache
          .bind_scatter_resource_set(
              this, values_alloc, values_offset, value_total_bytes, flags_alloc,
              flags_offset, prefix_bytes, prefix_alloc, output_alloc,
              output_offset, count_alloc, count_offset)
          .bindings;

  if (use_fused_recording && !reuse_prefix) {
    auto scan_plan = prepare_vulkan_i32_scan(this, cache.scan, prefix_alloc, n);
    cache.cached_bytes = cache.allocated_bytes();
    auto record_compact_fused =
        [flags_alloc, flags_offset, prefix_alloc, prefix_bytes, flags_pipeline,
         flag_groups, values_alloc, values_offset, output_alloc, output_offset,
         count_alloc, count_offset, scatter_pipeline, scan_plan,
         value_total_bytes, word_groups, flags_resource_set,
         scatter_resource_set, profiler_scopes](Device *op_device,
                                                CommandList *cmdlist) {
          {
            ShaderResourceSet *bindings = flags_resource_set;
            dispatch_pipeline(cmdlist, flags_pipeline, bindings,
                              flag_groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_flags"
                                              : nullptr);
            cmdlist->buffer_barrier(prefix_alloc);
          }
          record_vulkan_i32_scan(op_device, cmdlist, scan_plan,
                                 profiler_scopes);
          {
            ShaderResourceSet *bindings = scatter_resource_set;
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings,
                              word_groups, 1, 1,
                              profiler_scopes ? "vulkan_compact_i32_scatter"
                                              : nullptr);
          }
          cmdlist->buffer_barrier(output_alloc);
          cmdlist->buffer_barrier(count_alloc);
        };
    VulkanCommandReplayKey command_key = make_vulkan_compact_fused_command_key(
        true, static_cast<int>(item_words), values_alloc, values_offset,
        value_total_bytes, flags_alloc, flags_offset, output_alloc, output_offset, count_alloc,
        count_offset, prefix_alloc, prefix_bytes, flag_groups, word_groups,
        flags_pipeline, scatter_pipeline, flags_resource_set,
        scatter_resource_set, scan_plan);
    if (!cache.dense_field_fused_command_replay.submit_or_record(
            this, device, command_key, profiler_scopes,
            record_compact_fused)) {
      enqueue_compute_op_lambda(record_compact_fused, {});
    }
    return cache.cached_bytes;
  }

  auto record_compact_flags =
      [flags_alloc, flags_offset, prefix_alloc, prefix_bytes, flags_pipeline,
       flag_groups, flags_resource_set,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        ShaderResourceSet *bindings = flags_resource_set;
        dispatch_pipeline(cmdlist, flags_pipeline, bindings, flag_groups,
                          1, 1,
                          profiler_scopes ? "vulkan_compact_i32_flags"
                                          : nullptr);
        cmdlist->buffer_barrier(prefix_alloc);
      };
  VulkanCommandReplayKey flags_command_key;
  flags_command_key.push(72);
  push_vulkan_command_key_range(flags_command_key, flags_alloc, flags_offset,
                                prefix_bytes);
  push_vulkan_command_key_range(flags_command_key, prefix_alloc, 0,
                                prefix_bytes);
  flags_command_key.push(flag_groups);
  flags_command_key.push_ptr(flags_pipeline);
  flags_command_key.push_ptr(flags_resource_set);
  if (!reuse_prefix &&
      !cache.dense_field_flags_command_replay.submit_or_record(
          this, device, flags_command_key, profiler_scopes, record_compact_flags)) {
    enqueue_compute_op_lambda(record_compact_flags, {});
  }

  if (!reuse_prefix) {
    enqueue_vulkan_i32_scan(this, cache.scan, prefix_alloc, n, profiler_scopes);
  }
  cache.cached_bytes = cache.allocated_bytes();

  auto record_compact_scatter =
      [values_alloc, values_offset, flags_alloc, flags_offset, prefix_alloc,
       output_alloc, output_offset, count_alloc, count_offset, prefix_bytes,
       value_total_bytes, scatter_pipeline, word_groups, scatter_resource_set,
       profiler_scopes](
           Device * /*op_device*/, CommandList *cmdlist) {
        ShaderResourceSet *bindings = scatter_resource_set;
        dispatch_pipeline(cmdlist, scatter_pipeline, bindings,
                          word_groups, 1, 1,
                          profiler_scopes ? "vulkan_compact_i32_scatter"
                                          : nullptr);
        cmdlist->buffer_barrier(output_alloc);
        cmdlist->buffer_barrier(count_alloc);
      };
  VulkanCommandReplayKey scatter_command_key;
  scatter_command_key.push(73);
  scatter_command_key.push(static_cast<uint64_t>(item_words));
  push_vulkan_command_key_range(scatter_command_key, values_alloc,
                                values_offset, value_total_bytes);
  push_vulkan_command_key_range(scatter_command_key, flags_alloc, flags_offset,
                                prefix_bytes);
  push_vulkan_command_key_range(scatter_command_key, prefix_alloc, 0,
                                prefix_bytes);
  push_vulkan_command_key_range(scatter_command_key, output_alloc,
                                output_offset, value_total_bytes);
  push_vulkan_command_key_range(scatter_command_key, count_alloc, count_offset,
                                sizeof(int32_t));
  scatter_command_key.push(word_groups);
  scatter_command_key.push_ptr(scatter_pipeline);
  scatter_command_key.push_ptr(scatter_resource_set);
  if (!cache.dense_field_scatter_command_replay.submit_or_record(
          this, device, scatter_command_key, profiler_scopes,
          record_compact_scatter)) {
    enqueue_compute_op_lambda(record_compact_scatter, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_compact_i32_ndarray(Ndarray *values,
                                                Ndarray *flags,
                                                Ndarray *output,
                                                Ndarray *count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_compact_ndarray(values, flags, output, count, 0);
}

std::size_t Program::vulkan_histogram_i32_ndarray(Ndarray *values,
                                                  Ndarray *bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_histogram_ndarray(values, bins, 0, 0);
}

std::size_t Program::vulkan_histogram_ndarray(Ndarray *values,
                                              Ndarray *bins,
                                              int value_type,
                                              int bin_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native histogram is only available on Vulkan.");
  TI_ERROR_IF(!values || !bins,
              "Vulkan native histogram received null ndarray.");
  TI_ERROR_IF(values->shape.size() != 1 || bins->shape.size() != 1,
              "Vulkan native histogram expects 1D ndarrays.");
  TI_ERROR_IF(!vulkan_histogram_value_type_available(value_type, bin_type),
              "Vulkan native histogram received an unsupported value/bin type.");
  const size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                            : sizeof(int32_t);
  const size_t bin_size = bin_type == 4 ? sizeof(int64_t) : sizeof(int32_t);
  TI_ERROR_IF(values->get_element_size() != value_size ||
                  bins->get_element_size() != bin_size,
              "Vulkan native histogram received mismatched value/bin dtypes.");
  TI_ERROR_IF(bins->get_nelement() == 0,
              "Vulkan native histogram expects at least one bin.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native histogram requires a compute device.");
  auto cache_lease = get_histogram_cache(this, device);
  auto &cache = *cache_lease;

  const size_t n = values->get_nelement();
  const size_t num_bins = bins->get_nelement();
  const size_t value_bytes = n * value_size;
  const size_t bin_bytes = num_bins * bin_size;
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N", 65536);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS", 512);
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N", 4096);
  const bool shared_bins_supported = bin_type == 0 && num_bins <= 4096;
  const bool private_shared_supported = num_bins <= 4096;
  const bool use_single_shared =
      n > 0 && shared_bins_supported && single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);
  const bool use_private =
      n > 0 && !use_single_shared &&
      (private_min_n_config <= 0 ||
       n >= static_cast<size_t>(private_min_n_config)) &&
      (private_max_bins_config <= 0 ||
       num_bins <= static_cast<size_t>(private_max_bins_config));

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (use_private) {
    num_chunks = (n + kHistogramPrivateChunkSize - 1) /
                 kHistogramPrivateChunkSize;
    partial_bytes = num_chunks * num_bins * bin_size;
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }

  DeviceAllocation values_alloc = values->ndarray_alloc_;
  DeviceAllocation bins_alloc = bins->ndarray_alloc_;
  DeviceAllocation partial_alloc = cache.partial;
  Pipeline *clear_pipeline = nullptr;
  Pipeline *count_direct_pipeline = nullptr;
  Pipeline *count_private_pipeline = nullptr;
  Pipeline *count_private_shared_pipeline = nullptr;
  Pipeline *reduce_private_pipeline = nullptr;
  Pipeline *single_shared_pipeline = nullptr;
  const char *single_shared_scope = nullptr;
  const char *clear_scope = nullptr;
  const char *reduce_private_scope = nullptr;
  const char *count_direct_scope = nullptr;
  const char *count_private_scope = nullptr;
  const char *count_private_shared_scope = nullptr;
  if (use_single_shared) {
    single_shared_pipeline = cache.single_shared_pipeline(value_type);
    single_shared_scope = cache.single_shared_scope(value_type);
  } else if (use_private) {
    count_private_shared_pipeline =
        private_shared_supported
            ? cache.count_private_shared_pipeline(value_type, bin_type)
            : nullptr;
    if (!count_private_shared_pipeline) {
      clear_pipeline = cache.clear_pipeline(bin_type);
      clear_scope = cache.clear_scope(bin_type);
      count_private_pipeline = cache.count_private_pipeline(value_type, bin_type);
      count_private_scope = cache.count_private_scope(value_type, bin_type, false);
    }
    reduce_private_pipeline = cache.reduce_private_pipeline(bin_type);
    reduce_private_scope = cache.reduce_private_scope(bin_type);
    count_private_shared_scope =
        cache.count_private_scope(value_type, bin_type, true);
  } else {
    clear_pipeline = cache.clear_pipeline(bin_type);
    count_direct_pipeline = cache.count_direct_pipeline(value_type, bin_type);
    clear_scope = cache.clear_scope(bin_type);
    count_direct_scope = cache.count_direct_scope(value_type, bin_type);
  }
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t bin_groups = static_cast<uint32_t>(
      (num_bins + kBlockSize - 1) / kBlockSize);
  const uint32_t partial_groups = static_cast<uint32_t>(
      ((use_private ? num_chunks * num_bins : 0) + kBlockSize - 1) /
      kBlockSize);
  const uint32_t value_groups = static_cast<uint32_t>(
      (n + kBlockSize - 1) / kBlockSize);
  ShaderResourceSet *single_shared_bindings = nullptr;
  ShaderResourceSet *clear_bindings = nullptr;
  ShaderResourceSet *count_direct_bindings = nullptr;
  ShaderResourceSet *count_private_bindings = nullptr;
  ShaderResourceSet *reduce_private_bindings = nullptr;
  if (use_single_shared) {
    single_shared_bindings =
        cache
            .bind_single_shared_resource_set(this, values_alloc, 0,
                                             value_bytes, bins_alloc, 0,
                                             bin_bytes)
            .bindings;
  } else if (use_private) {
    if (!count_private_shared_pipeline) {
      clear_bindings =
          cache.bind_clear_resource_set(this, partial_alloc, 0, partial_bytes)
              .bindings;
    }
    count_private_bindings =
        cache
            .bind_count_private_resource_set(
                this, count_private_shared_pipeline != nullptr, values_alloc,
                0, value_bytes, bins_alloc, 0, bin_bytes, partial_alloc, 0,
                partial_bytes)
            .bindings;
    reduce_private_bindings =
        cache
            .bind_reduce_private_resource_set(this, partial_alloc, 0,
                                              partial_bytes, bins_alloc, 0,
                                              bin_bytes)
            .bindings;
  } else {
    clear_bindings =
        cache.bind_clear_resource_set(this, bins_alloc, 0, bin_bytes).bindings;
    if (value_groups > 0) {
      count_direct_bindings =
          cache
              .bind_count_direct_resource_set(this, values_alloc, 0,
                                              value_bytes, bins_alloc, 0,
                                              bin_bytes)
              .bindings;
    }
  }

  auto record_histogram =
      [values_alloc, bins_alloc, partial_alloc, value_bytes, bin_bytes,
       partial_bytes, clear_pipeline, count_direct_pipeline,
       count_private_pipeline, count_private_shared_pipeline,
       reduce_private_pipeline, single_shared_pipeline, single_shared_scope,
       clear_scope, reduce_private_scope, count_direct_scope,
       count_private_scope, count_private_shared_scope, value_groups,
       bin_groups, partial_groups, num_chunks, use_private, use_single_shared,
       profiler_scopes, single_shared_bindings, clear_bindings,
       count_direct_bindings, count_private_bindings,
       reduce_private_bindings](Device * /*op_device*/, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        if (use_single_shared) {
          dispatch_pipeline(cmdlist, single_shared_pipeline,
                            single_shared_bindings, 1, 1, 1,
                            scope_name(single_shared_scope));
          cmdlist->buffer_barrier(bins_alloc);
          return;
        }
        if (use_private) {
          if (!count_private_shared_pipeline) {
            dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings,
                              partial_groups, 1, 1, scope_name(clear_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            Pipeline *count_pipeline = count_private_shared_pipeline
                                           ? count_private_shared_pipeline
                                           : count_private_pipeline;
            const char *count_scope =
                count_private_shared_pipeline ? count_private_shared_scope
                                              : count_private_scope;
            uint32_t count_groups =
                count_private_shared_pipeline
                    ? static_cast<uint32_t>(num_chunks)
                    : value_groups;
            dispatch_pipeline(cmdlist, count_pipeline, count_private_bindings,
                              count_groups, 1, 1, scope_name(count_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            dispatch_pipeline(cmdlist, reduce_private_pipeline,
                              reduce_private_bindings, bin_groups, 1, 1,
                              scope_name(reduce_private_scope));
            cmdlist->buffer_barrier(bins_alloc);
          }
          return;
        }
        {
          dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings, bin_groups,
                            1, 1, scope_name(clear_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
        if (value_groups > 0) {
          dispatch_pipeline(cmdlist, count_direct_pipeline,
                            count_direct_bindings, value_groups, 1, 1,
                            scope_name(count_direct_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
      };
  VulkanCommandReplayKey command_key = make_vulkan_histogram_command_key(
      false, value_type, bin_type, use_single_shared, use_private,
      count_private_shared_pipeline != nullptr, values_alloc, 0, value_bytes,
      bins_alloc, 0, bin_bytes, partial_alloc, partial_bytes, value_groups,
      bin_groups, partial_groups, num_chunks, clear_pipeline,
      count_direct_pipeline, count_private_pipeline,
      count_private_shared_pipeline, reduce_private_pipeline,
      single_shared_pipeline, single_shared_bindings, clear_bindings,
      count_direct_bindings, count_private_bindings, reduce_private_bindings);
  if (!cache.ndarray_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_histogram)) {
    enqueue_compute_op_lambda(record_histogram, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_histogram_dense_field(SNode *values,
                                                  SNode *bins,
                                                  int value_type,
                                                  int bin_type,
                                                  std::size_t n,
                                                  std::size_t num_bins) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field histogram is only available on "
              "Vulkan.");
  TI_ERROR_IF(!values || !bins,
              "Vulkan native dense field histogram received null field.");
  TI_ERROR_IF(!vulkan_histogram_value_type_available(value_type, bin_type),
              "Vulkan native dense field histogram received an unsupported "
              "value/bin type.");
  const size_t value_size = value_type == 2 ? sizeof(uint32_t)
                                            : sizeof(int32_t);
  const size_t bin_size = bin_type == 4 ? sizeof(int64_t) : sizeof(int32_t);
  TI_ERROR_IF(num_bins == 0,
              "Vulkan native dense field histogram expects at least one bin.");
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr bins_ptr = get_dense_field_device_ptr(bins);
  const size_t value_stride = get_dense_field_stride(values, value_size);
  const size_t bin_stride = get_dense_field_stride(bins, bin_size);
  TI_ERROR_IF(value_stride != value_size || bin_stride != bin_size,
              "Vulkan native dense field histogram requires contiguous dense "
              "field values and bins.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field histogram requires a compute device.");
  auto cache_lease = get_histogram_cache(this, device);
  auto &cache = *cache_lease;

  const size_t value_bytes = n * value_size;
  const size_t bin_bytes = num_bins * bin_size;
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N", 65536);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS", 512);
  const int single_shared_max_n_config =
      get_environ_config("TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N", 4096);
  const bool shared_bins_supported = bin_type == 0 && num_bins <= 4096;
  const bool private_shared_supported = num_bins <= 4096;
  const bool use_single_shared =
      n > 0 && shared_bins_supported && single_shared_max_n_config > 0 &&
      n <= static_cast<size_t>(single_shared_max_n_config);
  const bool use_private =
      n > 0 && !use_single_shared &&
      (private_min_n_config <= 0 ||
       n >= static_cast<size_t>(private_min_n_config)) &&
      (private_max_bins_config <= 0 ||
       num_bins <= static_cast<size_t>(private_max_bins_config));

  size_t num_chunks = 0;
  size_t partial_bytes = 0;
  if (use_private) {
    num_chunks = (n + kHistogramPrivateChunkSize - 1) /
                 kHistogramPrivateChunkSize;
    partial_bytes = num_chunks * num_bins * bin_size;
    if (cache.has_workspace_allocs() &&
        cache.needs_partial_realloc(partial_bytes)) {
      synchronize();
    }
    cache.ensure_partial(partial_bytes);
  }

  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  DeviceAllocation bins_alloc{bins_ptr.device, bins_ptr.alloc_id};
  DeviceAllocation partial_alloc = cache.partial;
  Pipeline *clear_pipeline = nullptr;
  Pipeline *count_direct_pipeline = nullptr;
  Pipeline *count_private_pipeline = nullptr;
  Pipeline *count_private_shared_pipeline = nullptr;
  Pipeline *reduce_private_pipeline = nullptr;
  Pipeline *single_shared_pipeline = nullptr;
  const char *single_shared_scope = nullptr;
  const char *clear_scope = nullptr;
  const char *reduce_private_scope = nullptr;
  const char *count_direct_scope = nullptr;
  const char *count_private_scope = nullptr;
  const char *count_private_shared_scope = nullptr;
  if (use_single_shared) {
    single_shared_pipeline = cache.single_shared_pipeline(value_type);
    single_shared_scope = cache.single_shared_scope(value_type);
  } else if (use_private) {
    count_private_shared_pipeline =
        private_shared_supported
            ? cache.count_private_shared_pipeline(value_type, bin_type)
            : nullptr;
    if (!count_private_shared_pipeline) {
      clear_pipeline = cache.clear_pipeline(bin_type);
      clear_scope = cache.clear_scope(bin_type);
      count_private_pipeline = cache.count_private_pipeline(value_type, bin_type);
      count_private_scope = cache.count_private_scope(value_type, bin_type, false);
    }
    reduce_private_pipeline = cache.reduce_private_pipeline(bin_type);
    reduce_private_scope = cache.reduce_private_scope(bin_type);
    count_private_shared_scope =
        cache.count_private_scope(value_type, bin_type, true);
  } else {
    clear_pipeline = cache.clear_pipeline(bin_type);
    count_direct_pipeline = cache.count_direct_pipeline(value_type, bin_type);
    clear_scope = cache.clear_scope(bin_type);
    count_direct_scope = cache.count_direct_scope(value_type, bin_type);
  }
  const bool profiler_scopes = profiler != nullptr;
  const uint32_t bin_groups = static_cast<uint32_t>(
      (num_bins + kBlockSize - 1) / kBlockSize);
  const uint32_t partial_groups = static_cast<uint32_t>(
      ((use_private ? num_chunks * num_bins : 0) + kBlockSize - 1) /
      kBlockSize);
  const uint32_t value_groups = static_cast<uint32_t>(
      (n + kBlockSize - 1) / kBlockSize);
  ShaderResourceSet *single_shared_bindings = nullptr;
  ShaderResourceSet *clear_bindings = nullptr;
  ShaderResourceSet *count_direct_bindings = nullptr;
  ShaderResourceSet *count_private_bindings = nullptr;
  ShaderResourceSet *reduce_private_bindings = nullptr;
  if (use_single_shared) {
    single_shared_bindings =
        cache
            .bind_single_shared_resource_set(this, values_alloc,
                                             values_ptr.offset, value_bytes,
                                             bins_alloc, bins_ptr.offset,
                                             bin_bytes)
            .bindings;
  } else if (use_private) {
    if (!count_private_shared_pipeline) {
      clear_bindings =
          cache.bind_clear_resource_set(this, partial_alloc, 0, partial_bytes)
              .bindings;
    }
    count_private_bindings =
        cache
            .bind_count_private_resource_set(
                this, count_private_shared_pipeline != nullptr, values_alloc,
                values_ptr.offset, value_bytes, bins_alloc, bins_ptr.offset,
                bin_bytes, partial_alloc, 0, partial_bytes)
            .bindings;
    reduce_private_bindings =
        cache
            .bind_reduce_private_resource_set(this, partial_alloc, 0,
                                              partial_bytes, bins_alloc,
                                              bins_ptr.offset, bin_bytes)
            .bindings;
  } else {
    clear_bindings =
        cache
            .bind_clear_resource_set(this, bins_alloc, bins_ptr.offset,
                                     bin_bytes)
            .bindings;
    if (value_groups > 0) {
      count_direct_bindings =
          cache
              .bind_count_direct_resource_set(
                  this, values_alloc, values_ptr.offset, value_bytes,
                  bins_alloc, bins_ptr.offset, bin_bytes)
              .bindings;
    }
  }

  auto record_histogram =
      [values_alloc, bins_alloc, partial_alloc, values_ptr, bins_ptr,
       value_bytes, bin_bytes, partial_bytes, clear_pipeline,
       count_direct_pipeline, count_private_pipeline,
       count_private_shared_pipeline, reduce_private_pipeline,
       single_shared_pipeline, single_shared_scope, clear_scope,
       reduce_private_scope, count_direct_scope, count_private_scope,
       count_private_shared_scope, value_groups, bin_groups, partial_groups,
       num_chunks, use_private, use_single_shared, profiler_scopes,
       single_shared_bindings, clear_bindings, count_direct_bindings,
       count_private_bindings, reduce_private_bindings](
          Device * /*op_device*/, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        if (use_single_shared) {
          dispatch_pipeline(cmdlist, single_shared_pipeline,
                            single_shared_bindings, 1, 1, 1,
                            scope_name(single_shared_scope));
          cmdlist->buffer_barrier(bins_alloc);
          return;
        }
        if (use_private) {
          if (!count_private_shared_pipeline) {
            dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings,
                              partial_groups, 1, 1, scope_name(clear_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            Pipeline *count_pipeline = count_private_shared_pipeline
                                           ? count_private_shared_pipeline
                                           : count_private_pipeline;
            const char *count_scope =
                count_private_shared_pipeline ? count_private_shared_scope
                                              : count_private_scope;
            uint32_t count_groups =
                count_private_shared_pipeline
                    ? static_cast<uint32_t>(num_chunks)
                    : value_groups;
            dispatch_pipeline(cmdlist, count_pipeline, count_private_bindings,
                              count_groups, 1, 1, scope_name(count_scope));
            cmdlist->buffer_barrier(partial_alloc);
          }
          {
            dispatch_pipeline(cmdlist, reduce_private_pipeline,
                              reduce_private_bindings, bin_groups, 1, 1,
                              scope_name(reduce_private_scope));
            cmdlist->buffer_barrier(bins_alloc);
          }
          return;
        }
        {
          dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings, bin_groups,
                            1, 1, scope_name(clear_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
        if (value_groups > 0) {
          dispatch_pipeline(cmdlist, count_direct_pipeline,
                            count_direct_bindings, value_groups, 1, 1,
                            scope_name(count_direct_scope));
          cmdlist->buffer_barrier(bins_alloc);
        }
      };
  VulkanCommandReplayKey command_key = make_vulkan_histogram_command_key(
      true, value_type, bin_type, use_single_shared, use_private,
      count_private_shared_pipeline != nullptr, values_alloc, values_ptr.offset,
      value_bytes, bins_alloc, bins_ptr.offset, bin_bytes, partial_alloc,
      partial_bytes, value_groups, bin_groups, partial_groups, num_chunks,
      clear_pipeline, count_direct_pipeline, count_private_pipeline,
      count_private_shared_pipeline, reduce_private_pipeline,
      single_shared_pipeline, single_shared_bindings, clear_bindings,
      count_direct_bindings, count_private_bindings, reduce_private_bindings);
  if (!cache.dense_field_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_histogram)) {
    enqueue_compute_op_lambda(record_histogram, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_reduce_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op, 0,
                                    vulkan_transform_value_size(value_type), 0,
                                    vulkan_transform_value_size(value_type),
                                    false, false);
}

std::size_t Program::vulkan_reduce_member_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  std::size_t offset,
                                                  std::size_t stride,
                                                  int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op,
                                    offset, stride, 0,
                                    vulkan_transform_value_size(value_type),
                                    true, false);
}

std::size_t Program::vulkan_reduce_i32_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_reduce_ndarray(values, output, 0, op);
}

std::size_t Program::vulkan_check_count_ndarray(Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                int check_op,
                                                int lower,
                                                int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native check_count is only available on Vulkan.");
  return vulkan_check_count_ndarray_impl(this, values, output, value_type,
                                         check_op, lower, upper, 0,
                                         vulkan_transform_value_size(value_type),
                                         false);
}

std::size_t Program::vulkan_check_count_strided_ndarray(Ndarray *values,
                                                        Ndarray *output,
                                                        int value_type,
                                                        std::size_t offset,
                                                        std::size_t stride,
                                                        int check_op,
                                                        int lower,
                                                        int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided check_count is only available on "
              "Vulkan.");
  return vulkan_check_count_ndarray_impl(this, values, output, value_type,
                                         check_op, lower, upper, offset,
                                         stride, true);
}

std::size_t Program::vulkan_check_count_dense_field(SNode *values,
                                                    Ndarray *output,
                                                    int value_type,
                                                    std::size_t n,
                                                    int check_op,
                                                    int lower,
                                                    int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field check_count is only available on "
              "Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native dense field check_count received a null "
              "argument.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1 ||
                  output->get_element_size() != sizeof(int32_t),
              "Vulkan native dense field check_count output must be a "
              "non-empty i32 ndarray.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native dense field check_count received an unsupported "
              "value type.");
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  const size_t stride = get_dense_field_stride(values, value_size);
  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  return vulkan_check_count_storage_impl(
      this, values_alloc, output->ndarray_alloc_, n, value_size, value_type,
      check_op, lower, upper, values_ptr.offset, stride);
}

std::size_t Program::vulkan_metric_reduce_ndarray(Ndarray *values,
                                                  Ndarray *other,
                                                  Ndarray *output,
                                                  int value_type,
                                                  int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native metric_reduce is only available on Vulkan.");
  return vulkan_metric_reduce_ndarray_impl(this, values, other, output,
                                           value_type, metric_op, 0,
                                           vulkan_transform_value_size(value_type),
                                           0,
                                           vulkan_transform_value_size(value_type),
                                           false);
}

std::size_t Program::vulkan_metric_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *other,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t other_offset,
    std::size_t other_stride,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided metric_reduce is only available on "
              "Vulkan.");
  return vulkan_metric_reduce_ndarray_impl(
      this, values, other, output, value_type, metric_op, values_offset,
      values_stride, other_offset, other_stride, true);
}

std::size_t Program::vulkan_metric_reduce_dense_field(SNode *values,
                                                      SNode *other,
                                                      Ndarray *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field metric_reduce is only available on "
              "Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native dense field metric_reduce received a null "
              "argument.");
  if (!other) {
    other = values;
  }
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native dense field metric_reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(output->shape.size() != 1 || output->get_nelement() < 1 ||
                  output->get_element_size() != value_size,
              "Vulkan native dense field metric_reduce output must be a "
              "non-empty ndarray matching value type.");
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr other_ptr = get_dense_field_device_ptr(other);
  const size_t values_stride = get_dense_field_stride(values, value_size);
  const size_t other_stride = get_dense_field_stride(other, value_size);
  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  DeviceAllocation other_alloc{other_ptr.device, other_ptr.alloc_id};
  return vulkan_metric_reduce_storage_impl(
      this, values_alloc, other_alloc, output->ndarray_alloc_, n, value_type,
      metric_op, values_ptr.offset, values_stride, other_ptr.offset,
      other_stride);
}

std::size_t Program::vulkan_metric_reduce_dense_field_strided_ndarray(
    SNode *field,
    Ndarray *array,
    Ndarray *output,
    int value_type,
    std::size_t n,
    std::size_t array_offset,
    std::size_t array_stride,
    bool field_is_values,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native mixed metric_reduce is only available on "
              "Vulkan.");
  TI_ERROR_IF(!field || !array || !output,
              "Vulkan native mixed metric_reduce received a null argument.");
  TI_ERROR_IF(array->shape.size() != 1 || output->shape.size() != 1,
              "Vulkan native mixed metric_reduce expects 1D ndarrays.");
  TI_ERROR_IF(n == 0,
              "Vulkan native mixed metric_reduce expects at least one input "
              "item.");
  TI_ERROR_IF(array->get_nelement() != n,
              "Vulkan native mixed metric_reduce inputs must have the same "
              "length.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native mixed metric_reduce received an unsupported "
              "value type.");
  TI_ERROR_IF(output->get_nelement() < 1 ||
                  output->get_element_size() != value_size,
              "Vulkan native mixed metric_reduce output must be a non-empty "
              "ndarray matching value type.");
  check_vulkan_strided_range("Vulkan native mixed metric_reduce", "ndarray",
                             array, n, value_size, array_offset,
                             array_stride);
  DevicePtr field_ptr = get_dense_field_device_ptr(field);
  const size_t field_stride = get_dense_field_stride(field, value_size);
  DeviceAllocation field_alloc{field_ptr.device, field_ptr.alloc_id};
  if (field_is_values) {
    return vulkan_metric_reduce_storage_impl(
        this, field_alloc, array->ndarray_alloc_, output->ndarray_alloc_, n,
        value_type, metric_op, field_ptr.offset, field_stride, array_offset,
        array_stride);
  }
  return vulkan_metric_reduce_storage_impl(
      this, array->ndarray_alloc_, field_alloc, output->ndarray_alloc_, n,
      value_type, metric_op, array_offset, array_stride, field_ptr.offset,
      field_stride);
}

std::size_t Program::vulkan_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_reduce_ndarray_impl(this, values, output, value_type, op,
                                    values_offset, values_stride,
                                    output_offset, output_stride, true, true);
}

std::size_t Program::vulkan_reduce_dense_field(SNode *values,
                                               SNode *output,
                                               int value_type,
                                               std::size_t n,
                                               int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field reduce is only available on Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native dense field reduce received null field.");
  TI_ERROR_IF(n == 0,
              "Vulkan native dense field reduce expects at least one input "
              "item.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr output_ptr = get_dense_field_device_ptr(output);
  const size_t values_stride = get_dense_field_stride(values, value_size);
  const size_t output_stride = get_dense_field_stride(output, value_size);
  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  DeviceAllocation output_alloc{output_ptr.device, output_ptr.alloc_id};
  const bool contiguous =
      values_stride == value_size && output_stride == value_size;
  return vulkan_reduce_storage_impl(
      this, values_alloc, output_alloc, n, value_size, value_size, value_type,
      op, values_ptr.offset, values_stride, output_ptr.offset, output_stride,
      !contiguous, !contiguous);
}

std::size_t Program::vulkan_reduce_dense_field_packed(SNode *values,
                                                      SNode *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int lane_count,
                                                      int op) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field reduce is only available on "
              "Vulkan.");
  TI_ERROR_IF(!values || !output,
              "Vulkan native packed dense field reduce received null field.");
  TI_ERROR_IF(n == 0,
              "Vulkan native packed dense field reduce expects at least one "
              "input item.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field reduce received an invalid "
              "value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr output_ptr = get_dense_field_device_ptr(output);
  TI_ERROR_IF(get_dense_field_stride(values, value_size) != item_bytes ||
                  get_dense_field_stride(output, value_size) != item_bytes,
              "Vulkan native packed dense field reduce expects a packed "
              "contiguous MatrixField layout.");
  DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  DeviceAllocation output_alloc{output_ptr.device, output_ptr.alloc_id};
  std::size_t temp_bytes = 0;
  for (int lane = 0; lane < lane_count; ++lane) {
    const size_t lane_offset = static_cast<size_t>(lane) * value_size;
    temp_bytes = std::max(
        temp_bytes,
        vulkan_reduce_storage_impl(
            this, values_alloc, output_alloc, n, value_size, value_size,
            value_type, op, values_ptr.offset + lane_offset, item_bytes,
            output_ptr.offset + lane_offset, value_size, true, true));
  }
  return temp_bytes;
}

std::size_t Program::vulkan_transform_affine_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     double scale,
                                                     double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, 0,
      vulkan_transform_value_size(value_type), 0,
      vulkan_transform_value_size(value_type), scale, bias, false, false);
}

std::size_t Program::vulkan_transform_affine_ndarray_trusted(Ndarray *src,
                                                             Ndarray *dst,
                                                             int value_type,
                                                             double scale,
                                                             double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  const size_t value_size = vulkan_transform_value_size(value_type);
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, 0, value_size, 0, value_size, scale, bias,
      false, false, true);
}

std::size_t Program::vulkan_transform_indexed_affine_ndarray(Ndarray *src,
                                                              Ndarray *indices,
                                                              Ndarray *dst,
                                                              int value_type,
                                                              double scale,
                                                              double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_indexed_affine_ndarray_impl(
      this, src, indices, dst, value_type, scale, bias);
}

std::size_t Program::vulkan_transform_affine_member_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, offset, stride, 0,
      vulkan_transform_value_size(value_type), scale, bias, true, false);
}

std::size_t Program::vulkan_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, src_offset, src_stride, dst_offset,
      dst_stride, scale, bias, true, true);
}

std::size_t Program::vulkan_transform_affine_strided_ndarray_trusted(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, 1, src_offset, src_stride, dst_offset,
      dst_stride, scale, bias, true, true, true);
}

std::size_t Program::vulkan_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_transform_affine_ndarray_impl(
      this, src, dst, value_type, lane_count, src_offset, src_stride,
      dst_offset, dst_stride, scale, bias, true, true);
}

std::size_t Program::vulkan_transform_affine_dense_field(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         double scale,
                                                         double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field transform is only available on "
              "Vulkan.");
  TI_ERROR_IF(!src || !dst,
              "Vulkan native dense field transform received null field.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, value_size);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const bool contiguous = src_stride == value_size && dst_stride == value_size;
  return vulkan_transform_affine_storage_impl(
      this, src_alloc, dst_alloc, n, value_size, value_size, value_type, 1,
      src_ptr.offset, src_stride, dst_ptr.offset, dst_stride, scale, bias,
      !contiguous, !contiguous);
}

std::size_t Program::vulkan_transform_affine_dense_field_trusted(SNode *src,
                                                                 SNode *dst,
                                                                 int value_type,
                                                                 std::size_t n,
                                                                 double scale,
                                                                 double bias) {
  const size_t value_size = vulkan_transform_value_size(value_type);
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, value_size);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const bool contiguous = src_stride == value_size && dst_stride == value_size;
  return vulkan_transform_affine_storage_impl(
      this, src_alloc, dst_alloc, n, value_size, value_size, value_type, 1,
      src_ptr.offset, src_stride, dst_ptr.offset, dst_stride, scale, bias,
      !contiguous, !contiguous, true);
}

std::size_t Program::vulkan_transform_affine_dense_field_packed(
    SNode *src,
    SNode *dst,
    int value_type,
    std::size_t n,
    int lane_count,
    double scale,
    double bias) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field transform is only available "
              "on Vulkan.");
  TI_ERROR_IF(!src || !dst,
              "Vulkan native packed dense field transform received null "
              "field.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field transform received an "
              "invalid value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field transform expects a packed "
              "contiguous MatrixField layout.");
  DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  return vulkan_transform_affine_storage_impl(
      this, src_alloc, dst_alloc, n, item_bytes, item_bytes, value_type,
      lane_count, src_ptr.offset, item_bytes, dst_ptr.offset, item_bytes, scale,
      bias, true, true);
}

std::size_t Program::vulkan_zero_dense_field(SNode *dst,
                                             int value_type,
                                             std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field zero-fill is only available on "
              "Vulkan.");
  TI_ERROR_IF(!dst, "Vulkan native dense field zero-fill received null field.");
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Vulkan native dense field zero-fill received an unsupported "
              "value type.");
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(dst_stride < value_size,
              "Vulkan native dense field zero-fill received an invalid field "
              "stride.");
  if (n == 0) {
    return 0;
  }
  DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  TI_ERROR_IF(!dst_alloc.device,
              "Vulkan native dense field zero-fill received null storage.");
  if (dst_stride == value_size) {
    Device *device = program_impl_->get_compute_device();
    TI_ERROR_IF(!device,
                "Vulkan native dense field zero-fill requires a compute "
                "device.");
    const size_t dst_bytes = n * value_size;
    auto record_zero = [dst_alloc, dst_offset = dst_ptr.offset, dst_bytes](
                           Device * /*op_device*/, CommandList *cmdlist) {
      cmdlist->buffer_fill(dst_alloc.get_ptr(dst_offset), dst_bytes, 0);
      cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_offset), dst_bytes);
    };
    enqueue_compute_op_lambda(record_zero, {});
    return 0;
  }
  return vulkan_transform_affine_storage_impl(
      this, dst_alloc, dst_alloc, n, value_size, value_size, value_type, 1,
      dst_ptr.offset, dst_stride, dst_ptr.offset, dst_stride, 0.0, 0.0, true,
      true);
}

std::size_t Program::vulkan_zero_dense_fields(
    const std::vector<SNode *> &dsts,
    const std::vector<int> &value_types,
    const std::vector<std::size_t> &ns) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field zero-fill is only available on "
              "Vulkan.");
  TI_ERROR_IF(dsts.size() != value_types.size() || dsts.size() != ns.size(),
              "Vulkan native dense field zero-fill batch received mismatched "
              "inputs.");
  struct FillRange {
    DeviceAllocation alloc;
    std::size_t offset{0};
    std::size_t bytes{0};
  };
  std::vector<FillRange> fills;
  fills.reserve(dsts.size());
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    SNode *dst = dsts[i];
    TI_ERROR_IF(!dst,
                "Vulkan native dense field zero-fill batch received null "
                "field.");
    const size_t value_size = vulkan_transform_value_size(value_types[i]);
    TI_ERROR_IF(value_size == 0,
                "Vulkan native dense field zero-fill batch received an "
                "unsupported value type.");
    DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
    const size_t dst_stride = get_dense_field_stride(dst, value_size);
    TI_ERROR_IF(dst_stride < value_size,
                "Vulkan native dense field zero-fill batch received an "
                "invalid field stride.");
    if (ns[i] == 0) {
      continue;
    }
    TI_ERROR_IF(dst_stride != value_size,
                "Vulkan native dense field zero-fill batch currently supports "
                "only contiguous dense fields.");
    DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
    TI_ERROR_IF(!dst_alloc.device,
                "Vulkan native dense field zero-fill batch received null "
                "storage.");
    fills.push_back(FillRange{dst_alloc, dst_ptr.offset, ns[i] * value_size});
  }
  if (fills.empty()) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field zero-fill batch requires a compute "
              "device.");
  auto record_zero = [fills = std::move(fills)](
                         Device * /*op_device*/, CommandList *cmdlist) {
    for (const auto &fill : fills) {
      cmdlist->buffer_fill(fill.alloc.get_ptr(fill.offset), fill.bytes, 0);
    }
    for (const auto &fill : fills) {
      cmdlist->buffer_barrier(fill.alloc.get_ptr(fill.offset), fill.bytes);
    }
  };
  enqueue_compute_op_lambda(record_zero, {});
  return 0;
}

std::size_t Program::vulkan_gather_ndarray(Ndarray *src,
                                           Ndarray *indices,
                                           Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native gather is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native gather received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native gather currently expects 1D ndarrays.");
  TI_ERROR_IF(indices->get_nelement() != dst->get_nelement(),
              "Vulkan native gather expects indices and destination sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native gather source and destination dtypes differ.");
  const size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native gather currently expects 4-byte aligned values "
              "and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native gather currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native gather word count exceeds UINT32_MAX.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native gather requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, false);
  const size_t value_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  cache.gather_replay.rw_buffer(bindings, 0, src_alloc, 0, src_bytes);
  cache.gather_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                indices_bytes);
  cache.gather_replay.rw_buffer(bindings, 2, dst_alloc, 0, value_bytes);
  auto record_gather =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, value_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_gather_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), value_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(40);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, value_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_gather)) {
    enqueue_compute_op_lambda(record_gather, {});
  }
  return 0;
}

std::size_t Program::vulkan_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided gather is only available on Vulkan.");
  check_vulkan_indexed_copy_strided_request(
      src, indices, dst, item_bytes, src_offset, src_stride, dst_offset,
      dst_stride, false);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided gather word count exceeds UINT32_MAX.");
  TI_ERROR_IF(src->get_nelement() >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided gather source size exceeds UINT32_MAX.");
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native strided gather {} exceeds UINT32_MAX words.",
                name);
  };
  check_word_param("source offset", src_offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native strided gather requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, true);
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const bool profiler_scopes = profiler != nullptr;
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src->get_nelement()),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  const size_t params_bytes = params.size() * sizeof(uint32_t);
  cache.gather_strided_replay.rw_buffer(bindings, 0, src_alloc, 0,
                                        src_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                        indices_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 2, dst_alloc, 0,
                                        dst_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 3, params_alloc, 0,
                                        params_bytes);
  auto record_gather_strided =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, dst_bytes, groups, params_alloc, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        dispatch_pipeline(
            cmdlist, pipeline, bindings, groups, 1, 1,
            profiler_scopes ? "vulkan_gather_strided_u32_by_i32" : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(41);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  push_vulkan_command_key_range(command_key, params_alloc, 0, params_bytes);
  command_key.push(groups);
  for (uint32_t word : params) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_strided_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_gather_strided)) {
    enqueue_compute_op_lambda(record_gather_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_gather_dense_field(SNode *src,
                                               Ndarray *indices,
                                               SNode *dst,
                                               int value_type,
                                               std::size_t src_n,
                                               std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field gather is only available on Vulkan.");
  check_vulkan_indexed_copy_dense_field_request(
      this, src, indices, dst, value_type, src_n, dst_n, false);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_bytes = vulkan_transform_value_size(value_type);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field gather word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(src_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field gather source size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, item_bytes);
  const size_t dst_stride = get_dense_field_stride(dst, item_bytes);
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native dense field gather {} exceeds UINT32_MAX "
                "words.",
                name);
  };
  check_word_param("source offset", src_ptr.offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_ptr.offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field gather requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t indices_bytes = n * sizeof(int32_t);
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    ShaderResourceSet *bindings = cache.cached_resource_set(false);
    Pipeline *pipeline = cache.indexed_copy_pipeline(false, false);
    const size_t src_bytes = src_n * item_bytes;
    const size_t dst_bytes = n * item_bytes;
    cache.gather_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                  src_bytes);
    cache.gather_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                  indices_bytes);
    cache.gather_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                  dst_bytes);
    auto record_gather_dense =
        [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_ptr,
         dst_ptr, src_bytes, indices_bytes, dst_bytes, groups,
         profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
          dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                            profiler_scopes
                                ? "vulkan_gather_dense_field_u32_by_i32"
                                : nullptr);
          cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                  dst_bytes);
        };
    VulkanCommandReplayKey command_key;
    command_key.push(42);
    command_key.push(static_cast<uint64_t>(value_type));
    push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                  src_bytes);
    push_vulkan_command_key_range(command_key, indices_alloc, 0,
                                  indices_bytes);
    push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                  dst_bytes);
    command_key.push(groups);
    command_key.push_ptr(pipeline);
    command_key.push_ptr(bindings);
    if (!cache.gather_command_replay.submit_or_record(
            this, device, command_key, profiler_scopes,
            record_gather_dense)) {
      enqueue_compute_op_lambda(record_gather_dense, {});
    }
    return 0;
  }
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, true);
  const size_t src_bytes =
      src_ptr.offset + (src_n == 0 ? item_bytes
                                   : (src_n - 1) * src_stride + item_bytes);
  const size_t dst_bytes =
      dst_ptr.offset + (dst_n == 0 ? item_bytes
                                   : (dst_n - 1) * dst_stride + item_bytes);
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_n),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_ptr.offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_ptr.offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const size_t params_bytes = params.size() * sizeof(uint32_t);
  cache.gather_strided_replay.rw_buffer(bindings, 0, src_alloc, 0,
                                        src_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                        indices_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 2, dst_alloc, 0,
                                        dst_bytes);
  cache.gather_strided_replay.rw_buffer(bindings, 3, params_alloc, 0,
                                        params_bytes);
  auto record_gather_dense_strided =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, dst_bytes, groups, params_alloc, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_gather_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(43);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  push_vulkan_command_key_range(command_key, params_alloc, 0, params_bytes);
  command_key.push(groups);
  for (uint32_t word : params) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_strided_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_gather_dense_strided)) {
    enqueue_compute_op_lambda(record_gather_dense_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_gather_dense_field_packed(SNode *src,
                                                      Ndarray *indices,
                                                      SNode *dst,
                                                      int value_type,
                                                      std::size_t src_n,
                                                      std::size_t dst_n,
                                                      int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field gather is only available on "
              "Vulkan.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != dst_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native packed dense field gather expects 1D i32 "
              "indices matching destination size.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field gather received an invalid "
              "value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field gather word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(src_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field gather source size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field gather expects a packed "
              "contiguous MatrixField layout.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field gather requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, false);
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = src_n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  cache.gather_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                src_bytes);
  cache.gather_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                indices_bytes);
  cache.gather_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  auto record_gather_packed =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       src_bytes, indices_bytes, dst_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_gather_packed_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(142);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_gather_packed)) {
    enqueue_compute_op_lambda(record_gather_packed, {});
  }
  return 0;
}

std::size_t Program::vulkan_gather_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field gather is only available on "
              "Vulkan.");
  TI_ERROR_IF(indices_n != dst_n,
              "Vulkan native packed dense field gather expects field indices "
              "matching destination size.");
  const size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field gather received an invalid "
              "value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field gather word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(src_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field gather source size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field gather expects a packed "
              "contiguous MatrixField layout.");
  TI_ERROR_IF(get_dense_field_stride(indices, sizeof(int32_t)) !=
                  sizeof(int32_t),
              "Vulkan native packed dense field gather requires contiguous "
              "i32 field indices.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field gather requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, false);
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = src_n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  cache.gather_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                src_bytes);
  cache.gather_replay.rw_buffer(bindings, 1, indices_alloc,
                                indices_ptr.offset, indices_bytes);
  cache.gather_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  auto record_gather_packed_indices_field =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       dst_bytes, groups, profiler_scopes](Device * /*op_device*/,
                                           CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_gather_packed_field_indices_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(143);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_gather_packed_indices_field)) {
    enqueue_compute_op_lambda(record_gather_packed_indices_field, {});
  }
  return 0;
}

std::size_t Program::vulkan_gather_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field gather is only available on Vulkan.");
  check_vulkan_indexed_copy_dense_field_indices_field_request(
      this, src, indices, dst, value_type, src_n, indices_n, dst_n, false);
  const size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const size_t item_bytes = vulkan_transform_value_size(value_type);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field gather word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(src_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field gather source size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, item_bytes);
  const size_t dst_stride = get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride != item_bytes || dst_stride != item_bytes,
              "Vulkan native dense field gather with field indices currently "
              "requires contiguous source and destination fields.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field gather requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = src_n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  ShaderResourceSet *bindings = cache.cached_resource_set(false);
  Pipeline *pipeline = cache.indexed_copy_pipeline(false, false);
  cache.gather_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                src_bytes);
  cache.gather_replay.rw_buffer(bindings, 1, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  cache.gather_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  auto record_gather_dense_indices_field =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       dst_bytes, groups, profiler_scopes](Device * /*op_device*/,
                                           CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_gather_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(44);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.gather_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_gather_dense_indices_field)) {
    enqueue_compute_op_lambda(record_gather_dense_indices_field, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scatter is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native scatter received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native scatter currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "Vulkan native scatter expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native scatter source and destination dtypes differ.");
  const size_t item_bytes = src->get_element_size();
  TI_ERROR_IF(item_bytes == 0 || item_bytes % sizeof(uint32_t) != 0 ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native scatter currently expects 4-byte aligned values "
              "and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter currently supports at most UINT32_MAX "
              "items.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter word count exceeds UINT32_MAX.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scatter requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = item_words == 1
                           ? cache.indexed_copy_dense_u32_scatter_pipeline()
                           : cache.indexed_copy_pipeline(true, false);
  const size_t value_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  cache.scatter_replay.rw_buffer(bindings, 0, src_alloc, 0, value_bytes);
  cache.scatter_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                 indices_bytes);
  cache.scatter_replay.rw_buffer(bindings, 2, dst_alloc, 0, dst_bytes);
  auto record_scatter =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, value_bytes,
       indices_bytes, dst_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? "vulkan_scatter_u32_by_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(45);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, 0, value_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_scatter)) {
    enqueue_compute_op_lambda(record_scatter, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scatter is only available on Vulkan.");
  check_vulkan_indexed_copy_strided_request(
      src, indices, dst, item_bytes, src_offset, src_stride, dst_offset,
      dst_stride, true);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter word count exceeds UINT32_MAX.");
  TI_ERROR_IF(dst->get_nelement() >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter destination size exceeds "
              "UINT32_MAX.");
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native strided scatter {} exceeds UINT32_MAX words.",
                name);
  };
  check_word_param("source offset", src_offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native strided scatter requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(true);
  Pipeline *pipeline = cache.indexed_copy_pipeline(true, true);
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const bool profiler_scopes = profiler != nullptr;
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(dst->get_nelement()),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  const size_t params_bytes = params.size() * sizeof(uint32_t);
  cache.scatter_strided_replay.rw_buffer(bindings, 0, src_alloc, 0,
                                         src_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                         indices_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 2, dst_alloc, 0,
                                         dst_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 3, params_alloc, 0,
                                         params_bytes);
  auto record_scatter_strided =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, dst_bytes, groups, params_alloc, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        dispatch_pipeline(
            cmdlist, pipeline, bindings, groups, 1, 1,
            profiler_scopes ? "vulkan_scatter_strided_u32_by_i32" : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(46);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  push_vulkan_command_key_range(command_key, params_alloc, 0, params_bytes);
  command_key.push(groups);
  for (uint32_t word : params) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_strided_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_strided)) {
    enqueue_compute_op_lambda(record_scatter_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_dense_field(SNode *src,
                                                Ndarray *indices,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t src_n,
                                                std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field scatter is only available on "
              "Vulkan.");
  check_vulkan_indexed_copy_dense_field_request(
      this, src, indices, dst, value_type, src_n, dst_n, true);
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t item_bytes = vulkan_transform_value_size(value_type);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(dst_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter destination size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, item_bytes);
  const size_t dst_stride = get_dense_field_stride(dst, item_bytes);
  auto check_word_param = [](const char *name, size_t value) {
    TI_ERROR_IF(value / sizeof(uint32_t) >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native dense field scatter {} exceeds UINT32_MAX "
                "words.",
                name);
  };
  check_word_param("source offset", src_ptr.offset);
  check_word_param("source stride", src_stride);
  check_word_param("destination offset", dst_ptr.offset);
  check_word_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field scatter requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t indices_bytes = n * sizeof(int32_t);
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  if (src_stride == item_bytes && dst_stride == item_bytes) {
    ShaderResourceSet *bindings = cache.cached_resource_set(true);
    Pipeline *pipeline = item_words == 1
                             ? cache.indexed_copy_dense_u32_scatter_pipeline()
                             : cache.indexed_copy_pipeline(true, false);
    const size_t src_bytes = n * item_bytes;
    const size_t dst_bytes = dst_n * item_bytes;
    cache.scatter_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                   src_bytes);
    cache.scatter_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                   indices_bytes);
    cache.scatter_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                   dst_bytes);
    auto record_scatter_dense =
        [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_ptr,
         dst_ptr, src_bytes, indices_bytes, dst_bytes, groups,
         profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
          dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                            profiler_scopes
                                ? "vulkan_scatter_dense_field_u32_by_i32"
                                : nullptr);
          cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                  dst_bytes);
        };
    VulkanCommandReplayKey command_key;
    command_key.push(47);
    command_key.push(static_cast<uint64_t>(value_type));
    push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                  src_bytes);
    push_vulkan_command_key_range(command_key, indices_alloc, 0,
                                  indices_bytes);
    push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                  dst_bytes);
    command_key.push(groups);
    command_key.push_ptr(pipeline);
    command_key.push_ptr(bindings);
    if (!cache.scatter_command_replay.submit_or_record(
            this, device, command_key, profiler_scopes,
            record_scatter_dense)) {
      enqueue_compute_op_lambda(record_scatter_dense, {});
    }
    return 0;
  }
  cache.ensure_indexed_copy_params();
  ShaderResourceSet *bindings = cache.cached_strided_resource_set(true);
  Pipeline *pipeline = cache.indexed_copy_pipeline(true, true);
  const size_t src_bytes =
      src_ptr.offset + (src_n == 0 ? item_bytes
                                   : (src_n - 1) * src_stride + item_bytes);
  const size_t dst_bytes =
      dst_ptr.offset + (dst_n == 0 ? item_bytes
                                   : (dst_n - 1) * dst_stride + item_bytes);
  std::array<uint32_t, 7> params{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(dst_n),
      static_cast<uint32_t>(item_words),
      static_cast<uint32_t>(src_ptr.offset / sizeof(uint32_t)),
      static_cast<uint32_t>(src_stride / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_ptr.offset / sizeof(uint32_t)),
      static_cast<uint32_t>(dst_stride / sizeof(uint32_t)),
  };
  const DeviceAllocation params_alloc = cache.indexed_copy_params;
  const size_t params_bytes = params.size() * sizeof(uint32_t);
  cache.scatter_strided_replay.rw_buffer(bindings, 0, src_alloc, 0,
                                         src_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                         indices_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 2, dst_alloc, 0,
                                         dst_bytes);
  cache.scatter_strided_replay.rw_buffer(bindings, 3, params_alloc, 0,
                                         params_bytes);
  auto record_scatter_dense_strided =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_bytes,
       indices_bytes, dst_bytes, groups, params_alloc, params,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        for (uint32_t i = 0; i < params.size(); ++i) {
          cmdlist->buffer_fill(params_alloc.get_ptr(i * sizeof(uint32_t)),
                               sizeof(uint32_t), params[i]);
        }
        cmdlist->buffer_barrier(params_alloc);
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_scatter_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(48);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  push_vulkan_command_key_range(command_key, params_alloc, 0, params_bytes);
  command_key.push(groups);
  for (uint32_t word : params) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_strided_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_dense_strided)) {
    enqueue_compute_op_lambda(record_scatter_dense_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_dense_field_packed(SNode *src,
                                                       Ndarray *indices,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t src_n,
                                                       std::size_t dst_n,
                                                       int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field scatter is only available on "
              "Vulkan.");
  TI_ERROR_IF(!indices || indices->shape.size() != 1 ||
                  indices->get_nelement() != src_n ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native packed dense field scatter expects 1D i32 "
              "indices matching source size.");
  const size_t n = indices->get_nelement();
  if (n == 0) {
    return 0;
  }
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field scatter received an invalid "
              "value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(dst_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter destination size "
              "exceeds UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field scatter expects a packed "
              "contiguous MatrixField layout.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field scatter requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = item_words == 1
                           ? cache.indexed_copy_dense_u32_scatter_pipeline()
                           : cache.indexed_copy_pipeline(true, false);
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst_n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  cache.scatter_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                 src_bytes);
  cache.scatter_replay.rw_buffer(bindings, 1, indices_alloc, 0,
                                 indices_bytes);
  cache.scatter_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                 dst_bytes);
  auto record_scatter_packed =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       src_bytes, indices_bytes, dst_bytes, groups,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_scatter_packed_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(147);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_scatter_packed)) {
    enqueue_compute_op_lambda(record_scatter_packed, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field scatter is only available on "
              "Vulkan.");
  TI_ERROR_IF(src_n != indices_n,
              "Vulkan native packed dense field scatter expects field indices "
              "matching source size.");
  const size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const size_t value_size = vulkan_transform_value_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field scatter received an invalid "
              "value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(dst_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter destination size "
              "exceeds UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field scatter expects a packed "
              "contiguous MatrixField layout.");
  TI_ERROR_IF(get_dense_field_stride(indices, sizeof(int32_t)) !=
                  sizeof(int32_t),
              "Vulkan native packed dense field scatter requires contiguous "
              "i32 field indices.");
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field scatter requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = item_words == 1
                           ? cache.indexed_copy_dense_u32_scatter_pipeline()
                           : cache.indexed_copy_pipeline(true, false);
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst_n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  cache.scatter_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                 src_bytes);
  cache.scatter_replay.rw_buffer(bindings, 1, indices_alloc,
                                 indices_ptr.offset, indices_bytes);
  cache.scatter_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                 dst_bytes);
  auto record_scatter_packed_indices_field =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       dst_bytes, groups, profiler_scopes](Device * /*op_device*/,
                                           CommandList *cmdlist) {
        dispatch_pipeline(
            cmdlist, pipeline, bindings, groups, 1, 1,
            profiler_scopes ? "vulkan_scatter_packed_field_indices_u32_by_i32"
                            : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(148);
  command_key.push(static_cast<uint64_t>(item_bytes));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_packed_indices_field)) {
    enqueue_compute_op_lambda(record_scatter_packed_indices_field, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field scatter is only available on "
              "Vulkan.");
  check_vulkan_indexed_copy_dense_field_indices_field_request(
      this, src, indices, dst, value_type, src_n, indices_n, dst_n, true);
  const size_t n = indices_n;
  if (n == 0) {
    return 0;
  }
  const size_t item_bytes = vulkan_transform_value_size(value_type);
  const size_t item_words = item_bytes / sizeof(uint32_t);
  const size_t word_count = n * item_words;
  TI_ERROR_IF(word_count >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter word count exceeds "
              "UINT32_MAX.");
  TI_ERROR_IF(dst_n >
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter destination size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, item_bytes);
  const size_t dst_stride = get_dense_field_stride(dst, item_bytes);
  TI_ERROR_IF(src_stride != item_bytes || dst_stride != item_bytes,
              "Vulkan native dense field scatter with field indices currently "
              "requires contiguous source and destination fields.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field scatter requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t src_bytes = n * item_bytes;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst_n * item_bytes;
  const uint32_t groups =
      static_cast<uint32_t>((word_count + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  ShaderResourceSet *bindings = cache.cached_resource_set(true);
  Pipeline *pipeline = item_words == 1
                           ? cache.indexed_copy_dense_u32_scatter_pipeline()
                           : cache.indexed_copy_pipeline(true, false);
  cache.scatter_replay.rw_buffer(bindings, 0, src_alloc, src_ptr.offset,
                                 src_bytes);
  cache.scatter_replay.rw_buffer(bindings, 1, indices_alloc, indices_ptr.offset,
                                 indices_bytes);
  cache.scatter_replay.rw_buffer(bindings, 2, dst_alloc, dst_ptr.offset,
                                 dst_bytes);
  auto record_scatter_dense_indices_field =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       dst_bytes, groups, profiler_scopes](Device * /*op_device*/,
                                           CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? "vulkan_scatter_dense_field_u32_by_i32"
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(49);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_dense_indices_field)) {
    enqueue_compute_op_lambda(record_scatter_dense_indices_field, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_add_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native scatter-add is only available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native scatter-add received a null ndarray.");
  TI_ERROR_IF(src->shape.size() != 1 || indices->shape.size() != 1 ||
                  dst->shape.size() != 1,
              "Vulkan native scatter-add currently expects 1D ndarrays.");
  TI_ERROR_IF(src->get_nelement() != indices->get_nelement(),
              "Vulkan native scatter-add expects source and indices sizes to "
              "match.");
  TI_ERROR_IF(src->get_element_size() != dst->get_element_size(),
              "Vulkan native scatter-add source and destination dtypes differ.");
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native scatter-add does not support the requested value "
              "type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0, "Unsupported Vulkan scatter-add value type.");
  TI_ERROR_IF(src->get_element_size() != value_size ||
                  indices->get_element_size() != sizeof(int32_t),
              "Vulkan native scatter-add dtype does not match the requested "
              "value type and i32 indices.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native scatter-add currently supports at most UINT32_MAX "
              "source items.");
  const size_t n = indices->get_nelement();
  if (n == 0 || dst->get_nelement() == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native scatter-add requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native scatter-add could not find a pipeline for the "
              "requested value type.");
  const size_t value_bytes = n * value_size;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * value_size;
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  cache.scatter_add_replay[value_type].rw_buffer(bindings, 0, src_alloc, 0,
                                                 value_bytes);
  cache.scatter_add_replay[value_type].rw_buffer(bindings, 1, indices_alloc, 0,
                                                 indices_bytes);
  cache.scatter_add_replay[value_type].rw_buffer(bindings, 2, dst_alloc, 0,
                                                 dst_bytes);
  auto record_scatter_add =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, value_bytes,
       indices_bytes, dst_bytes, groups, value_type,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_scatter_add_f32_by_i32"
                                     : value_type == 2
                                     ? "vulkan_scatter_add_u32_by_i32"
                                     : value_type == 3
                                     ? "vulkan_scatter_add_u64_by_i32"
                                     : value_type == 4
                                     ? "vulkan_scatter_add_i64_by_i32"
                                     : value_type == 5
                                     ? "vulkan_scatter_add_f64_by_i32"
                                     : "vulkan_scatter_add_i32_by_i32")
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(50);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, value_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes, record_scatter_add)) {
    enqueue_compute_op_lambda(record_scatter_add, {});
  }
  return 0;
}

std::size_t Program::vulkan_scatter_add_member_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_scatter_add_strided_ndarray(
      src, indices, dst, value_type, offset, stride, 0,
      vulkan_scan_value_type_size(value_type));
}

std::size_t Program::vulkan_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native strided scatter-add is only available on Vulkan.");
  check_vulkan_scatter_add_strided_request(
      src, indices, dst, value_type, src_offset, src_stride, dst_offset,
      dst_stride);
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native strided scatter-add does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan strided scatter-add value type.");
  TI_ERROR_IF(indices->get_nelement() >
                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native strided scatter-add currently supports at most "
              "UINT32_MAX source items.");
  const size_t n = indices->get_nelement();
  if (n == 0 || dst->get_nelement() == 0) {
    return 0;
  }
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native strided scatter-add requires a compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  cache.ensure_scatter_add_strided_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_strided_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_strided_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native strided scatter-add could not find a pipeline "
              "for the requested value type.");
  const std::array<uint32_t, 6> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_offset / value_size),
      static_cast<uint32_t>(src_stride / value_size),
      static_cast<uint32_t>(dst->get_nelement()),
      static_cast<uint32_t>(dst_offset / value_size),
      static_cast<uint32_t>(dst_stride / value_size),
  };
  const size_t src_bytes = src->get_nelement() * src->get_element_size();
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst->get_nelement() * dst->get_element_size();
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const DeviceAllocation src_alloc = src->ndarray_alloc_;
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc = dst->ndarray_alloc_;
  const bool profiler_scopes = profiler != nullptr;
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 0, src_alloc, 0, src_bytes);
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 1, indices_alloc, 0, indices_bytes);
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 2, dst_alloc, 0, dst_bytes);
  auto record_scatter_add_strided =
      [dst_alloc, pipeline, bindings, dst_bytes, param_words, push_bytes, groups,
       value_type, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
            1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_scatter_add_f32_by_i32_strided"
                       : value_type == 2
                       ? "vulkan_scatter_add_u32_by_i32_strided"
                       : value_type == 3
                       ? "vulkan_scatter_add_u64_by_i32_strided"
                       : value_type == 4
                       ? "vulkan_scatter_add_i64_by_i32_strided"
                       : value_type == 5
                       ? "vulkan_scatter_add_f64_by_i32_strided"
                       : "vulkan_scatter_add_i32_by_i32_strided")
                : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(51);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_strided_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_add_strided)) {
    enqueue_compute_op_lambda(record_scatter_add_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_add_dense_field(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field scatter-add is only available on "
              "Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native dense field scatter-add received a null "
              "argument.");
  TI_ERROR_IF(indices->shape.size() != 1,
              "Vulkan native dense field scatter-add expects 1D indices.");
  TI_ERROR_IF(indices->get_element_size() != sizeof(int32_t),
              "Vulkan native dense field scatter-add expects i32 indices.");
  TI_ERROR_IF(src_n != indices->get_nelement(),
              "Vulkan native dense field scatter-add expects source and "
              "indices sizes to match.");
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native dense field scatter-add does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan dense field scatter-add value type.");
  const size_t n = indices->get_nelement();
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter-add currently supports at "
              "most UINT32_MAX source items.");
  TI_ERROR_IF(dst_n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter-add destination size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, value_size);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  auto check_param = [value_size](const char *name, size_t value) {
    TI_ERROR_IF(value % value_size != 0,
                "Vulkan native dense field scatter-add {} must align to "
                "value size.",
                name);
    TI_ERROR_IF(value / value_size >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native dense field scatter-add {} exceeds UINT32_MAX "
                "items.",
                name);
  };
  TI_ERROR_IF(src_stride < value_size || dst_stride < value_size,
              "Vulkan native dense field scatter-add received an invalid "
              "field stride.");
  check_param("source offset", src_ptr.offset);
  check_param("source stride", src_stride);
  check_param("destination offset", dst_ptr.offset);
  check_param("destination stride", dst_stride);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field scatter-add requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t indices_bytes = n * sizeof(int32_t);
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  if (src_stride == value_size && dst_stride == value_size) {
    ShaderResourceSet *bindings =
        cache.cached_scatter_add_resource_set(value_type);
    Pipeline *pipeline = cache.scatter_add_pipeline(value_type);
    TI_ERROR_IF(!pipeline,
                "Vulkan native dense field scatter-add could not find a "
                "pipeline for the requested value type.");
    const size_t value_bytes = n * value_size;
    const size_t dst_bytes = dst_n * value_size;
    cache.scatter_add_replay[value_type].rw_buffer(
        bindings, 0, src_alloc, src_ptr.offset, value_bytes);
    cache.scatter_add_replay[value_type].rw_buffer(
        bindings, 1, indices_alloc, 0, indices_bytes);
    cache.scatter_add_replay[value_type].rw_buffer(
        bindings, 2, dst_alloc, dst_ptr.offset, dst_bytes);
    auto record_scatter_add_dense =
        [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, src_ptr,
         dst_ptr, value_bytes, indices_bytes, dst_bytes, groups, value_type,
         profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
          dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                            profiler_scopes
                                ? (value_type == 1
                                       ? "vulkan_scatter_add_dense_f32_by_i32"
                                       : value_type == 2
                                       ? "vulkan_scatter_add_dense_u32_by_i32"
                                       : value_type == 3
                                       ? "vulkan_scatter_add_dense_u64_by_i32"
                                       : value_type == 4
                                       ? "vulkan_scatter_add_dense_i64_by_i32"
                                       : value_type == 5
                                       ? "vulkan_scatter_add_dense_f64_by_i32"
                                       : "vulkan_scatter_add_dense_i32_by_i32")
                                : nullptr);
          cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                  dst_bytes);
        };
    VulkanCommandReplayKey command_key;
    command_key.push(52);
    command_key.push(static_cast<uint64_t>(value_type));
    push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                  value_bytes);
    push_vulkan_command_key_range(command_key, indices_alloc, 0,
                                  indices_bytes);
    push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                  dst_bytes);
    command_key.push(groups);
    command_key.push_ptr(pipeline);
    command_key.push_ptr(bindings);
    if (!cache.scatter_add_command_replay[value_type].submit_or_record(
            this, device, command_key, profiler_scopes,
            record_scatter_add_dense)) {
      enqueue_compute_op_lambda(record_scatter_add_dense, {});
    }
    return 0;
  }

  cache.ensure_scatter_add_strided_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_strided_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_strided_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native dense field scatter-add could not find a strided "
              "pipeline for the requested value type.");
  const std::array<uint32_t, 6> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_ptr.offset / value_size),
      static_cast<uint32_t>(src_stride / value_size),
      static_cast<uint32_t>(dst_n),
      static_cast<uint32_t>(dst_ptr.offset / value_size),
      static_cast<uint32_t>(dst_stride / value_size),
  };
  const size_t src_bytes =
      src_ptr.offset + (src_n == 0 ? value_size
                                   : (src_n - 1) * src_stride + value_size);
  const size_t dst_bytes =
      dst_ptr.offset + (dst_n == 0 ? value_size
                                   : (dst_n - 1) * dst_stride + value_size);
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 0, src_alloc, 0, src_bytes);
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 1, indices_alloc, 0, indices_bytes);
  cache.scatter_add_strided_replay[value_type].rw_buffer(
      bindings, 2, dst_alloc, 0, dst_bytes);
  auto record_scatter_add_dense_strided =
      [dst_alloc, pipeline, bindings, dst_bytes, param_words, push_bytes, groups,
       value_type, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
            1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_scatter_add_dense_f32_by_i32"
                       : value_type == 2
                       ? "vulkan_scatter_add_dense_u32_by_i32"
                       : value_type == 3
                       ? "vulkan_scatter_add_dense_u64_by_i32"
                       : value_type == 4
                       ? "vulkan_scatter_add_dense_i64_by_i32"
                       : value_type == 5
                       ? "vulkan_scatter_add_dense_f64_by_i32"
                       : "vulkan_scatter_add_dense_i32_by_i32")
                : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(53);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_strided_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_add_dense_strided)) {
    enqueue_compute_op_lambda(record_scatter_add_dense_strided, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_add_dense_field_packed(
    SNode *src,
    Ndarray *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t dst_n,
    int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field scatter-add is only "
              "available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native packed dense field scatter-add received a null "
              "argument.");
  TI_ERROR_IF(indices->shape.size() != 1 ||
                  indices->get_element_size() != sizeof(int32_t) ||
                  indices->get_nelement() != src_n,
              "Vulkan native packed dense field scatter-add expects 1D i32 "
              "indices matching source size.");
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native packed dense field scatter-add does not support "
              "the requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field scatter-add received an "
              "invalid value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t n = indices->get_nelement();
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  dst_n >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter-add item count "
              "exceeds UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field scatter-add expects a packed "
              "contiguous MatrixField layout.");
  auto check_param = [value_size](const char *name, size_t value) {
    TI_ERROR_IF(value % value_size != 0,
                "Vulkan native packed dense field scatter-add {} must align "
                "to value size.",
                name);
    TI_ERROR_IF(value / value_size >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native packed dense field scatter-add {} exceeds "
                "UINT32_MAX items.",
                name);
  };
  check_param("source offset", src_ptr.offset);
  check_param("destination offset", dst_ptr.offset);
  check_param("item stride", item_bytes);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field scatter-add requires a "
              "compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  TI_ERROR_IF(
      static_cast<size_t>(lane_count) >
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / n ||
          static_cast<size_t>(lane_count) >
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / dst_n,
      "Vulkan native packed dense field scatter-add scalar item count "
      "exceeds UINT32_MAX.");
  cache.ensure_scatter_add_packed_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_packed_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_packed_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native packed dense field scatter-add could not find a "
              "packed pipeline for the requested value type.");

  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc = indices->ndarray_alloc_;
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src_ptr.offset + src_n * item_bytes;
  const size_t dst_bytes = dst_ptr.offset + dst_n * item_bytes;
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 0, src_alloc, 0, src_bytes);
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 1, indices_alloc, 0, indices_bytes);
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 2, dst_alloc, 0, dst_bytes);

  const uint32_t scalar_items =
      static_cast<uint32_t>(n * static_cast<size_t>(lane_count));
  const uint32_t groups =
      static_cast<uint32_t>((scalar_items + kBlockSize - 1) / kBlockSize);
  const std::array<uint32_t, 5> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_ptr.offset / value_size),
      static_cast<uint32_t>(dst_n),
      static_cast<uint32_t>(dst_ptr.offset / value_size),
      static_cast<uint32_t>(lane_count),
  };
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;
  auto record_scatter_add_packed =
      [dst_alloc, pipeline, bindings, dst_bytes, param_words, push_bytes, groups,
       value_type, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
            1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_scatter_add_packed_f32_by_i32"
                       : value_type == 2
                       ? "vulkan_scatter_add_packed_u32_by_i32"
                       : value_type == 3
                       ? "vulkan_scatter_add_packed_u64_by_i32"
                       : value_type == 4
                       ? "vulkan_scatter_add_packed_i64_by_i32"
                       : value_type == 5
                       ? "vulkan_scatter_add_packed_f64_by_i32"
                       : "vulkan_scatter_add_packed_i32_by_i32")
                : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(155);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, 0, indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_packed_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_add_packed)) {
    enqueue_compute_op_lambda(record_scatter_add_packed, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_add_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native packed dense field scatter-add is only "
              "available on Vulkan.");
  TI_ERROR_IF(!src || !indices || !dst,
              "Vulkan native packed dense field scatter-add received a null "
              "argument.");
  TI_ERROR_IF(src_n != indices_n,
              "Vulkan native packed dense field scatter-add expects source "
              "and field-index sizes to match.");
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native packed dense field scatter-add does not support "
              "the requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0 || lane_count <= 0,
              "Vulkan native packed dense field scatter-add received an "
              "invalid value type or lane count.");
  const size_t item_bytes = value_size * static_cast<size_t>(lane_count);
  const size_t n = indices_n;
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  dst_n >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native packed dense field scatter-add item count "
              "exceeds UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  TI_ERROR_IF(get_dense_field_stride(src, value_size) != item_bytes ||
                  get_dense_field_stride(dst, value_size) != item_bytes,
              "Vulkan native packed dense field scatter-add expects a packed "
              "contiguous MatrixField layout.");
  TI_ERROR_IF(get_dense_field_stride(indices, sizeof(int32_t)) !=
                  sizeof(int32_t),
              "Vulkan native packed dense field scatter-add requires "
              "contiguous i32 field indices.");
  auto check_param = [value_size](const char *name, size_t value) {
    TI_ERROR_IF(value % value_size != 0,
                "Vulkan native packed dense field scatter-add {} must align "
                "to value size.",
                name);
    TI_ERROR_IF(value / value_size >
                    static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                "Vulkan native packed dense field scatter-add {} exceeds "
                "UINT32_MAX items.",
                name);
  };
  check_param("source offset", src_ptr.offset);
  check_param("destination offset", dst_ptr.offset);
  check_param("item stride", item_bytes);

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native packed dense field scatter-add requires a "
              "compute device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  TI_ERROR_IF(
      static_cast<size_t>(lane_count) >
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / n ||
          static_cast<size_t>(lane_count) >
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / dst_n,
      "Vulkan native packed dense field scatter-add scalar item count "
      "exceeds UINT32_MAX.");
  cache.ensure_scatter_add_packed_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_packed_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_packed_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native packed dense field scatter-add could not find a "
              "packed pipeline for the requested value type.");

  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t src_bytes = src_ptr.offset + src_n * item_bytes;
  const size_t dst_bytes = dst_ptr.offset + dst_n * item_bytes;
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 0, src_alloc, 0, src_bytes);
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 1, indices_alloc, indices_ptr.offset, indices_bytes);
  cache.scatter_add_packed_replay[value_type].rw_buffer(
      bindings, 2, dst_alloc, 0, dst_bytes);

  const uint32_t scalar_items =
      static_cast<uint32_t>(n * static_cast<size_t>(lane_count));
  const uint32_t groups =
      static_cast<uint32_t>((scalar_items + kBlockSize - 1) / kBlockSize);
  const std::array<uint32_t, 5> param_words{
      static_cast<uint32_t>(n),
      static_cast<uint32_t>(src_ptr.offset / value_size),
      static_cast<uint32_t>(dst_n),
      static_cast<uint32_t>(dst_ptr.offset / value_size),
      static_cast<uint32_t>(lane_count),
  };
  const uint32_t push_bytes =
      static_cast<uint32_t>(param_words.size() * sizeof(uint32_t));
  const bool profiler_scopes = profiler != nullptr;
  auto record_scatter_add_packed_indices_field =
      [dst_alloc, pipeline, bindings, dst_bytes, param_words, push_bytes, groups,
       value_type, profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline_with_push_constants(
            cmdlist, pipeline, bindings, param_words.data(), push_bytes, groups,
            1, 1,
            profiler_scopes
                ? (value_type == 1
                       ? "vulkan_scatter_add_packed_f32_by_i32"
                       : value_type == 2
                       ? "vulkan_scatter_add_packed_u32_by_i32"
                       : value_type == 3
                       ? "vulkan_scatter_add_packed_u64_by_i32"
                       : value_type == 4
                       ? "vulkan_scatter_add_packed_i64_by_i32"
                       : value_type == 5
                       ? "vulkan_scatter_add_packed_f64_by_i32"
                       : "vulkan_scatter_add_packed_i32_by_i32")
                : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(0), dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(156);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, 0, src_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, 0, dst_bytes);
  command_key.push(groups);
  for (uint32_t word : param_words) {
    command_key.push(word);
  }
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_packed_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_add_packed_indices_field)) {
    enqueue_compute_op_lambda(record_scatter_add_packed_indices_field, {});
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_scatter_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field scatter-add is only available on "
              "Vulkan.");
  check_vulkan_indexed_copy_dense_field_indices_field_request(
      this, src, indices, dst, value_type, src_n, indices_n, dst_n, true);
  TI_ERROR_IF(!vulkan_scatter_add_value_type_available(value_type),
              "Vulkan native dense field scatter-add does not support the "
              "requested value type.");
  const size_t value_size = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(value_size == 0,
              "Unsupported Vulkan dense field scatter-add value type.");
  const size_t n = indices_n;
  if (n == 0 || dst_n == 0) {
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter-add currently supports at "
              "most UINT32_MAX source items.");
  TI_ERROR_IF(dst_n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field scatter-add destination size exceeds "
              "UINT32_MAX.");
  DevicePtr src_ptr = get_dense_field_device_ptr(src);
  DevicePtr indices_ptr = get_dense_field_device_ptr(indices);
  DevicePtr dst_ptr = get_dense_field_device_ptr(dst);
  const size_t src_stride = get_dense_field_stride(src, value_size);
  const size_t dst_stride = get_dense_field_stride(dst, value_size);
  TI_ERROR_IF(src_stride != value_size || dst_stride != value_size,
              "Vulkan native dense field scatter-add with field indices "
              "currently requires contiguous source and destination fields.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan native dense field scatter-add requires a compute "
              "device.");
  auto cache_lease = get_indexed_copy_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation src_alloc{src_ptr.device, src_ptr.alloc_id};
  const DeviceAllocation indices_alloc{indices_ptr.device, indices_ptr.alloc_id};
  const DeviceAllocation dst_alloc{dst_ptr.device, dst_ptr.alloc_id};
  const size_t value_bytes = n * value_size;
  const size_t indices_bytes = n * sizeof(int32_t);
  const size_t dst_bytes = dst_n * value_size;
  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  ShaderResourceSet *bindings =
      cache.cached_scatter_add_resource_set(value_type);
  Pipeline *pipeline = cache.scatter_add_pipeline(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native dense field scatter-add could not find a "
              "pipeline for the requested value type.");
  cache.scatter_add_replay[value_type].rw_buffer(
      bindings, 0, src_alloc, src_ptr.offset, value_bytes);
  cache.scatter_add_replay[value_type].rw_buffer(
      bindings, 1, indices_alloc, indices_ptr.offset, indices_bytes);
  cache.scatter_add_replay[value_type].rw_buffer(
      bindings, 2, dst_alloc, dst_ptr.offset, dst_bytes);
  auto record_scatter_add_dense_indices_field =
      [src_alloc, indices_alloc, dst_alloc, pipeline, bindings, dst_ptr,
       dst_bytes, groups, value_type,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes
                              ? (value_type == 1
                                     ? "vulkan_scatter_add_dense_f32_by_i32"
                                     : value_type == 2
                                     ? "vulkan_scatter_add_dense_u32_by_i32"
                                     : value_type == 3
                                     ? "vulkan_scatter_add_dense_u64_by_i32"
                                     : value_type == 4
                                     ? "vulkan_scatter_add_dense_i64_by_i32"
                                     : value_type == 5
                                     ? "vulkan_scatter_add_dense_f64_by_i32"
                                     : "vulkan_scatter_add_dense_i32_by_i32")
                              : nullptr);
        cmdlist->buffer_barrier(dst_alloc.get_ptr(dst_ptr.offset),
                                dst_bytes);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(54);
  command_key.push(static_cast<uint64_t>(value_type));
  push_vulkan_command_key_range(command_key, src_alloc, src_ptr.offset,
                                value_bytes);
  push_vulkan_command_key_range(command_key, indices_alloc, indices_ptr.offset,
                                indices_bytes);
  push_vulkan_command_key_range(command_key, dst_alloc, dst_ptr.offset,
                                dst_bytes);
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.scatter_add_command_replay[value_type].submit_or_record(
          this, device, command_key, profiler_scopes,
          record_scatter_add_dense_indices_field)) {
    enqueue_compute_op_lambda(record_scatter_add_dense_indices_field, {});
  }
  return 0;
}

bool Program::vulkan_bucket_builder_value_type_available(int value_type) const {
  return compile_config().arch == Arch::vulkan && value_type >= 0 &&
         value_type <= 5;
}

std::size_t Program::vulkan_bucket_builder_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *offsets,
                                                       Ndarray *output,
                                                       Ndarray *cursor) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_bucket_builder_ndarray(keys, values, offsets, output, cursor,
                                       0);
}

std::size_t Program::vulkan_bucket_builder_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *offsets,
                                                   Ndarray *output,
                                                   Ndarray *cursor,
                                                   int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native bucket builder is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "Vulkan native bucket builder received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  offsets->shape.size() != 1 || output->shape.size() != 1 ||
                  cursor->shape.size() != 1,
              "Vulkan native bucket builder expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native bucket builder keys and values sizes differ.");
  TI_ERROR_IF(offsets->get_nelement() < 2,
              "Vulkan native bucket builder offsets must contain num_bins + 1 items.");
  const size_t n = keys->get_nelement();
  const size_t num_bins = offsets->get_nelement() - 1;
  TI_ERROR_IF(cursor->get_nelement() < num_bins,
              "Vulkan native bucket builder cursor is smaller than num_bins.");
  TI_ERROR_IF(output->get_nelement() < n,
              "Vulkan native bucket builder output is smaller than input values.");
  TI_ERROR_IF(!vulkan_bucket_builder_value_type_available(value_type),
              "Vulkan native bucket builder received an unsupported value type.");
  const bool value_is_64bit =
      value_type == 3 || value_type == 4 || value_type == 5;
  const size_t expected_value_size =
      value_is_64bit ? sizeof(uint64_t) : sizeof(uint32_t);
  const size_t item_bytes = values->get_element_size();
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  item_bytes == 0 ||
                  item_bytes % sizeof(uint32_t) != 0 ||
                  output->get_element_size() != item_bytes ||
                  cursor->get_element_size() != sizeof(int32_t),
              "Vulkan native bucket builder dtype does not match value type or "
              "keys/offsets/cursor are not i32, or payload is not 4-byte aligned.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_bins >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native bucket builder input is too large for u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native bucket builder requires a compute device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation keys_alloc = keys->ndarray_alloc_;
  const DeviceAllocation values_alloc = values->ndarray_alloc_;
  const DeviceAllocation offsets_alloc = offsets->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const DeviceAllocation cursor_alloc = cursor->ndarray_alloc_;
  const size_t key_bytes = n * sizeof(int32_t);
  const size_t value_bytes = n * item_bytes;
  const size_t offset_bytes = (num_bins + 1) * sizeof(int32_t);
  const size_t cursor_bytes = num_bins * sizeof(int32_t);
  const uint32_t item_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t offset_groups = static_cast<uint32_t>(
      (num_bins + 1 + kBlockSize - 1) / kBlockSize);
  constexpr size_t kPrivateChunkSize = 2048;
  const size_t private_chunks =
      n == 0 ? 0 : (n + kPrivateChunkSize - 1) / kPrivateChunkSize;
  const size_t private_partial_bytes =
      private_chunks * num_bins * sizeof(int32_t);
  const int private_enabled =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE", 1);
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MIN_N", 65536);
  const size_t private_min_n =
      private_min_n_config <= 0 ? 0 : static_cast<size_t>(private_min_n_config);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BINS", 1024);
  const size_t private_max_bins =
      private_max_bins_config <= 0 ? 0
                                   : static_cast<size_t>(private_max_bins_config);
  const int private_max_bytes_config = get_environ_config(
      "TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BYTES", 4 * 1024 * 1024);
  const size_t private_max_bytes =
      private_max_bytes_config <= 0
          ? 0
          : static_cast<size_t>(private_max_bytes_config);
  const bool use_private =
      private_enabled != 0 && n >= private_min_n && n > 0 &&
      num_bins <= private_max_bins && num_bins <= 4096 &&
      private_partial_bytes > 0 && private_partial_bytes <= private_max_bytes;
  if (use_private && cache.needs_workspace_realloc(private_partial_bytes) &&
      cache.partial != kDeviceNullAllocation) {
    synchronize();
  }
  if (use_private) {
    cache.ensure_workspace(private_partial_bytes);
  }
  Pipeline *clear_pipeline = nullptr;
  Pipeline *count_pipeline = nullptr;
  Pipeline *count_private_pipeline = nullptr;
  Pipeline *prefix_pipeline = cache.bucket_prefix_i32_pipeline();
  Pipeline *prefix_chunks_pipeline = nullptr;
  const int scatter_value_type =
      item_bytes == expected_value_size ? value_type : 7;
  const DeviceAllocation partial_alloc = cache.partial;
  Pipeline *scatter_pipeline = nullptr;
  Pipeline *scatter_private_pipeline = nullptr;
  ShaderResourceSet *clear_bindings = nullptr;
  ShaderResourceSet *count_bindings = nullptr;
  ShaderResourceSet *count_private_bindings = nullptr;
  ShaderResourceSet *prefix_bindings =
      cache
          .bind_bucket_prefix_resource_set(this, offsets_alloc, 0, offset_bytes,
                                           cursor_alloc, 0, cursor_bytes)
          .bindings;
  ShaderResourceSet *prefix_chunks_bindings = nullptr;
  ShaderResourceSet *scatter_bindings = nullptr;
  ShaderResourceSet *scatter_private_bindings = nullptr;
  const char *scatter_scope = nullptr;
  const char *scatter_private_scope = nullptr;
  if (use_private) {
    count_private_pipeline = cache.bucket_count_private_shared_i32_pipeline();
    prefix_chunks_pipeline = cache.bucket_prefix_chunks_i32_pipeline();
    scatter_private_pipeline =
        cache.bucket_scatter_private_pipeline(scatter_value_type);
    count_private_bindings =
        cache
            .bind_bucket_count_private_resource_set(
                this, keys_alloc, 0, key_bytes, partial_alloc,
                private_partial_bytes)
            .bindings;
    prefix_chunks_bindings =
        cache
            .bind_bucket_prefix_chunks_resource_set(
                this, partial_alloc, private_partial_bytes, offsets_alloc, 0,
                offset_bytes)
            .bindings;
    scatter_private_bindings =
        cache
            .bind_bucket_scatter_private_resource_set(
                this, scatter_value_type, keys_alloc, 0, key_bytes,
                values_alloc, 0, value_bytes, partial_alloc,
                private_partial_bytes, offsets_alloc, 0, offset_bytes,
                output_alloc, 0)
            .bindings;
    scatter_private_scope =
        cache.bucket_scatter_private_scope(scatter_value_type);
  } else {
    clear_pipeline = cache.bucket_clear_i32_pipeline();
    count_pipeline = cache.bucket_count_i32_pipeline();
    scatter_pipeline = cache.bucket_scatter_pipeline(scatter_value_type);
    clear_bindings =
        cache
            .bind_bucket_clear_resource_set(this, offsets_alloc, 0,
                                            offset_bytes, cursor_alloc, 0,
                                            cursor_bytes)
            .bindings;
    count_bindings =
        cache
            .bind_bucket_count_resource_set(this, keys_alloc, 0, key_bytes,
                                            offsets_alloc, 0, offset_bytes)
            .bindings;
    scatter_bindings =
        cache
            .bind_bucket_scatter_resource_set(
                this, scatter_value_type, keys_alloc, 0, key_bytes,
                values_alloc, 0, value_bytes, cursor_alloc, 0, cursor_bytes,
                output_alloc, 0)
            .bindings;
    scatter_scope = cache.bucket_scatter_scope(scatter_value_type);
  }
  const uint32_t private_groups = static_cast<uint32_t>(private_chunks);
  const uint32_t prefix_chunk_groups =
      static_cast<uint32_t>((num_bins + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  auto record_bucket =
      [keys_alloc, values_alloc, offsets_alloc, output_alloc, cursor_alloc,
       partial_alloc, key_bytes, value_bytes, offset_bytes, cursor_bytes,
       private_partial_bytes, item_groups, offset_groups, use_private,
       private_groups, prefix_chunk_groups, clear_pipeline, count_pipeline,
       count_private_pipeline, prefix_pipeline, prefix_chunks_pipeline,
       scatter_pipeline, scatter_private_pipeline, clear_bindings,
       count_bindings, count_private_bindings, prefix_bindings,
       prefix_chunks_bindings, scatter_bindings, scatter_private_bindings,
       scatter_scope, scatter_private_scope, profiler_scopes](
          Device * /*op_device*/, CommandList *cmdlist) {
        if (use_private) {
          dispatch_pipeline(
              cmdlist, count_private_pipeline, count_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? "vulkan_bucket_count_private_shared_i32"
                              : nullptr);
          cmdlist->buffer_barrier(partial_alloc);

          dispatch_pipeline(cmdlist, prefix_chunks_pipeline,
                            prefix_chunks_bindings, prefix_chunk_groups, 1, 1,
                            profiler_scopes
                                ? "vulkan_bucket_prefix_chunks_i32"
                                : nullptr);
          cmdlist->buffer_barrier(partial_alloc);
          cmdlist->buffer_barrier(offsets_alloc);

          dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                            profiler_scopes ? "vulkan_bucket_prefix_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
          cmdlist->buffer_barrier(cursor_alloc);

          dispatch_pipeline(
              cmdlist, scatter_private_pipeline, scatter_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? scatter_private_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
          return;
        }

        dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings,
                          offset_groups, 1, 1,
                          profiler_scopes ? "vulkan_bucket_clear_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          dispatch_pipeline(cmdlist, count_pipeline, count_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? "vulkan_bucket_count_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
        }

        dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                          profiler_scopes ? "vulkan_bucket_prefix_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          dispatch_pipeline(cmdlist, scatter_pipeline, scatter_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? scatter_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
        }
      };
  VulkanCommandReplayKey command_key;
  command_key.push(0);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(static_cast<uint64_t>(scatter_value_type));
  command_key.push(use_private ? 1 : 0);
  command_key.push(keys_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(keys_alloc));
  command_key.push(static_cast<uint64_t>(key_bytes));
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(static_cast<uint64_t>(value_bytes));
  command_key.push(offsets_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(offsets_alloc));
  command_key.push(static_cast<uint64_t>(offset_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(cursor_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(cursor_alloc));
  command_key.push(static_cast<uint64_t>(cursor_bytes));
  command_key.push(partial_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(partial_alloc));
  command_key.push(static_cast<uint64_t>(private_partial_bytes));
  command_key.push(item_groups);
  command_key.push(offset_groups);
  command_key.push(private_groups);
  command_key.push(prefix_chunk_groups);
  command_key.push_ptr(clear_pipeline);
  command_key.push_ptr(count_pipeline);
  command_key.push_ptr(count_private_pipeline);
  command_key.push_ptr(prefix_pipeline);
  command_key.push_ptr(prefix_chunks_pipeline);
  command_key.push_ptr(scatter_pipeline);
  command_key.push_ptr(scatter_private_pipeline);
  command_key.push_ptr(clear_bindings);
  command_key.push_ptr(count_bindings);
  command_key.push_ptr(count_private_bindings);
  command_key.push_ptr(prefix_bindings);
  command_key.push_ptr(prefix_chunks_bindings);
  command_key.push_ptr(scatter_bindings);
  command_key.push_ptr(scatter_private_bindings);
  if (!cache.bucket_ndarray_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_bucket)) {
    enqueue_compute_op_lambda(record_bucket, {});
  }
  return use_private ? cache.cached_bytes : 0;
}

std::size_t Program::vulkan_bucket_builder_dense_field(SNode *keys,
                                                       SNode *values,
                                                       SNode *offsets,
                                                       SNode *output,
                                                       Ndarray *cursor,
                                                       int value_type,
                                                       std::size_t n,
                                                       std::size_t num_bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field bucket builder is only available on "
              "Vulkan.");
  TI_ERROR_IF(!keys || !values || !offsets || !output || !cursor,
              "Vulkan native dense field bucket builder received a null input.");
  TI_ERROR_IF(num_bins == 0,
              "Vulkan native dense field bucket builder expects at least one "
              "bucket.");
  TI_ERROR_IF(cursor->shape.size() != 1 ||
                  cursor->get_nelement() < num_bins ||
                  cursor->get_element_size() != sizeof(int32_t),
              "Vulkan native dense field bucket builder cursor must be a 1D "
              "i32 ndarray with at least num_bins items.");
  TI_ERROR_IF(!vulkan_bucket_builder_value_type_available(value_type),
              "Vulkan native dense field bucket builder received an "
              "unsupported value type.");
  const size_t item_bytes = vulkan_scan_value_type_size(value_type);
  TI_ERROR_IF(item_bytes == 0,
              "Vulkan native dense field bucket builder received an "
              "unsupported value type.");
  DevicePtr keys_ptr = get_dense_field_device_ptr(keys);
  DevicePtr values_ptr = get_dense_field_device_ptr(values);
  DevicePtr offsets_ptr = get_dense_field_device_ptr(offsets);
  DevicePtr output_ptr = get_dense_field_device_ptr(output);
  const size_t keys_stride = get_dense_field_stride(keys, sizeof(int32_t));
  const size_t values_stride = get_dense_field_stride(values, item_bytes);
  const size_t offsets_stride =
      get_dense_field_stride(offsets, sizeof(int32_t));
  const size_t output_stride = get_dense_field_stride(output, item_bytes);
  TI_ERROR_IF(keys_stride != sizeof(int32_t) || values_stride != item_bytes ||
                  offsets_stride != sizeof(int32_t) ||
                  output_stride != item_bytes,
              "Vulkan native dense field bucket builder requires contiguous "
              "keys, values, offsets, and output fields.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_bins >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native dense field bucket builder input is too large for "
              "u32 dispatch.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(device == nullptr,
              "Vulkan native dense field bucket builder requires a compute "
              "device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation keys_alloc{keys_ptr.device, keys_ptr.alloc_id};
  const DeviceAllocation values_alloc{values_ptr.device, values_ptr.alloc_id};
  const DeviceAllocation offsets_alloc{offsets_ptr.device,
                                       offsets_ptr.alloc_id};
  const DeviceAllocation output_alloc{output_ptr.device, output_ptr.alloc_id};
  const DeviceAllocation cursor_alloc = cursor->ndarray_alloc_;
  const size_t key_bytes = n * sizeof(int32_t);
  const size_t value_bytes = n * item_bytes;
  const size_t offset_bytes = (num_bins + 1) * sizeof(int32_t);
  const size_t cursor_bytes = num_bins * sizeof(int32_t);
  const uint32_t item_groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t offset_groups = static_cast<uint32_t>(
      (num_bins + 1 + kBlockSize - 1) / kBlockSize);
  constexpr size_t kPrivateChunkSize = 2048;
  const size_t private_chunks =
      n == 0 ? 0 : (n + kPrivateChunkSize - 1) / kPrivateChunkSize;
  const size_t private_partial_bytes =
      private_chunks * num_bins * sizeof(int32_t);
  const int private_enabled =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE", 1);
  const int private_min_n_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MIN_N", 65536);
  const size_t private_min_n =
      private_min_n_config <= 0 ? 0 : static_cast<size_t>(private_min_n_config);
  const int private_max_bins_config =
      get_environ_config("TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BINS", 1024);
  const size_t private_max_bins =
      private_max_bins_config <= 0 ? 0
                                   : static_cast<size_t>(private_max_bins_config);
  const int private_max_bytes_config = get_environ_config(
      "TI_VULKAN_BUCKET_BUILDER_PRIVATE_MAX_BYTES", 4 * 1024 * 1024);
  const size_t private_max_bytes =
      private_max_bytes_config <= 0
          ? 0
          : static_cast<size_t>(private_max_bytes_config);
  const bool use_private =
      private_enabled != 0 && n >= private_min_n && n > 0 &&
      num_bins <= private_max_bins && num_bins <= 4096 &&
      private_partial_bytes > 0 && private_partial_bytes <= private_max_bytes;
  if (use_private && cache.needs_workspace_realloc(private_partial_bytes) &&
      cache.partial != kDeviceNullAllocation) {
    synchronize();
  }
  if (use_private) {
    cache.ensure_workspace(private_partial_bytes);
  }
  Pipeline *clear_pipeline = nullptr;
  Pipeline *count_pipeline = nullptr;
  Pipeline *count_private_pipeline = nullptr;
  Pipeline *prefix_pipeline = cache.bucket_prefix_i32_pipeline();
  Pipeline *prefix_chunks_pipeline = nullptr;
  Pipeline *scatter_pipeline = nullptr;
  Pipeline *scatter_private_pipeline = nullptr;
  const DeviceAllocation partial_alloc = cache.partial;
  ShaderResourceSet *clear_bindings = nullptr;
  ShaderResourceSet *count_bindings = nullptr;
  ShaderResourceSet *count_private_bindings = nullptr;
  ShaderResourceSet *prefix_bindings =
      cache
          .bind_bucket_prefix_resource_set(this, offsets_alloc,
                                           offsets_ptr.offset, offset_bytes,
                                           cursor_alloc, 0, cursor_bytes)
          .bindings;
  ShaderResourceSet *prefix_chunks_bindings = nullptr;
  ShaderResourceSet *scatter_bindings = nullptr;
  ShaderResourceSet *scatter_private_bindings = nullptr;
  const char *scatter_scope = nullptr;
  const char *scatter_private_scope = nullptr;
  if (use_private) {
    count_private_pipeline = cache.bucket_count_private_shared_i32_pipeline();
    prefix_chunks_pipeline = cache.bucket_prefix_chunks_i32_pipeline();
    scatter_private_pipeline =
        cache.bucket_scatter_private_pipeline(value_type);
    count_private_bindings =
        cache
            .bind_bucket_count_private_resource_set(
                this, keys_alloc, keys_ptr.offset, key_bytes, partial_alloc,
                private_partial_bytes)
            .bindings;
    prefix_chunks_bindings =
        cache
            .bind_bucket_prefix_chunks_resource_set(
                this, partial_alloc, private_partial_bytes, offsets_alloc,
                offsets_ptr.offset, offset_bytes)
            .bindings;
    scatter_private_bindings =
        cache
            .bind_bucket_scatter_private_resource_set(
                this, value_type, keys_alloc, keys_ptr.offset, key_bytes,
                values_alloc, values_ptr.offset, value_bytes, partial_alloc,
                private_partial_bytes, offsets_alloc, offsets_ptr.offset,
                offset_bytes, output_alloc, output_ptr.offset)
            .bindings;
    scatter_private_scope = cache.bucket_scatter_private_scope(value_type);
  } else {
    clear_pipeline = cache.bucket_clear_i32_pipeline();
    count_pipeline = cache.bucket_count_i32_pipeline();
    scatter_pipeline = cache.bucket_scatter_pipeline(value_type);
    clear_bindings =
        cache
            .bind_bucket_clear_resource_set(this, offsets_alloc,
                                            offsets_ptr.offset, offset_bytes,
                                            cursor_alloc, 0, cursor_bytes)
            .bindings;
    count_bindings =
        cache
            .bind_bucket_count_resource_set(
                this, keys_alloc, keys_ptr.offset, key_bytes, offsets_alloc,
                offsets_ptr.offset, offset_bytes)
            .bindings;
    scatter_bindings =
        cache
            .bind_bucket_scatter_resource_set(
                this, value_type, keys_alloc, keys_ptr.offset, key_bytes,
                values_alloc, values_ptr.offset, value_bytes, cursor_alloc, 0,
                cursor_bytes, output_alloc, output_ptr.offset)
            .bindings;
    scatter_scope = cache.bucket_scatter_scope(value_type);
  }
  const uint32_t private_groups = static_cast<uint32_t>(private_chunks);
  const uint32_t prefix_chunk_groups =
      static_cast<uint32_t>((num_bins + kBlockSize - 1) / kBlockSize);
  const bool profiler_scopes = profiler != nullptr;
  auto record_bucket =
      [keys_alloc, values_alloc, offsets_alloc, output_alloc, cursor_alloc,
       partial_alloc, key_bytes, value_bytes, offset_bytes, cursor_bytes,
       private_partial_bytes, item_groups, offset_groups, use_private,
       private_groups, prefix_chunk_groups, clear_pipeline, count_pipeline,
       count_private_pipeline, prefix_pipeline, prefix_chunks_pipeline,
       scatter_pipeline, scatter_private_pipeline, clear_bindings,
       count_bindings, count_private_bindings, prefix_bindings,
       prefix_chunks_bindings, scatter_bindings, scatter_private_bindings,
       scatter_scope, scatter_private_scope, keys_ptr, values_ptr, offsets_ptr,
       output_ptr, profiler_scopes](Device * /*op_device*/,
                                    CommandList *cmdlist) {
        if (use_private) {
          dispatch_pipeline(
              cmdlist, count_private_pipeline, count_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? "vulkan_bucket_count_private_shared_i32"
                              : nullptr);
          cmdlist->buffer_barrier(partial_alloc);

          dispatch_pipeline(cmdlist, prefix_chunks_pipeline,
                            prefix_chunks_bindings, prefix_chunk_groups, 1, 1,
                            profiler_scopes
                                ? "vulkan_bucket_prefix_chunks_i32"
                                : nullptr);
          cmdlist->buffer_barrier(partial_alloc);
          cmdlist->buffer_barrier(offsets_alloc);

          dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                            profiler_scopes ? "vulkan_bucket_prefix_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
          cmdlist->buffer_barrier(cursor_alloc);

          dispatch_pipeline(
              cmdlist, scatter_private_pipeline, scatter_private_bindings,
              private_groups, 1, 1,
              profiler_scopes ? scatter_private_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
          return;
        }

        dispatch_pipeline(cmdlist, clear_pipeline, clear_bindings,
                          offset_groups, 1, 1,
                          profiler_scopes ? "vulkan_bucket_clear_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          dispatch_pipeline(cmdlist, count_pipeline, count_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? "vulkan_bucket_count_i32"
                                            : nullptr);
          cmdlist->buffer_barrier(offsets_alloc);
        }

        dispatch_pipeline(cmdlist, prefix_pipeline, prefix_bindings, 1, 1, 1,
                          profiler_scopes ? "vulkan_bucket_prefix_i32"
                                          : nullptr);
        cmdlist->buffer_barrier(offsets_alloc);
        cmdlist->buffer_barrier(cursor_alloc);

        if (item_groups > 0) {
          dispatch_pipeline(cmdlist, scatter_pipeline, scatter_bindings,
                            item_groups, 1, 1,
                            profiler_scopes ? scatter_scope : nullptr);
          cmdlist->buffer_barrier(output_alloc);
        }
      };
  VulkanCommandReplayKey command_key;
  command_key.push(1);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(use_private ? 1 : 0);
  command_key.push(keys_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(keys_alloc));
  command_key.push(keys_ptr.offset);
  command_key.push(static_cast<uint64_t>(key_bytes));
  command_key.push(values_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(values_alloc));
  command_key.push(values_ptr.offset);
  command_key.push(static_cast<uint64_t>(value_bytes));
  command_key.push(offsets_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(offsets_alloc));
  command_key.push(offsets_ptr.offset);
  command_key.push(static_cast<uint64_t>(offset_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(output_ptr.offset);
  command_key.push(cursor_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(cursor_alloc));
  command_key.push(static_cast<uint64_t>(cursor_bytes));
  command_key.push(partial_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(partial_alloc));
  command_key.push(static_cast<uint64_t>(private_partial_bytes));
  command_key.push(item_groups);
  command_key.push(offset_groups);
  command_key.push(private_groups);
  command_key.push(prefix_chunk_groups);
  command_key.push_ptr(clear_pipeline);
  command_key.push_ptr(count_pipeline);
  command_key.push_ptr(count_private_pipeline);
  command_key.push_ptr(prefix_pipeline);
  command_key.push_ptr(prefix_chunks_pipeline);
  command_key.push_ptr(scatter_pipeline);
  command_key.push_ptr(scatter_private_pipeline);
  command_key.push_ptr(clear_bindings);
  command_key.push_ptr(count_bindings);
  command_key.push_ptr(count_private_bindings);
  command_key.push_ptr(prefix_bindings);
  command_key.push_ptr(prefix_chunks_bindings);
  command_key.push_ptr(scatter_bindings);
  command_key.push_ptr(scatter_private_bindings);
  if (!cache.bucket_dense_field_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes, record_bucket)) {
    enqueue_compute_op_lambda(record_bucket, {});
  }
  return use_private ? cache.cached_bytes : 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       Ndarray *offsets,
                                                       Ndarray *scratch,
                                                       Ndarray *cursor,
                                                       int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  return vulkan_grouped_reduce_ndarray(keys, values, output, offsets, scratch,
                                       cursor, 0, op);
}

std::size_t Program::vulkan_grouped_reduce_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *output,
                                                   Ndarray *offsets,
                                                   Ndarray *scratch,
                                                   Ndarray *cursor,
                                                   int value_type,
                                                   int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native grouped reduce is only available on Vulkan.");
  TI_ERROR_IF(!keys || !values || !output || !offsets || !scratch || !cursor,
              "Vulkan native grouped reduce received a null ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1 || values->shape.size() != 1 ||
                  output->shape.size() != 1 || offsets->shape.size() != 1 ||
                  scratch->shape.size() != 1 || cursor->shape.size() != 1,
              "Vulkan native grouped reduce expects 1D ndarrays.");
  TI_ERROR_IF(keys->get_nelement() != values->get_nelement(),
              "Vulkan native grouped reduce keys and values sizes differ.");
  TI_ERROR_IF(output->get_nelement() == 0,
              "Vulkan native grouped reduce output must contain at least one group.");
  const size_t n = keys->get_nelement();
  const size_t num_groups = output->get_nelement();
  TI_ERROR_IF(offsets->get_nelement() < num_groups + 1,
              "Vulkan native grouped reduce offsets must contain num_groups + 1 items.");
  TI_ERROR_IF(scratch->get_nelement() < n,
              "Vulkan native grouped reduce scratch is smaller than input values.");
  TI_ERROR_IF(cursor->get_nelement() < num_groups,
              "Vulkan native grouped reduce cursor is smaller than num_groups.");
  TI_ERROR_IF(!vulkan_grouped_reduce_value_type_available(value_type),
              "Vulkan native grouped reduce received an unsupported value type.");
  const bool value_is_64bit =
      value_type == 3 || value_type == 4 || value_type == 5;
  const size_t value_size =
      value_is_64bit ? sizeof(uint64_t) : sizeof(uint32_t);
  TI_ERROR_IF(keys->get_element_size() != sizeof(int32_t) ||
                  values->get_element_size() != value_size ||
                  output->get_element_size() != value_size ||
                  offsets->get_element_size() != sizeof(int32_t) ||
                  scratch->get_element_size() != value_size ||
                  cursor->get_element_size() != sizeof(int32_t),
              "Vulkan native grouped reduce value type or i32 metadata size mismatch.");
  TI_ERROR_IF(op != 0, "Vulkan native grouped reduce currently supports only sum.");
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                  num_groups >
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native grouped reduce input is too large for u32 dispatch.");

  std::size_t bucket_workspace =
      vulkan_bucket_builder_ndarray(keys, values, offsets, scratch, cursor,
                                    value_type);
  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native grouped reduce requires a compute device.");
  auto cache_lease = get_bucket_builder_cache(this, device);
  auto &cache = *cache_lease;
  const DeviceAllocation offsets_alloc = offsets->ndarray_alloc_;
  const DeviceAllocation scratch_alloc = scratch->ndarray_alloc_;
  const DeviceAllocation output_alloc = output->ndarray_alloc_;
  const size_t offset_bytes = (num_groups + 1) * sizeof(int32_t);
  const size_t scratch_bytes = n * value_size;
  const size_t output_bytes = num_groups * value_size;
  Pipeline *pipeline = cache.grouped_reduce_sum_pipeline(value_type);
  ShaderResourceSet *bindings =
      cache
          .bind_grouped_reduce_sum_resource_set(
              this, value_type, offsets_alloc, offset_bytes, scratch_alloc,
              scratch_bytes, output_alloc, output_bytes)
          .bindings;
  const char *reduce_scope = cache.grouped_reduce_sum_scope(value_type);
  TI_ERROR_IF(!pipeline,
              "Vulkan native grouped reduce could not find a sum pipeline.");
  const uint32_t groups = static_cast<uint32_t>(num_groups);
  const bool profiler_scopes = profiler != nullptr;
  auto record_grouped_reduce_sum =
      [offsets_alloc, scratch_alloc, output_alloc, offset_bytes, scratch_bytes,
       output_bytes, pipeline, bindings, groups, reduce_scope,
       profiler_scopes](Device * /*op_device*/, CommandList *cmdlist) {
        dispatch_pipeline(cmdlist, pipeline, bindings, groups, 1, 1,
                          profiler_scopes ? reduce_scope : nullptr);
        cmdlist->buffer_barrier(output_alloc);
      };
  VulkanCommandReplayKey command_key;
  command_key.push(23);
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(offsets_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(offsets_alloc));
  command_key.push(static_cast<uint64_t>(offset_bytes));
  command_key.push(scratch_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(scratch_alloc));
  command_key.push(static_cast<uint64_t>(scratch_bytes));
  command_key.push(output_alloc.alloc_id);
  command_key.push(vulkan_allocation_generation(output_alloc));
  command_key.push(static_cast<uint64_t>(output_bytes));
  command_key.push(groups);
  command_key.push_ptr(pipeline);
  command_key.push_ptr(bindings);
  if (!cache.grouped_reduce_sum_command_replay.submit_or_record(
          this, device, command_key, profiler_scopes,
          record_grouped_reduce_sum)) {
    enqueue_compute_op_lambda(record_grouped_reduce_sum, {});
  }
  return bucket_workspace;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type,
                                                   int value_type,
                                                   std::size_t key_offset,
                                                   std::size_t value_offset) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  const bool cpu_profile_enabled = vulkan_sort_cpu_profile_enabled();
  VulkanSortCpuProfileSample front_profile;
  VulkanSortCpuProfileSample *front =
      cpu_profile_enabled ? &front_profile : nullptr;
  const double total_start = front ? profile_time_us() : 0.0;
  if (front) {
    front->sort_calls++;
  }
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native radix sort is only available on Vulkan.");
  TI_ERROR_IF(!keys, "Vulkan native radix sort received null keys ndarray.");
  TI_ERROR_IF(keys->shape.size() != 1,
              "Vulkan native radix sort expects a 1D ndarray.");
  const size_t key_size = vulkan_sort_key_type_size(key_type);
  TI_ERROR_IF(key_size == 0,
              "Vulkan native radix sort received an unsupported key type.");
  TI_ERROR_IF(keys->get_element_size() != key_size,
              "Vulkan native radix sort key dtype does not match the "
              "requested key type.");
  const bool use_values = values != nullptr;
  const size_t expected_value_size =
      use_values ? vulkan_scan_value_type_size(value_type) : sizeof(uint32_t);
  size_t value_size = expected_value_size;
  if (use_values) {
    TI_ERROR_IF(values->shape.size() != 1,
                "Vulkan native radix sort values must be a 1D ndarray.");
    TI_ERROR_IF(values->get_nelement() != keys->get_nelement(),
                "Vulkan native radix sort keys and values must have the same "
                "length.");
    TI_ERROR_IF(expected_value_size == 0,
                "Vulkan native radix sort received an unsupported value type.");
    value_size = values->get_element_size();
    TI_ERROR_IF(value_size == 0 || value_size % sizeof(uint32_t) != 0,
                "Vulkan native radix sort value payload must be 4-byte aligned.");
  }

  const size_t n = keys->get_nelement();
  if (n <= 1) {
    if (front) {
      front->total_call_us += profile_time_us() - total_start;
      g_vulkan_sort_cpu_profile.merge(front_profile);
    }
    return 0;
  }
  TI_ERROR_IF(n > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
              "Vulkan native radix sort currently supports at most "
              "UINT32_MAX items.");

  Device *device = program_impl_->get_compute_device();
  TI_ERROR_IF(!device, "Vulkan native radix sort requires a compute device.");
  double start = front ? profile_time_us() : 0.0;
  auto cache_lease = get_cache(this, device);
  auto &cache = *cache_lease;
  if (front) {
    front->get_cache_us += profile_time_us() - start;
  }
  const bool use_radix8 = cache.radix8_enabled;
  const bool raw_value_payload =
      use_values && value_size != expected_value_size;
  const bool use_index_sort = key_type >= 2 || raw_value_payload;
  const bool workspace_use_values = use_values || use_index_sort;
  const size_t workspace_value_size =
      use_index_sort ? std::max(sizeof(uint32_t), value_size) : value_size;
  const size_t workspace_key_size =
      use_index_sort ? std::max(sizeof(uint32_t), key_size) : sizeof(uint32_t);
  const bool needs_realloc =
      cache.needs_workspace_realloc(n, workspace_use_values, use_radix8,
                                    workspace_value_size, workspace_key_size,
                                    use_index_sort);
  if (front && needs_realloc) {
    front->workspace_reallocs++;
  }
  if (cache.has_workspace_allocs() && needs_realloc) {
    if (front) {
      start = profile_time_us();
    }
    synchronize();
    if (front) {
      front->realloc_sync_calls++;
      front->realloc_sync_us += profile_time_us() - start;
    }
  }
  if (front) {
    start = profile_time_us();
  }
  cache.ensure_workspace(n, workspace_use_values, use_radix8,
                         workspace_value_size, workspace_key_size,
                         use_index_sort);
  if (front) {
    front->ensure_workspace_us += profile_time_us() - start;
  }

  const uint32_t groups =
      static_cast<uint32_t>((n + kBlockSize - 1) / kBlockSize);
  const uint32_t radix8_partitions =
      static_cast<uint32_t>((n + kRadix8PartitionSize - 1) /
                            kRadix8PartitionSize);
  const bool signed_keys = key_type == 1;
  const bool wide_keys = key_type >= 3;
  const bool profiler_scopes = profiler != nullptr;
  TI_ERROR_IF(key_offset % key_size != 0,
              "Vulkan native radix sort key offset must align to key type size.");
  TI_ERROR_IF(use_values && value_offset % value_size != 0,
              "Vulkan native radix sort value offset must align to value type size.");
  DeviceAllocation key_alloc = keys->ndarray_alloc_;
  DeviceAllocation value_alloc =
      use_values ? values->ndarray_alloc_ : kDeviceNullAllocation;
  const size_t key_bytes = n * sizeof(uint32_t);
  const size_t user_key_bytes = n * key_size;
  const size_t index_bytes = n * sizeof(uint32_t);
  const size_t value_bytes = n * value_size;
  const bool raw64_values = use_values && value_size == sizeof(uint64_t);
  const size_t table_bytes =
      static_cast<size_t>(groups) * kRadixBins * sizeof(uint32_t);
  const uint32_t chunk_groups =
      (groups + kBlockSize - 1) / kBlockSize;
  const size_t chunk_table_bytes =
      static_cast<size_t>(chunk_groups) * kRadixBins * sizeof(uint32_t);
  const bool inline_chunk_offsets =
      groups > kSingleChunkPrefixMaxBlocks &&
      chunk_groups <= kInlineChunkPrefixMaxChunks &&
      cache.inline_chunk_prefix_allowed;
  const size_t radix8_global_hist_bytes = kRadix8Bins * sizeof(uint32_t);
  const size_t radix8_partition_hist_bytes =
      static_cast<size_t>(radix8_partitions) * kRadix8Bins * sizeof(uint32_t);
  cache.ensure_sort_pipelines(device, key_type, use_values, raw64_values,
                              use_index_sort, use_radix8,
                              inline_chunk_offsets);

  std::shared_ptr<VulkanSortCpuProfileSample> lambda_profile;
  if (cpu_profile_enabled) {
    lambda_profile = std::make_shared<VulkanSortCpuProfileSample>();
  }
  if (front) {
    start = profile_time_us();
  }
  auto record_sort =
      [&, groups, n, key_type, signed_keys, wide_keys,
       use_index_sort, use_values, key_alloc, value_alloc, key_offset, value_offset, key_bytes,
       user_key_bytes, index_bytes, value_bytes, table_bytes, chunk_groups,
       chunk_table_bytes, raw64_values, inline_chunk_offsets, use_radix8,
       radix8_partitions,
       radix8_global_hist_bytes, radix8_partition_hist_bytes,
       lambda_profile, profiler_scopes](
          Device *op_device, CommandList *cmdlist) {
        auto scope_name = [profiler_scopes](const char *name) {
          return profiler_scopes ? name : nullptr;
        };
        VulkanSortCpuProfileSample *profile = lambda_profile.get();
        const double lambda_start = profile ? profile_time_us() : 0.0;
        if (profile) {
          profile->lambda_calls++;
        }
        auto dispatch_unary_cached =
            [&](Pipeline *pipeline,
                std::unique_ptr<ShaderResourceSet> &resource_set,
                VulkanRwBufferReplay<2> &replay,
                DeviceAllocation in,
                size_t in_offset,
                DeviceAllocation out,
                size_t out_offset,
                size_t bytes,
                uint32_t unary_groups,
                const char *scope) {
              ShaderResourceSet *bindings =
                  cache.cached_resource_set(resource_set, profile);
              profiled_replay_rw_buffer(replay, bindings, 0, in, in_offset,
                                        bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 1, out, out_offset,
                                        bytes, profile);
              dispatch_pipeline(cmdlist, pipeline, bindings, unary_groups, 1, 1,
                                scope_name(scope), profile);
            };
        auto gather_words_by_index = [&](DeviceAllocation src_alloc,
                                         size_t src_offset,
                                         size_t src_bytes,
                                         DeviceAllocation indices_alloc,
                                         DeviceAllocation dst_alloc,
                                         size_t dst_offset,
                                         size_t dst_bytes,
                                         const char *scope,
                                         std::unique_ptr<ShaderResourceSet>
                                             &resource_set,
                                         VulkanRwBufferReplay<3> &replay) {
          ShaderResourceSet *bindings =
              cache.cached_resource_set(resource_set, profile);
          profiled_replay_rw_buffer(replay, bindings, 0, src_alloc, src_offset,
                                    src_bytes, profile);
          profiled_replay_rw_buffer(replay, bindings, 1, indices_alloc, 0,
                                    index_bytes, profile);
          profiled_replay_rw_buffer(replay, bindings, 2, dst_alloc, dst_offset,
                                    dst_bytes, profile);
          const uint32_t word_groups = static_cast<uint32_t>(
              ((dst_bytes / sizeof(uint32_t)) + kBlockSize - 1) / kBlockSize);
          dispatch_pipeline(cmdlist, cache.gather_u32_by_u32.get(),
                            bindings, word_groups, 1, 1,
                            scope_name(scope), profile);
          profiled_buffer_barrier(cmdlist, dst_alloc, profile);
        };

        auto record_radix32_index_sort = [&]() {
          if (use_radix8) {
            DeviceAllocation key_read = cache.key_in;
            DeviceAllocation key_write = cache.key_out;
            DeviceAllocation value_read = cache.value_in;
            DeviceAllocation value_write = cache.value_out;
            for (int pass = 0; pass < 4; ++pass) {
              const uint32_t radix8_shift = static_cast<uint32_t>(pass * 8);
              profiled_buffer_fill(cmdlist,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, 0, profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              {
                ShaderResourceSet *bindings =
                    cache.cached_resource_set(
                        cache.radix8_upsweep_bindings[pass], profile);
                auto &replay = cache.radix8_upsweep_replay[pass];
                profiled_replay_rw_buffer(replay, bindings, 0, key_read, 0,
                                          key_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 1,
                                          cache.radix8_global_hist, 0,
                                          radix8_global_hist_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 2,
                                          cache.radix8_partition_hist, 0,
                                          radix8_partition_hist_bytes, profile);
                dispatch_pipeline_with_push_constants(
                    cmdlist, cache.radix8_upsweep.get(), bindings,
                    &radix8_shift, sizeof(radix8_shift), radix8_partitions, 1,
                    1, scope_name("vulkan_sort_radix8_upsweep"), profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                        profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                        profile);
              }
              {
                const bool init_static_bindings =
                    !cache.radix8_spine_bindings;
                ShaderResourceSet *bindings =
                    cache.cached_resource_set(cache.radix8_spine_bindings,
                                              profile);
                if (init_static_bindings) {
                  profiled_rw_buffer(bindings, 0,
                                     cache.radix8_global_hist.get_ptr(0),
                                     radix8_global_hist_bytes, profile);
                  profiled_rw_buffer(bindings, 1,
                                     cache.radix8_partition_hist.get_ptr(0),
                                     radix8_partition_hist_bytes, profile);
                }
                dispatch_pipeline(cmdlist, cache.radix8_spine.get(), bindings,
                                  kRadix8Bins, 1, 1,
                                  scope_name("vulkan_sort_radix8_spine"),
                                  profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                        profile);
                profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                        profile);
              }
              {
                ShaderResourceSet *bindings = cache.cached_resource_set(
                    cache.radix8_downsweep_pairs_bindings[pass], profile);
                auto &replay = cache.radix8_downsweep_pairs_replay[pass];
                profiled_replay_rw_buffer(replay, bindings, 0, key_read, 0,
                                          key_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 1, key_write, 0,
                                          key_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 2,
                                          cache.radix8_global_hist, 0,
                                          radix8_global_hist_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 3,
                                          cache.radix8_partition_hist, 0,
                                          radix8_partition_hist_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 4, value_read, 0,
                                          index_bytes, profile);
                profiled_replay_rw_buffer(replay, bindings, 5, value_write, 0,
                                          index_bytes, profile);
                dispatch_pipeline_with_push_constants(
                    cmdlist, cache.radix8_downsweep_pairs.get(), bindings,
                    &radix8_shift, sizeof(radix8_shift), radix8_partitions, 1,
                    1, scope_name("vulkan_sort_radix8_downsweep_pairs"),
                    profile);
                profiled_buffer_barrier(cmdlist, key_write, profile);
                profiled_buffer_barrier(cmdlist, value_write, profile);
              }
              std::swap(key_read, key_write);
              std::swap(value_read, value_write);
            }
            return;
          }

          DeviceAllocation key_read = cache.key_in;
          DeviceAllocation key_write = cache.key_out;
          DeviceAllocation value_read = cache.value_in;
          DeviceAllocation value_write = cache.value_out;
          for (int pass = 0; pass < 8; ++pass) {
            {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
              bindings->rw_buffer(1, cache.rank.get_ptr(0), key_bytes);
              bindings->rw_buffer(2, cache.hist.get_ptr(0), table_bytes);
              Pipeline *rank_pipeline =
                  cache.subgroup_rank_enabled
                      ? cache.rank_hist_subgroup[pass].get()
                      : cache.rank_hist[pass].get();
              const char *rank_scope =
                  cache.subgroup_rank_enabled ? "vulkan_sort_rank_hist_subgroup"
                                              : "vulkan_sort_rank_hist";
              dispatch_pipeline(cmdlist, rank_pipeline, bindings.get(), groups,
                                1, 1, scope_name(rank_scope), profile);
              profiled_buffer_barrier(cmdlist, cache.rank, profile);
              profiled_buffer_barrier(cmdlist, cache.hist, profile);
            }
            if (groups <= kSingleChunkPrefixMaxBlocks) {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
              bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
              bindings->rw_buffer(2, cache.chunk_offsets.get_ptr(0),
                                  chunk_table_bytes);
              dispatch_pipeline(cmdlist, cache.prefix_single_chunk.get(),
                                bindings.get(), 1, kRadixBins, 1,
                                scope_name("vulkan_sort_prefix_single_chunk"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.offsets, profile);
              profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
            } else {
              {
                auto bindings = op_device->create_resource_set_unique();
                bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
                bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
                bindings->rw_buffer(2, cache.chunk_sums.get_ptr(0),
                                    chunk_table_bytes);
                dispatch_pipeline(cmdlist, cache.prefix_block.get(),
                                  bindings.get(), chunk_groups, kRadixBins, 1,
                                  scope_name("vulkan_sort_prefix_block"),
                                  profile);
                profiled_buffer_barrier(cmdlist, cache.offsets, profile);
                profiled_buffer_barrier(cmdlist, cache.chunk_sums, profile);
              }
              if (!inline_chunk_offsets) {
                auto bindings = op_device->create_resource_set_unique();
                bindings->rw_buffer(0, cache.chunk_sums.get_ptr(0),
                                    chunk_table_bytes);
                bindings->rw_buffer(1, cache.chunk_offsets.get_ptr(0),
                                    chunk_table_bytes);
                dispatch_pipeline(cmdlist, cache.prefix_chunks.get(),
                                  bindings.get(), 1, 1, 1,
                                  scope_name("vulkan_sort_prefix_chunks"),
                                  profile);
                profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
              }
            }
            {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
              bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
              bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
              bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
              bindings->rw_buffer(4, value_read.get_ptr(0), index_bytes);
              bindings->rw_buffer(5, value_write.get_ptr(0), index_bytes);
              bindings->rw_buffer(6,
                                  (inline_chunk_offsets ? cache.chunk_sums
                                                        : cache.chunk_offsets)
                                      .get_ptr(0),
                                  chunk_table_bytes);
              Pipeline *scatter_pipeline =
                  inline_chunk_offsets
                      ? cache.scatter_pairs_inline_chunks[pass].get()
                      : cache.scatter_pairs[pass].get();
              const char *scatter_scope =
                  inline_chunk_offsets
                      ? "vulkan_sort_scatter_pairs_inline_chunks"
                      : "vulkan_sort_scatter_pairs";
              dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                                groups, 1, 1, scope_name(scatter_scope),
                                profile);
              profiled_buffer_barrier(cmdlist, key_write, profile);
              profiled_buffer_barrier(cmdlist, value_write, profile);
            }
            std::swap(key_read, key_write);
            std::swap(value_read, value_write);
          }
        };

        if (use_index_sort) {
          const double radix_start = profile ? profile_time_us() : 0.0;
          Pipeline *init_pipeline = cache.sort_init_index_pipeline(key_type);
          {
            ShaderResourceSet *bindings = cache.cached_resource_set(
                cache.sort_init_index_bindings[key_type], profile);
            auto &replay = cache.sort_init_index_replay[key_type];
            profiled_replay_rw_buffer(replay, bindings, 0, key_alloc,
                                      key_offset, user_key_bytes, profile);
            profiled_replay_rw_buffer(replay, bindings, 1, cache.key_in, 0,
                                      key_bytes, profile);
            profiled_replay_rw_buffer(replay, bindings, 2, cache.key_high, 0,
                                      key_bytes, profile);
            profiled_replay_rw_buffer(replay, bindings, 3, cache.value_in, 0,
                                      index_bytes, profile);
            dispatch_pipeline(cmdlist, init_pipeline, bindings, groups,
                              1, 1,
                              scope_name("vulkan_sort_init_key_index"),
                              profile);
            profiled_buffer_barrier(cmdlist, cache.key_in, profile);
            profiled_buffer_barrier(cmdlist, cache.value_in, profile);
            if (wide_keys) {
              profiled_buffer_barrier(cmdlist, cache.key_high, profile);
            }
          }

          record_radix32_index_sort();

          if (wide_keys) {
            gather_words_by_index(cache.key_high, 0, key_bytes, cache.value_in,
                                  cache.key_in, 0, key_bytes,
                                  "vulkan_sort_gather_high32",
                                  cache.gather_high32_bindings,
                                  cache.gather_high32_replay);
            record_radix32_index_sort();
          }

          gather_words_by_index(key_alloc, key_offset, user_key_bytes, cache.value_in,
                                cache.key_out, 0, user_key_bytes,
                                "vulkan_sort_gather_keys",
                                cache.gather_keys_bindings,
                                cache.gather_keys_replay);
          profiled_buffer_copy(cmdlist, key_alloc.get_ptr(key_offset),
                               cache.key_out.get_ptr(0), user_key_bytes,
                               profile);
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
          if (use_values) {
            gather_words_by_index(value_alloc, value_offset, value_bytes, cache.value_in,
                                  cache.value_out, 0, value_bytes,
                                  "vulkan_sort_gather_values",
                                  cache.gather_values_bindings,
                                  cache.gather_values_replay);
            profiled_buffer_copy(cmdlist, value_alloc.get_ptr(value_offset),
                                 cache.value_out.get_ptr(0), value_bytes,
                                 profile);
            profiled_buffer_barrier(cmdlist, value_alloc, profile);
          }
          if (profile) {
            if (use_radix8) {
              profile->radix8_body_us += profile_time_us() - radix_start;
            }
            profile->lambda_total_us += profile_time_us() - lambda_start;
            g_vulkan_sort_cpu_profile.merge(*profile);
          }
          return;
        }

        if (use_radix8) {
          const double radix8_start = profile ? profile_time_us() : 0.0;
          DeviceAllocation key_read = signed_keys ? cache.key_in : key_alloc;
          DeviceAllocation key_write = cache.key_out;
          DeviceAllocation value_read =
              use_values ? value_alloc : cache.value_in;
          DeviceAllocation value_write = cache.value_out;
          size_t key_read_offset = signed_keys ? 0 : key_offset;
          size_t key_write_offset = 0;
          size_t value_read_offset = use_values ? value_offset : 0;
          size_t value_write_offset = 0;
          if (signed_keys) {
            dispatch_unary_cached(cache.init_i32.get(),
                                  cache.init_i32_bindings,
                                  cache.init_i32_replay, key_alloc, key_offset,
                                  cache.key_in, 0, key_bytes, groups,
                                  "vulkan_sort_init_i32");
            profiled_buffer_barrier(cmdlist, cache.key_in, profile);
          }
          bool keys_written_to_user = false;
          bool values_written_to_user = false;
          for (int pass = 0; pass < 4; ++pass) {
            const uint32_t radix8_shift = static_cast<uint32_t>(pass * 8);
            const bool last_pass = (pass == 3);
            const bool direct_key_output = last_pass && !signed_keys;
            const bool direct_value_output = last_pass && use_values;
            DeviceAllocation pass_key_write =
                direct_key_output ? key_alloc : key_write;
            DeviceAllocation pass_value_write =
                direct_value_output ? value_alloc : value_write;
            size_t pass_key_write_offset =
                direct_key_output ? key_offset : key_write_offset;
            size_t pass_value_write_offset =
                direct_value_output ? value_offset : value_write_offset;
            profiled_buffer_fill(cmdlist, cache.radix8_global_hist.get_ptr(0),
                                 radix8_global_hist_bytes, 0, profile);
            profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                    profile);
            {
              ShaderResourceSet *bindings =
                  cache.cached_resource_set(cache.radix8_upsweep_bindings[pass],
                                            profile);
              auto &replay = cache.radix8_upsweep_replay[pass];
              profiled_replay_rw_buffer(replay, bindings, 0, key_read,
                                        key_read_offset, key_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 1,
                                        cache.radix8_global_hist, 0,
                                        radix8_global_hist_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 2,
                                        cache.radix8_partition_hist, 0,
                                        radix8_partition_hist_bytes, profile);
              dispatch_pipeline_with_push_constants(
                  cmdlist, cache.radix8_upsweep.get(), bindings,
                  &radix8_shift, sizeof(radix8_shift), radix8_partitions, 1, 1,
                  scope_name("vulkan_sort_radix8_upsweep"), profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                      profile);
            }
            {
              const bool init_static_bindings = !cache.radix8_spine_bindings;
              ShaderResourceSet *bindings =
                  cache.cached_resource_set(cache.radix8_spine_bindings,
                                            profile);
              if (init_static_bindings) {
                profiled_rw_buffer(bindings, 0,
                                   cache.radix8_global_hist.get_ptr(0),
                                   radix8_global_hist_bytes, profile);
                profiled_rw_buffer(bindings, 1,
                                   cache.radix8_partition_hist.get_ptr(0),
                                   radix8_partition_hist_bytes, profile);
              }
              dispatch_pipeline(cmdlist, cache.radix8_spine.get(),
                                bindings, kRadix8Bins, 1, 1,
                                scope_name("vulkan_sort_radix8_spine"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_global_hist,
                                      profile);
              profiled_buffer_barrier(cmdlist, cache.radix8_partition_hist,
                                      profile);
            }
            if (use_values) {
              ShaderResourceSet *bindings = cache.cached_resource_set(
                  cache.radix8_downsweep_pairs_bindings[pass], profile);
              auto &replay = cache.radix8_downsweep_pairs_replay[pass];
              profiled_replay_rw_buffer(replay, bindings, 0, key_read,
                                        key_read_offset, key_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 1, pass_key_write,
                                        pass_key_write_offset, key_bytes,
                                        profile);
              profiled_replay_rw_buffer(replay, bindings, 2,
                                        cache.radix8_global_hist, 0,
                                        radix8_global_hist_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 3,
                                        cache.radix8_partition_hist, 0,
                                        radix8_partition_hist_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 4, value_read,
                                        value_read_offset, value_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 5, pass_value_write,
                                        pass_value_write_offset, value_bytes,
                                        profile);
              dispatch_pipeline_with_push_constants(
                  cmdlist,
                  (raw64_values ? cache.radix8_downsweep_pairs_raw64.get()
                                : cache.radix8_downsweep_pairs.get()),
                  bindings, &radix8_shift, sizeof(radix8_shift),
                  radix8_partitions, 1, 1,
                  scope_name("vulkan_sort_radix8_downsweep_pairs"), profile);
              profiled_buffer_barrier(cmdlist, pass_key_write, profile);
              profiled_buffer_barrier(cmdlist, pass_value_write, profile);
              if (direct_value_output) {
                value_read = pass_value_write;
                value_read_offset = pass_value_write_offset;
                values_written_to_user = true;
              } else {
                std::swap(value_read, value_write);
                std::swap(value_read_offset, value_write_offset);
              }
            } else {
              ShaderResourceSet *bindings = cache.cached_resource_set(
                  cache.radix8_downsweep_keys_bindings[pass], profile);
              auto &replay = cache.radix8_downsweep_keys_replay[pass];
              profiled_replay_rw_buffer(replay, bindings, 0, key_read,
                                        key_read_offset, key_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 1, pass_key_write,
                                        pass_key_write_offset, key_bytes,
                                        profile);
              profiled_replay_rw_buffer(replay, bindings, 2,
                                        cache.radix8_global_hist, 0,
                                        radix8_global_hist_bytes, profile);
              profiled_replay_rw_buffer(replay, bindings, 3,
                                        cache.radix8_partition_hist, 0,
                                        radix8_partition_hist_bytes, profile);
              dispatch_pipeline_with_push_constants(
                  cmdlist, cache.radix8_downsweep_keys.get(), bindings,
                  &radix8_shift, sizeof(radix8_shift), radix8_partitions, 1, 1,
                  scope_name("vulkan_sort_radix8_downsweep_keys"), profile);
              profiled_buffer_barrier(cmdlist, pass_key_write, profile);
            }
            if (direct_key_output) {
              key_read = pass_key_write;
              key_read_offset = pass_key_write_offset;
              keys_written_to_user = true;
            } else {
              std::swap(key_read, key_write);
              std::swap(key_read_offset, key_write_offset);
            }
          }

          if (signed_keys) {
            dispatch_unary_cached(cache.copy_i32.get(),
                                  cache.copy_i32_bindings,
                                  cache.copy_i32_replay, key_read,
                                  key_read_offset, key_alloc, key_offset,
                                  key_bytes, groups, "vulkan_sort_copy_i32");
            profiled_buffer_barrier(cmdlist, key_alloc, profile);
          } else if (!keys_written_to_user) {
            profiled_buffer_copy(cmdlist, key_alloc.get_ptr(key_offset),
                                 key_read.get_ptr(key_read_offset), key_bytes,
                                 profile);
            profiled_buffer_barrier(cmdlist, key_alloc, profile);
          }
          if (use_values && !values_written_to_user) {
            profiled_buffer_copy(cmdlist, value_alloc.get_ptr(value_offset),
                                 value_read.get_ptr(value_read_offset),
                                 value_bytes, profile);
            profiled_buffer_barrier(cmdlist, value_alloc, profile);
          }
          if (profile) {
            profile->radix8_body_us += profile_time_us() - radix8_start;
            profile->lambda_total_us += profile_time_us() - lambda_start;
            g_vulkan_sort_cpu_profile.merge(*profile);
          }
          return;
        }
        if (signed_keys) {
          dispatch_unary_cached(cache.init_i32.get(), cache.init_i32_bindings,
                                cache.init_i32_replay, key_alloc, key_offset,
                                cache.key_in, 0, key_bytes, groups,
                                "vulkan_sort_init_i32");
          profiled_buffer_barrier(cmdlist, cache.key_in, profile);
        } else {
          profiled_buffer_copy(cmdlist, cache.key_in.get_ptr(0),
                               key_alloc.get_ptr(key_offset), key_bytes,
                               profile);
          profiled_buffer_barrier(cmdlist, cache.key_in, profile);
        }
        if (use_values) {
          profiled_buffer_copy(cmdlist, cache.value_in.get_ptr(0),
                               value_alloc.get_ptr(value_offset), value_bytes,
                               profile);
          profiled_buffer_barrier(cmdlist, cache.value_in, profile);
        }

        DeviceAllocation key_read = cache.key_in;
        DeviceAllocation key_write = cache.key_out;
        DeviceAllocation value_read = cache.value_in;
        DeviceAllocation value_write = cache.value_out;
        for (int pass = 0; pass < 8; ++pass) {
          {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.hist.get_ptr(0), table_bytes);
            Pipeline *rank_pipeline =
                cache.subgroup_rank_enabled
                    ? cache.rank_hist_subgroup[pass].get()
                    : cache.rank_hist[pass].get();
            const char *rank_scope =
                cache.subgroup_rank_enabled ? "vulkan_sort_rank_hist_subgroup"
                                            : "vulkan_sort_rank_hist";
            dispatch_pipeline(cmdlist, rank_pipeline, bindings.get(), groups,
                              1, 1, scope_name(rank_scope), profile);
            profiled_buffer_barrier(cmdlist, cache.rank, profile);
            profiled_buffer_barrier(cmdlist, cache.hist, profile);
          }
          if (groups <= kSingleChunkPrefixMaxBlocks) {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
            bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(2, cache.chunk_offsets.get_ptr(0),
                                chunk_table_bytes);
            dispatch_pipeline(cmdlist, cache.prefix_single_chunk.get(),
                              bindings.get(), 1, kRadixBins, 1,
                              scope_name("vulkan_sort_prefix_single_chunk"),
                              profile);
            profiled_buffer_barrier(cmdlist, cache.offsets, profile);
            profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
          } else {
            {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, cache.hist.get_ptr(0), table_bytes);
              bindings->rw_buffer(1, cache.offsets.get_ptr(0), table_bytes);
              bindings->rw_buffer(2, cache.chunk_sums.get_ptr(0),
                                  chunk_table_bytes);
              dispatch_pipeline(cmdlist, cache.prefix_block.get(),
                                bindings.get(), chunk_groups, kRadixBins, 1,
                                scope_name("vulkan_sort_prefix_block"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.offsets, profile);
              profiled_buffer_barrier(cmdlist, cache.chunk_sums, profile);
            }
            if (!inline_chunk_offsets) {
              auto bindings = op_device->create_resource_set_unique();
              bindings->rw_buffer(0, cache.chunk_sums.get_ptr(0),
                                  chunk_table_bytes);
              bindings->rw_buffer(1, cache.chunk_offsets.get_ptr(0),
                                  chunk_table_bytes);
              dispatch_pipeline(cmdlist, cache.prefix_chunks.get(),
                                bindings.get(), 1, 1, 1,
                                scope_name("vulkan_sort_prefix_chunks"),
                                profile);
              profiled_buffer_barrier(cmdlist, cache.chunk_offsets, profile);
            }
          }
          if (use_values) {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(4, value_read.get_ptr(0), value_bytes);
            bindings->rw_buffer(5, value_write.get_ptr(0), value_bytes);
            bindings->rw_buffer(6,
                                (inline_chunk_offsets ? cache.chunk_sums
                                                      : cache.chunk_offsets)
                                    .get_ptr(0),
                                chunk_table_bytes);
            Pipeline *scatter_pipeline =
                raw64_values
                    ? (inline_chunk_offsets
                           ? cache.scatter_pairs_inline_chunks_raw64[pass].get()
                           : cache.scatter_pairs_raw64[pass].get())
                    : (inline_chunk_offsets
                           ? cache.scatter_pairs_inline_chunks[pass].get()
                           : cache.scatter_pairs[pass].get());
            const char *scatter_scope =
                inline_chunk_offsets ? "vulkan_sort_scatter_pairs_inline_chunks"
                                     : "vulkan_sort_scatter_pairs";
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              groups, 1, 1, scope_name(scatter_scope), profile);
            profiled_buffer_barrier(cmdlist, key_write, profile);
            profiled_buffer_barrier(cmdlist, value_write, profile);
            std::swap(value_read, value_write);
          } else {
            auto bindings = op_device->create_resource_set_unique();
            bindings->rw_buffer(0, key_read.get_ptr(0), key_bytes);
            bindings->rw_buffer(1, key_write.get_ptr(0), key_bytes);
            bindings->rw_buffer(2, cache.rank.get_ptr(0), key_bytes);
            bindings->rw_buffer(3, cache.offsets.get_ptr(0), table_bytes);
            bindings->rw_buffer(4,
                                (inline_chunk_offsets ? cache.chunk_sums
                                                      : cache.chunk_offsets)
                                    .get_ptr(0),
                                chunk_table_bytes);
            Pipeline *scatter_pipeline =
                inline_chunk_offsets
                    ? cache.scatter_keys_inline_chunks[pass].get()
                    : cache.scatter_keys[pass].get();
            const char *scatter_scope =
                inline_chunk_offsets ? "vulkan_sort_scatter_keys_inline_chunks"
                                     : "vulkan_sort_scatter_keys";
            dispatch_pipeline(cmdlist, scatter_pipeline, bindings.get(),
                              groups, 1, 1, scope_name(scatter_scope), profile);
            profiled_buffer_barrier(cmdlist, key_write, profile);
          }
          std::swap(key_read, key_write);
        }

        if (signed_keys) {
          dispatch_unary_cached(cache.copy_i32.get(), cache.copy_i32_bindings,
                                cache.copy_i32_replay, key_read, 0,
                                key_alloc, key_offset, key_bytes, groups,
                                "vulkan_sort_copy_i32");
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
        } else {
          profiled_buffer_copy(cmdlist, key_alloc.get_ptr(key_offset),
                               key_read.get_ptr(0), key_bytes, profile);
          profiled_buffer_barrier(cmdlist, key_alloc, profile);
        }
        if (use_values) {
          profiled_buffer_copy(cmdlist, value_alloc.get_ptr(value_offset),
                               value_read.get_ptr(0), value_bytes, profile);
          profiled_buffer_barrier(cmdlist, value_alloc, profile);
        }
        if (profile) {
          profile->lambda_total_us += profile_time_us() - lambda_start;
          g_vulkan_sort_cpu_profile.merge(*profile);
        }
      };
  VulkanCommandReplayKey command_key;
  command_key.push(80);
  command_key.push(static_cast<uint64_t>(n));
  command_key.push(static_cast<uint64_t>(key_type));
  command_key.push(static_cast<uint64_t>(value_type));
  command_key.push(use_index_sort ? 1 : 0);
  command_key.push(use_values ? 1 : 0);
  command_key.push(signed_keys ? 1 : 0);
  command_key.push(wide_keys ? 1 : 0);
  command_key.push(raw64_values ? 1 : 0);
  command_key.push(use_radix8 ? 1 : 0);
  command_key.push(inline_chunk_offsets ? 1 : 0);
  command_key.push(groups);
  command_key.push(radix8_partitions);
  command_key.push(chunk_groups);
  command_key.push(static_cast<uint64_t>(key_bytes));
  command_key.push(static_cast<uint64_t>(user_key_bytes));
  command_key.push(static_cast<uint64_t>(index_bytes));
  command_key.push(static_cast<uint64_t>(value_bytes));
  command_key.push(static_cast<uint64_t>(table_bytes));
  command_key.push(static_cast<uint64_t>(chunk_table_bytes));
  command_key.push(static_cast<uint64_t>(radix8_global_hist_bytes));
  command_key.push(static_cast<uint64_t>(radix8_partition_hist_bytes));
  push_vulkan_command_key_range(command_key, key_alloc, key_offset,
                                user_key_bytes);
  push_vulkan_command_key_range(command_key, value_alloc, value_offset,
                                use_values ? value_bytes : 0);
  push_vulkan_command_key_range(command_key, cache.key_in, 0,
                                cache.capacity * cache.key_bytes_per_item);
  push_vulkan_command_key_range(command_key, cache.key_out, 0,
                                cache.capacity * cache.key_bytes_per_item);
  push_vulkan_command_key_range(command_key, cache.rank, 0,
                                cache.capacity * sizeof(uint32_t));
  push_vulkan_command_key_range(command_key, cache.hist, 0, table_bytes);
  push_vulkan_command_key_range(command_key, cache.offsets, 0, table_bytes);
  push_vulkan_command_key_range(command_key, cache.chunk_sums, 0,
                                chunk_table_bytes);
  push_vulkan_command_key_range(command_key, cache.chunk_offsets, 0,
                                chunk_table_bytes);
  push_vulkan_command_key_range(command_key, cache.radix8_global_hist, 0,
                                radix8_global_hist_bytes);
  push_vulkan_command_key_range(command_key, cache.radix8_partition_hist, 0,
                                radix8_partition_hist_bytes);
  push_vulkan_command_key_range(command_key, cache.value_in, 0,
                                cache.capacity * cache.value_bytes_per_item);
  push_vulkan_command_key_range(command_key, cache.value_out, 0,
                                cache.capacity * cache.value_bytes_per_item);
  push_vulkan_command_key_range(command_key, cache.key_high, 0,
                                cache.capacity * sizeof(uint32_t));
  const bool command_replay_profiler_disabled =
      profiler_scopes || cpu_profile_enabled;
  if (!cache.command_replay.submit_or_record(this, device, command_key,
                                             command_replay_profiler_disabled,
                                             record_sort)) {
    program_impl_->enqueue_compute_op_lambda(record_sort, {});
  }
  if (front) {
    front->enqueue_us += profile_time_us() - start;
  }
  if (get_environ_config("TI_VULKAN_SORT_INTERNAL_SYNC", 0) != 0) {
    if (front) {
      start = profile_time_us();
    }
    synchronize();
    if (front) {
      front->internal_sync_calls++;
      front->internal_sync_us += profile_time_us() - start;
    }
  }
  if (front) {
    front->total_call_us += profile_time_us() - total_start;
    g_vulkan_sort_cpu_profile.merge(front_profile);
  }
  return cache.cached_bytes;
}

std::size_t Program::vulkan_radix_sort_u32_dense_field(SNode *keys,
                                                       SNode *values,
                                                       int key_type,
                                                       int value_type,
                                                       std::size_t n) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan native dense field radix sort is only available on Vulkan.");
  TI_ERROR_IF(!keys,
              "Vulkan native dense field radix sort received null keys field.");
  TI_ERROR_IF(n > static_cast<std::size_t>(std::numeric_limits<int>::max()),
              "Vulkan native dense field radix sort currently supports at most INT_MAX items.");
  const size_t key_size = vulkan_sort_key_type_size(key_type);
  DataType key_dtype = vulkan_sort_key_data_type(key_type);
  TI_ERROR_IF(key_size == 0 || static_cast<const Type *>(key_dtype) == nullptr,
              "Vulkan native dense field radix sort received an unsupported key type.");
  DevicePtr key_ptr = get_dense_field_device_ptr(keys);
  const size_t key_stride = get_dense_field_stride(keys, key_size);
  TI_ERROR_IF(key_stride != key_size,
              "Vulkan native dense field radix sort requires contiguous keys.");

  DeviceAllocation key_alloc{key_ptr.device, key_ptr.alloc_id};
  Ndarray key_view(key_alloc, key_dtype, {static_cast<int>(n)});

  if (!values) {
    return vulkan_radix_sort_u32_ndarray(&key_view, nullptr, key_type, 0,
                                         key_ptr.offset, 0);
  }

  const size_t value_size = vulkan_scan_value_type_size(value_type);
  DataType value_dtype = vulkan_sort_value_data_type(value_type);
  TI_ERROR_IF(value_size == 0 || static_cast<const Type *>(value_dtype) == nullptr,
              "Vulkan native dense field radix sort received an unsupported value type.");
  DevicePtr value_ptr = get_dense_field_device_ptr(values);
  const size_t value_stride = get_dense_field_stride(values, value_size);
  TI_ERROR_IF(value_stride != value_size,
              "Vulkan native dense field radix sort requires contiguous values.");
  DeviceAllocation value_alloc{value_ptr.device, value_ptr.alloc_id};
  Ndarray value_view(value_alloc, value_dtype, {static_cast<int>(n)});
  return vulkan_radix_sort_u32_ndarray(&key_view, &value_view, key_type,
                                       value_type, key_ptr.offset,
                                       value_ptr.offset);
}

void Program::vulkan_radix_sort_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::ordering);
}

void Program::vulkan_scan_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::scan);
}

void Program::vulkan_compact_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::compact);
}

void Program::vulkan_histogram_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::histogram);
}

void Program::vulkan_reduce_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::reduce);
}

void Program::vulkan_check_count_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::check);
}

void Program::vulkan_metric_reduce_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::metric);
}

void Program::vulkan_sparse_algebra_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::sparse_algebra);
}

void Program::vulkan_transform_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::transform);
}

void Program::vulkan_add_merge_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::scatter_add);
}

void Program::vulkan_indexed_copy_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::indexed);
}

void Program::vulkan_scatter_add_clear_workspace() {
  vulkan_indexed_copy_clear_workspace();
}

void Program::vulkan_bucket_builder_clear_workspace() {
  clear_primitive_workspaces_for(PrimitiveWorkspaceBackend::vulkan,
                                 PrimitiveWorkspaceFamily::bucket);
}

void Program::vulkan_grouped_reduce_clear_workspace() {
  vulkan_bucket_builder_clear_workspace();
}

void Program::vulkan_clear_primitive_caches() {
  if (primitive_workspace_arena_
          .snapshot(PrimitiveWorkspaceBackend::vulkan)
          .entries == 0) {
    return;
  }
  // A fatal Vulkan error invalidates the wait/submission path. Cache entries
  // must still be erased while their Device owner exists; their RHI wrappers
  // can destroy handles without attempting to recover the lost device.
  if (!runtime_has_fatal_fault()) {
    synchronize();
  }
  primitive_workspace_arena_.clear(PrimitiveWorkspaceBackend::vulkan);
}

std::size_t Program::vulkan_radix_sort_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::ordering)
      .reserved_bytes;
}

std::size_t Program::vulkan_scan_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::scan)
      .reserved_bytes;
}

std::size_t Program::vulkan_compact_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::compact)
      .reserved_bytes;
}

std::size_t Program::vulkan_histogram_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::histogram)
      .reserved_bytes;
}

std::size_t Program::vulkan_reduce_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::reduce)
      .reserved_bytes;
}

std::size_t Program::vulkan_check_count_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::check)
      .reserved_bytes;
}

std::size_t Program::vulkan_metric_reduce_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::metric)
      .reserved_bytes;
}

std::size_t Program::vulkan_sparse_algebra_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::sparse_algebra)
      .reserved_bytes;
}

std::uint64_t Program::vulkan_sparse_algebra_replay_generation() {
  TI_ERROR_IF(!vulkan_sparse_algebra_available(),
              "Vulkan sparse replay generation is only available on Vulkan.");
  Device *device = get_compute_device();
  TI_ERROR_IF(!device,
              "Vulkan sparse replay generation requires a compute device.");
  auto cache_lease = get_sparse_algebra_cache(this, device);
  return cache_lease->replay_generation;
}

std::size_t Program::vulkan_transform_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::transform)
      .reserved_bytes;
}

std::size_t Program::vulkan_add_merge_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::scatter_add)
      .reserved_bytes;
}

std::size_t Program::vulkan_indexed_copy_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::indexed)
      .reserved_bytes;
}

std::size_t Program::vulkan_scatter_add_workspace_bytes() const {
  return vulkan_indexed_copy_workspace_bytes();
}

std::size_t Program::vulkan_bucket_builder_workspace_bytes() const {
  return primitive_workspace_arena_
      .snapshot(PrimitiveWorkspaceBackend::vulkan,
                PrimitiveWorkspaceFamily::bucket)
      .reserved_bytes;
}

std::size_t Program::vulkan_grouped_reduce_workspace_bytes() const {
  return vulkan_bucket_builder_workspace_bytes();
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
  g_vulkan_sort_cpu_profile.clear();
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return g_vulkan_sort_cpu_profile.report_json();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_radix_sort_available() const {
  return false;
}

bool Program::vulkan_scan_available() const {
  return false;
}

bool Program::vulkan_scan_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_compact_available() const {
  return false;
}

bool Program::vulkan_histogram_available() const {
  return false;
}

bool Program::vulkan_histogram_value_type_available(int value_type,
                                                    int bin_type) const {
  return false;
}

bool Program::vulkan_reduce_available() const {
  return false;
}

bool Program::vulkan_reduce_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_check_count_available() const {
  return false;
}

bool Program::vulkan_check_count_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_metric_reduce_available() const {
  return false;
}

bool Program::vulkan_metric_reduce_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_sparse_algebra_available() const {
  return false;
}

bool Program::vulkan_sparse_assembly_available() const {
  return false;
}

VulkanSparseAssemblyDispatchInfo Program::vulkan_sparse_assemble_csr(
    Ndarray *packed_triplets,
    Ndarray *triplet_rows,
    Ndarray *triplet_columns,
    Ndarray *triplet_values,
    Ndarray *sorted_keys,
    Ndarray *sorted_values,
    Ndarray *segment_ids,
    Ndarray *unique_keys,
    Ndarray *segment_offsets,
    Ndarray *unique_values,
    Ndarray *row_offsets,
    Ndarray *column_indices,
    Ndarray *active_count,
    Ndarray *control,
    std::size_t capacity,
    std::size_t rows,
    std::size_t cols) {
  TI_ERROR("Vulkan sparse assembly requires TI_WITH_VULKAN=ON.");
  return {};
}

void Program::vulkan_copy_ndarray_prefix(Ndarray *dst,
                                         Ndarray *src,
                                         std::size_t bytes) {
  TI_ERROR("Vulkan ndarray prefix copy requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_csr_spmv(Ndarray *row_offsets,
                                     Ndarray *column_indices,
                                     Ndarray *values,
                                     Ndarray *x,
                                     Ndarray *y,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t nnz) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_bsr_spmv(Ndarray *row_offsets,
                                     Ndarray *column_indices,
                                     Ndarray *values,
                                     Ndarray *x,
                                     Ndarray *y,
                                     std::size_t block_rows,
                                     std::size_t block_cols,
                                     std::size_t block_nnz,
                                     std::size_t block_size) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_convergence(
    Ndarray *residual_squared,
    Ndarray *status,
    Ndarray *completed_iterations,
    Ndarray *rhs_squared,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t iteration) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_axpy(Ndarray *x,
                                        Ndarray *y,
                                        std::size_t n,
                                        float alpha) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_diagonal_refresh(
    Ndarray *values,
    Ndarray *diagonal_offsets,
    Ndarray *staging_inverse,
    Ndarray *status,
    std::size_t rows,
    std::size_t nnz) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_diagonal_apply(
    Ndarray *inverse_diagonal,
    Ndarray *input,
    Ndarray *output,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_block_cholesky_refresh(
    Ndarray *values,
    Ndarray *diagonal_block_offsets,
    Ndarray *staging_factors,
    Ndarray *status,
    std::size_t block_rows,
    std::size_t block_nnz,
    std::size_t block_size) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_block_diagonal_apply(
    Ndarray *factor_blocks,
    Ndarray *input,
    Ndarray *output,
    std::size_t block_rows,
    std::size_t block_size) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_dot(Ndarray *x,
                                       Ndarray *y,
                                       Ndarray *output,
                                       std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_norm(Ndarray *x,
                                        Ndarray *output,
                                        std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_scalar_divide(Ndarray *numerator,
                                                 Ndarray *denominator,
                                                 Ndarray *quotient,
                                                 Ndarray *status) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_cg_update(Ndarray *direction,
                                             Ndarray *applied_direction,
                                             Ndarray *alpha,
                                             Ndarray *solution,
                                             Ndarray *residual,
                                             std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_cg_direction(Ndarray *residual,
                                                Ndarray *beta,
                                                Ndarray *direction,
                                                std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_minres_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t stage,
    bool limit_reached,
    bool has_preconditioner,
    bool stop_on_estimate) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_minres_vector_state(
    Ndarray *source,
    Ndarray *destination,
    Ndarray *state,
    std::size_t n,
    std::uint32_t coefficient,
    bool add) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_minres_commit(
    Ndarray *v,
    Ndarray *r1,
    Ndarray *r2,
    Ndarray *lanczos_residual,
    Ndarray *w_older,
    Ndarray *w_old,
    Ndarray *w,
    Ndarray *solution,
    Ndarray *state,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_bicgstab_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot0,
    Ndarray *dot1,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::uint32_t stage,
    bool limit_reached) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_bicgstab_direction(
    Ndarray *residual,
    Ndarray *direction,
    Ndarray *operator_direction,
    Ndarray *state,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_bicgstab_intermediate(
    Ndarray *residual,
    Ndarray *operator_direction,
    Ndarray *intermediate,
    Ndarray *state,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_bicgstab_commit(
    Ndarray *solution_direction,
    Ndarray *solution_intermediate,
    Ndarray *intermediate,
    Ndarray *operator_intermediate,
    Ndarray *solution,
    Ndarray *residual,
    Ndarray *state,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_bicgstab_reconcile(
    Ndarray *true_residual,
    Ndarray *residual,
    Ndarray *shadow_residual,
    Ndarray *direction,
    Ndarray *operator_direction,
    Ndarray *solution,
    Ndarray *state,
    std::size_t n) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_gmres_multi_dot(
    Ndarray *basis,
    Ndarray *work,
    Ndarray *partials,
    Ndarray *projection,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t basis_count,
    std::size_t group_count) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_gmres_projection(
    Ndarray *basis,
    Ndarray *work,
    Ndarray *projection,
    Ndarray *hessenberg,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t restart,
    std::size_t step,
    std::size_t pass) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_gmres_basis(
    Ndarray *source,
    Ndarray *basis,
    Ndarray *current,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride,
    std::size_t row,
    std::size_t mode) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_gmres_combine(
    Ndarray *basis,
    Ndarray *coefficients,
    Ndarray *update,
    Ndarray *state,
    std::size_t n,
    std::size_t basis_stride) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

std::size_t Program::vulkan_sparse_gmres_scalar(
    Ndarray *initial_residual_squared,
    Ndarray *rhs_squared,
    Ndarray *dot0,
    Ndarray *dot1,
    Ndarray *hessenberg,
    Ndarray *cosines,
    Ndarray *sines,
    Ndarray *g,
    Ndarray *coefficients,
    Ndarray *state,
    float absolute_tolerance,
    float relative_tolerance,
    std::size_t restart,
    std::size_t max_iterations,
    std::size_t stage,
    std::size_t step,
    bool limit_reached) {
  TI_NOT_IMPLEMENTED;
  return 0;
}

bool Program::vulkan_transform_available() const {
  return false;
}

bool Program::vulkan_transform_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_add_merge_available() const {
  return false;
}

bool Program::vulkan_add_merge_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_indexed_copy_available() const {
  return false;
}

bool Program::vulkan_scatter_add_available() const {
  return false;
}

bool Program::vulkan_scatter_add_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_bucket_builder_available() const {
  return false;
}

bool Program::vulkan_bucket_builder_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_grouped_reduce_available() const {
  return false;
}

bool Program::vulkan_grouped_reduce_value_type_available(int value_type) const {
  return false;
}

bool Program::vulkan_grouped_reduce_atomic_value_type_available(
    int value_type) const {
  return false;
}

std::size_t Program::vulkan_radix_sort_u32_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   int key_type,
                                                   int value_type,
                                                   std::size_t key_offset,
                                                   std::size_t value_offset) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native radix sort requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_radix_sort_u32_dense_field(SNode *keys,
                                                       SNode *values,
                                                       int key_type,
                                                       int value_type,
                                                       std::size_t n) {
  TI_ERROR("Vulkan native dense field radix sort requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_ndarray(Ndarray *data,
                                                   int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_reverse_scan_ndarray(Ndarray *data,
                                                           int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native reverse scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_reverse_scan_member_ndarray(
    Ndarray *data,
    int value_type,
    std::size_t offset,
    std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided reverse scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_dense_field(SNode *data,
                                                       int value_type,
                                                       std::size_t n) {
  TI_ERROR("Vulkan native dense field scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_reverse_scan_dense_field(SNode *data,
                                                               int value_type,
                                                               std::size_t n) {
  TI_ERROR("Vulkan native dense field reverse scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_scan_dense_field_packed(SNode *data,
                                                              int value_type,
                                                              std::size_t n,
                                                              int lane_count) {
  TI_ERROR("Vulkan native packed dense field scan requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_inclusive_reverse_scan_dense_field_packed(
    SNode *data,
    int value_type,
    std::size_t n,
    int lane_count) {
  TI_ERROR(
      "Vulkan native packed dense field reverse scan requires "
      "TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_ndarray(Ndarray *values,
                                            Ndarray *flags,
                                            Ndarray *output,
                                            Ndarray *count,
                                            int value_type) {
  return vulkan_compact_ndarray_bounded(values, flags, output, count,
                                        value_type, nullptr, 0);
}

std::size_t Program::vulkan_compact_ndarray_bounded(
    Ndarray *values,
    Ndarray *flags,
    Ndarray *output,
    Ndarray *count,
    int value_type,
    Ndarray *dispatch_packet,
    int block_dim) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native compact requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_dense_field(SNode *values,
                                                SNode *flags,
                                                SNode *output,
                                                SNode *count,
                                                int value_type,
                                                std::size_t n) {
  TI_ERROR("Vulkan native dense field compact requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_ranges(DevicePtr values,
                                         DevicePtr flags,
                                         DevicePtr output,
                                         DevicePtr count,
                                         std::size_t item_bytes,
                                         std::size_t n,
                                         bool reuse_prefix) {
  TI_ERROR("Vulkan native compact requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_compact_i32_ndarray(Ndarray *values,
                                                Ndarray *flags,
                                                Ndarray *output,
                                                Ndarray *count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native compact requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_histogram_i32_ndarray(Ndarray *values,
                                                  Ndarray *bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native histogram requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_histogram_ndarray(Ndarray *values,
                                              Ndarray *bins,
                                              int value_type,
                                              int bin_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native histogram requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_histogram_dense_field(SNode *values,
                                                  SNode *bins,
                                                  int value_type,
                                                  int bin_type,
                                                  std::size_t n,
                                                  std::size_t num_bins) {
  TI_ERROR("Vulkan native dense field histogram requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_ndarray(Ndarray *values,
                                           Ndarray *output,
                                           int value_type,
                                           int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_member_ndarray(Ndarray *values,
                                                  Ndarray *output,
                                                  int value_type,
                                                  std::size_t offset,
                                                  std::size_t stride,
                                                  int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_i32_ndarray(Ndarray *values,
                                               Ndarray *output,
                                               int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_check_count_ndarray(Ndarray *values,
                                                Ndarray *output,
                                                int value_type,
                                                int check_op,
                                                int lower,
                                                int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native check_count requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_check_count_strided_ndarray(Ndarray *values,
                                                        Ndarray *output,
                                                        int value_type,
                                                        std::size_t offset,
                                                        std::size_t stride,
                                                        int check_op,
                                                        int lower,
                                                        int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided check_count requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_check_count_dense_field(SNode *values,
                                                    Ndarray *output,
                                                    int value_type,
                                                    std::size_t n,
                                                    int check_op,
                                                    int lower,
                                                    int upper) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field check_count requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_metric_reduce_ndarray(Ndarray *values,
                                                  Ndarray *other,
                                                  Ndarray *output,
                                                  int value_type,
                                                  int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native metric_reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_metric_reduce_strided_ndarray(
    Ndarray *values,
    Ndarray *other,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t other_offset,
    std::size_t other_stride,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided metric_reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_metric_reduce_dense_field(SNode *values,
                                                      SNode *other,
                                                      Ndarray *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field metric_reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_metric_reduce_dense_field_strided_ndarray(
    SNode *field,
    Ndarray *array,
    Ndarray *output,
    int value_type,
    std::size_t n,
    std::size_t array_offset,
    std::size_t array_stride,
    bool field_is_values,
    int metric_op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native mixed metric_reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_dense_field(SNode *values,
                                               SNode *output,
                                               int value_type,
                                               std::size_t n,
                                               int op) {
  TI_ERROR("Vulkan native dense field reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_reduce_dense_field_packed(SNode *values,
                                                      SNode *output,
                                                      int value_type,
                                                      std::size_t n,
                                                      int lane_count,
                                                      int op) {
  TI_ERROR(
      "Vulkan native packed dense field reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_ndarray(Ndarray *src,
                                                     Ndarray *dst,
                                                     int value_type,
                                                     double scale,
                                                     double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_indexed_affine_ndarray(Ndarray *src,
                                                              Ndarray *indices,
                                                              Ndarray *dst,
                                                              int value_type,
                                                              double scale,
                                                              double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native indexed transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_member_ndarray(Ndarray *src,
                                                            Ndarray *dst,
                                                            int value_type,
                                                            std::size_t offset,
                                                            std::size_t stride,
                                                            double scale,
                                                            double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_packed_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    int lane_count,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride,
    double scale,
    double bias) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native packed strided transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_dense_field(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         double scale,
                                                         double bias) {
  TI_ERROR("Vulkan native dense field transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_transform_affine_dense_field_packed(
    SNode *src,
    SNode *dst,
    int value_type,
    std::size_t n,
    int lane_count,
    double scale,
    double bias) {
  TI_ERROR(
      "Vulkan native packed dense field transform requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_zero_dense_field(SNode *dst,
                                             int value_type,
                                             std::size_t n) {
  TI_ERROR("Vulkan native dense field zero-fill requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_zero_dense_fields(
    const std::vector<SNode *> &dsts,
    const std::vector<int> &value_types,
    const std::vector<std::size_t> &ns) {
  TI_ERROR("Vulkan native dense field zero-fill requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_add_merge_ndarray(Ndarray *src,
                                              Ndarray *dst,
                                              int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native add-merge requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_add_merge_strided_ndarray(
    Ndarray *src,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided add-merge requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_add_merge_dense_field(Ndarray *src,
                                                  SNode *dst,
                                                  int value_type,
                                                  std::size_t n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field add-merge requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_add_merge_dense_field_packed(SNode *src,
                                                         SNode *dst,
                                                         int value_type,
                                                         std::size_t n,
                                                         int lane_count) {
  TI_ERROR(
      "Vulkan native packed dense field add-merge requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_add_scalar_field_to_dense_field(SNode *src,
                                                            SNode *dst,
                                                            int value_type,
                                                            std::size_t n) {
  TI_ERROR("Vulkan native scalar-to-dense add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_ndarray(Ndarray *src,
                                           Ndarray *indices,
                                           Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_dense_field(SNode *src,
                                               Ndarray *indices,
                                               SNode *dst,
                                               int value_type,
                                               std::size_t src_n,
                                               std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_dense_field_packed(SNode *src,
                                                      Ndarray *indices,
                                                      SNode *dst,
                                                      int value_type,
                                                      std::size_t src_n,
                                                      std::size_t dst_n,
                                                      int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR(
      "Vulkan native packed dense field gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR(
      "Vulkan native packed dense field gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_gather_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR("Vulkan native dense field gather requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_ndarray(Ndarray *src,
                                            Ndarray *indices,
                                            Ndarray *dst) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    std::size_t item_bytes,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field(SNode *src,
                                                Ndarray *indices,
                                                SNode *dst,
                                                int value_type,
                                                std::size_t src_n,
                                                std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field_packed(SNode *src,
                                                       Ndarray *indices,
                                                       SNode *dst,
                                                       int value_type,
                                                       std::size_t src_n,
                                                       std::size_t dst_n,
                                                       int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR(
      "Vulkan native packed dense field scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR(
      "Vulkan native packed dense field scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR("Vulkan native dense field scatter requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_ndarray(Ndarray *src,
                                                Ndarray *indices,
                                                Ndarray *dst,
                                                int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_member_ndarray(Ndarray *src,
                                                       Ndarray *indices,
                                                       Ndarray *dst,
                                                       int value_type,
                                                       std::size_t offset,
                                                       std::size_t stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_strided_ndarray(
    Ndarray *src,
    Ndarray *indices,
    Ndarray *dst,
    int value_type,
    std::size_t src_offset,
    std::size_t src_stride,
    std::size_t dst_offset,
    std::size_t dst_stride) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_dense_field(SNode *src,
                                                    Ndarray *indices,
                                                    SNode *dst,
                                                    int value_type,
                                                    std::size_t src_n,
                                                    std::size_t dst_n) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_dense_field_packed(SNode *src,
                                                           Ndarray *indices,
                                                           SNode *dst,
                                                           int value_type,
                                                           std::size_t src_n,
                                                           std::size_t dst_n,
                                                           int lane_count) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR(
      "Vulkan native packed dense field scatter-add requires "
      "TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_dense_field_packed_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n,
    int lane_count) {
  TI_ERROR(
      "Vulkan native packed dense field scatter-add requires "
      "TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_scatter_add_dense_field_indices_field(
    SNode *src,
    SNode *indices,
    SNode *dst,
    int value_type,
    std::size_t src_n,
    std::size_t indices_n,
    std::size_t dst_n) {
  TI_ERROR("Vulkan native dense field scatter-add requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_bucket_builder_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *offsets,
                                                       Ndarray *output,
                                                       Ndarray *cursor) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native bucket builder requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_bucket_builder_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *offsets,
                                                   Ndarray *output,
                                                   Ndarray *cursor,
                                                   int value_type) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native bucket builder requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_bucket_builder_dense_field(SNode *keys,
                                                       SNode *values,
                                                       SNode *offsets,
                                                       SNode *output,
                                                       Ndarray *cursor,
                                                       int value_type,
                                                       std::size_t n,
                                                       std::size_t num_bins) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native dense field bucket builder requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_atomic_ndarray(Ndarray *keys,
                                                              Ndarray *values,
                                                              Ndarray *output,
                                                              int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_ndarray(Ndarray *keys,
                                                          Ndarray *values,
                                                          Ndarray *output,
                                                          int value_type,
                                                          int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_dense_field(
    SNode *keys,
    SNode *values,
    SNode *output,
    int value_type,
    std::size_t n,
    std::size_t num_groups,
    int op) {
  TI_ERROR("Vulkan native dense field grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_member_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t offset,
    std::size_t stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_atomic_strided_keys_ndarray(
    Ndarray *keys,
    Ndarray *values,
    Ndarray *output,
    int value_type,
    std::size_t keys_offset,
    std::size_t keys_stride,
    std::size_t values_offset,
    std::size_t values_stride,
    std::size_t output_offset,
    std::size_t output_stride,
    int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native strided grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_i32_ndarray(Ndarray *keys,
                                                       Ndarray *values,
                                                       Ndarray *output,
                                                       Ndarray *offsets,
                                                       Ndarray *scratch,
                                                       Ndarray *cursor,
                                                       int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_ndarray(Ndarray *keys,
                                                   Ndarray *values,
                                                   Ndarray *output,
                                                   Ndarray *offsets,
                                                   Ndarray *scratch,
                                                   Ndarray *cursor,
                                                   int value_type,
                                                   int op) {
  auto native_ndarray_submission_guard =
      acquire_runtime_resource_submission_guard();
  TI_ERROR("Vulkan native grouped reduce requires TI_WITH_VULKAN=ON.");
  return 0;
}

void Program::vulkan_radix_sort_clear_workspace() {
}

void Program::vulkan_scan_clear_workspace() {
}

void Program::vulkan_compact_clear_workspace() {
}

void Program::vulkan_histogram_clear_workspace() {
}

void Program::vulkan_reduce_clear_workspace() {
}

void Program::vulkan_check_count_clear_workspace() {
}

void Program::vulkan_metric_reduce_clear_workspace() {
}

void Program::vulkan_sparse_algebra_clear_workspace() {
}

void Program::vulkan_transform_clear_workspace() {
}

void Program::vulkan_add_merge_clear_workspace() {
}

void Program::vulkan_indexed_copy_clear_workspace() {
}

void Program::vulkan_scatter_add_clear_workspace() {
}

void Program::vulkan_bucket_builder_clear_workspace() {
}

void Program::vulkan_grouped_reduce_clear_workspace() {
}

void Program::vulkan_clear_primitive_caches() {
}

std::size_t Program::vulkan_radix_sort_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_scan_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_compact_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_histogram_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_reduce_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_check_count_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_metric_reduce_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_sparse_algebra_workspace_bytes() const {
  return 0;
}

std::uint64_t Program::vulkan_sparse_algebra_replay_generation() {
  return 0;
}

std::size_t Program::vulkan_transform_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_add_merge_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_indexed_copy_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_scatter_add_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_bucket_builder_workspace_bytes() const {
  return 0;
}

std::size_t Program::vulkan_grouped_reduce_workspace_bytes() const {
  return 0;
}

void Program::vulkan_radix_sort_cpu_profile_clear() {
}

std::string Program::vulkan_radix_sort_cpu_profile_report() const {
  return "{}";
}

}  // namespace taichi::lang

#endif
