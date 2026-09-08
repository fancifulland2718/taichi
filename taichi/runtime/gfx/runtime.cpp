#include "taichi/runtime/gfx/runtime.h"
#include "taichi/program/program.h"
#include "taichi/common/filesystem.hpp"
#include "taichi/util/environ_config.h"
#include "taichi/system/profiler_annotation.h"

// FIXME: (penguinliong) Special offer for `run_codegen`. Find a new home for it
// in the future.
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/codegen/spirv/spirv_ir_builder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fp16.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace gfx {

namespace {

std::atomic<uint64_t> graph_replay_slot_saturation_fallbacks{0};

constexpr std::size_t kStructuredPacketWords = 3;
constexpr std::size_t kStructuredObservationWords = 5;
constexpr std::size_t kStructuredChainedSinkWords = 3;
constexpr std::size_t kStructuredMaximumPackets = 64;
constexpr std::size_t kStructuredMaximumEncodedActions = 4096;
constexpr std::size_t kStructuredChainedMaximumDispatches = 256;
constexpr std::size_t kStructuredChainedMaximumControlBytes = 64 * 1024;
constexpr std::size_t kNestedStructuredHeaderWords = 8;
constexpr std::size_t kNestedStructuredTraceWords = 6;
constexpr std::size_t kNestedStructuredMaximumOuterIterations = 64;
constexpr std::size_t kGraphReplayRetirementPollBudget = 8;

std::vector<uint32_t> make_compact_controller_spirv(
    const DeviceCapabilityConfig &caps,
    const std::vector<uint32_t> &group_counts,
    bool has_status,
    std::size_t control_word_offset = 0) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto control = builder.buffer_argument(u32, 0, 1, "control");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto one = builder.uint_immediate_number(u32, 1);
  const auto predicate_value = builder.load_variable(
      builder.struct_array_access(u32, predicate, zero), u32);
  const auto active = builder.ne(predicate_value, zero);
  const auto active_increment = builder.select(active, one, zero);
  const auto observation_base = builder.uint_immediate_number(
      u32, control_word_offset +
               group_counts.size() * kStructuredPacketWords);
  const auto logical_count_ptr =
      builder.struct_array_access(u32, control, observation_base);
  const auto logical_count = builder.load_variable(logical_count_ptr, u32);
  if (has_status) {
    const auto initial_status_ptr = builder.struct_array_access(
        u32, control,
        builder.uint_immediate_number(
            u32, control_word_offset +
                     group_counts.size() * kStructuredPacketWords + 4));
    const auto initial_status =
        builder.load_variable(initial_status_ptr, u32);
    const auto status_value = builder.load_variable(
        builder.struct_array_access(u32, status, zero), u32);
    builder.store_variable(
        initial_status_ptr,
        builder.select(builder.eq(logical_count, zero), status_value,
                       initial_status));
  }
  builder.store_variable(logical_count_ptr,
                         builder.add(logical_count, active_increment));

  for (std::size_t i = 0; i < group_counts.size(); ++i) {
    const auto packet_base = builder.uint_immediate_number(
        u32, control_word_offset + i * kStructuredPacketWords);
    const auto group_count =
        builder.uint_immediate_number(u32, group_counts[i]);
    builder.store_variable(
        builder.struct_array_access(u32, control, packet_base),
        builder.select(active, group_count, zero));
    builder.store_variable(
        builder.struct_array_access(
            u32, control,
            builder.uint_immediate_number(
                u32, control_word_offset + i * kStructuredPacketWords + 1)),
        one);
    builder.store_variable(
        builder.struct_array_access(
            u32, control,
            builder.uint_immediate_number(
                u32, control_word_offset + i * kStructuredPacketWords + 2)),
        one);
  }

  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, control, status};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_structured_terminal_spirv(
    const DeviceCapabilityConfig &caps,
    std::size_t observation_word_offset,
    bool has_status) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto counter = builder.buffer_argument(u32, 0, 1, "counter");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto control = builder.buffer_argument(u32, 0, 3, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto observation_base = observation_word_offset;
  const auto load_scalar = [&](spirv::Value buffer) {
    return builder.load_variable(
        builder.struct_array_access(u32, buffer, zero), u32);
  };
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, observation_base + 1)),
      load_scalar(predicate));
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, observation_base + 2)),
      load_scalar(counter));
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, observation_base + 3)),
      has_status ? load_scalar(status) : zero);
  const auto logical_count = builder.load_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, observation_base)),
      u32);
  const auto initial_status_ptr = builder.struct_array_access(
      u32, control,
      builder.uint_immediate_number(u32, observation_base + 4));
  const auto initial_status =
      builder.load_variable(initial_status_ptr, u32);
  const auto final_status = has_status ? load_scalar(status) : zero;
  builder.store_variable(
      initial_status_ptr,
      builder.select(builder.eq(logical_count, zero), final_status,
                     initial_status));
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, counter, status, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_nested_outer_snapshot_spirv(
    const DeviceCapabilityConfig &caps,
    std::size_t header_word_offset,
    bool has_status,
    bool terminal) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto counter = builder.buffer_argument(u32, 0, 1, "counter");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto control = builder.buffer_argument(u32, 0, 3, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto base = header_word_offset + (terminal ? 5 : 2);
  const auto load_scalar = [&](spirv::Value buffer) {
    return builder.load_variable(
        builder.struct_array_access(u32, buffer, zero), u32);
  };
  builder.store_variable(
      builder.struct_array_access(
          u32, control, builder.uint_immediate_number(u32, base)),
      load_scalar(predicate));
  builder.store_variable(
      builder.struct_array_access(
          u32, control, builder.uint_immediate_number(u32, base + 1)),
      load_scalar(counter));
  builder.store_variable(
      builder.struct_array_access(
          u32, control, builder.uint_immediate_number(u32, base + 2)),
      has_status ? load_scalar(status) : zero);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, counter, status, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_nested_inner_initial_spirv(
    const DeviceCapabilityConfig &caps,
    std::size_t header_word_offset,
    std::size_t trace_word_offset,
    std::size_t trace_row_words,
    bool has_status,
    bool advance_outer) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto counter = builder.buffer_argument(u32, 0, 0, "counter");
  const auto status = builder.buffer_argument(u32, 0, 1, "status");
  const auto control = builder.buffer_argument(u32, 0, 2, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto one = builder.uint_immediate_number(u32, 1);
  const auto logical_ptr = builder.struct_array_access(
      u32, control,
      builder.uint_immediate_number(u32, header_word_offset));
  const auto logical = builder.load_variable(logical_ptr, u32);
  if (advance_outer) {
    builder.store_variable(
        builder.struct_array_access(
            u32, control,
            builder.uint_immediate_number(u32, header_word_offset + 1)),
        logical);
    builder.store_variable(logical_ptr, builder.add(logical, one));
  }
  const auto row_index =
      advance_outer
          ? logical
          : builder.load_variable(
                builder.struct_array_access(
                    u32, control,
                    builder.uint_immediate_number(
                        u32, header_word_offset + 1)),
                u32);
  const auto row = builder.add(
      builder.uint_immediate_number(u32, trace_word_offset),
      builder.mul(row_index,
                  builder.uint_immediate_number(u32, trace_row_words)));
  const auto counter_value = builder.load_variable(
      builder.struct_array_access(u32, counter, zero), u32);
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(row, builder.uint_immediate_number(u32, 1))),
      counter_value);
  const auto status_value =
      has_status
          ? builder.load_variable(
                builder.struct_array_access(u32, status, zero), u32)
          : zero;
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(row, builder.uint_immediate_number(u32, 4))),
      status_value);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {counter, status, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_nested_inner_terminal_spirv(
    const DeviceCapabilityConfig &caps,
    std::size_t inner_observation_word_offset,
    std::size_t header_word_offset,
    std::size_t trace_word_offset,
    std::size_t trace_row_words,
    bool has_status) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto counter = builder.buffer_argument(u32, 0, 1, "counter");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto control = builder.buffer_argument(u32, 0, 3, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto row_index = builder.load_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, header_word_offset + 1)),
      u32);
  const auto row = builder.add(
      builder.uint_immediate_number(u32, trace_word_offset),
      builder.mul(row_index,
                  builder.uint_immediate_number(u32, trace_row_words)));
  const auto inner_logical = builder.load_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(
              u32, inner_observation_word_offset)),
      u32);
  builder.store_variable(builder.struct_array_access(u32, control, row),
                         inner_logical);
  const auto load_scalar = [&](spirv::Value buffer) {
    return builder.load_variable(
        builder.struct_array_access(u32, buffer, zero), u32);
  };
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(row, builder.uint_immediate_number(u32, 2))),
      load_scalar(counter));
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(row, builder.uint_immediate_number(u32, 3))),
      load_scalar(predicate));
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(row, builder.uint_immediate_number(u32, 5))),
      has_status ? load_scalar(status) : zero);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, counter, status, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_chained_controller_spirv(
    const DeviceCapabilityConfig &caps,
    const std::vector<uint32_t> &group_counts,
    std::uint32_t max_iterations,
    bool has_status) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto control = builder.buffer_argument(u32, 0, 1, "control");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto one = builder.uint_immediate_number(u32, 1);
  const auto segment_words = builder.uint_immediate_number(
      u32, (group_counts.size() + 1) * kStructuredPacketWords);
  const auto observation_base = builder.uint_immediate_number(
      u32, max_iterations * (group_counts.size() + 1) *
               kStructuredPacketWords);
  const auto logical_count_ptr =
      builder.struct_array_access(u32, control, observation_base);
  const auto logical_count = builder.load_variable(logical_count_ptr, u32);
  const auto predicate_value = builder.load_variable(
      builder.struct_array_access(u32, predicate, zero), u32);
  const auto active = builder.ne(predicate_value, zero);
  const auto active_increment = builder.select(active, one, zero);

  if (has_status) {
    const auto initial_status_ptr = builder.struct_array_access(
        u32, control,
        builder.add(observation_base,
                    builder.uint_immediate_number(u32, 4)));
    const auto initial_status =
        builder.load_variable(initial_status_ptr, u32);
    const auto status_value = builder.load_variable(
        builder.struct_array_access(u32, status, zero), u32);
    builder.store_variable(
        initial_status_ptr,
        builder.select(builder.eq(logical_count, zero), status_value,
                       initial_status));
  }
  builder.store_variable(logical_count_ptr,
                         builder.add(logical_count, active_increment));

  const auto segment_base = builder.mul(logical_count, segment_words);
  for (std::size_t i = 0; i < group_counts.size(); ++i) {
    const auto packet_base = builder.add(
        segment_base,
        builder.uint_immediate_number(
            u32, (i + 1) * kStructuredPacketWords));
    const auto group_count =
        builder.uint_immediate_number(u32, group_counts[i]);
    builder.store_variable(
        builder.struct_array_access(u32, control, packet_base),
        builder.select(active, group_count, zero));
    builder.store_variable(
        builder.struct_array_access(
            u32, control,
            builder.add(packet_base, builder.uint_immediate_number(u32, 1))),
        one);
    builder.store_variable(
        builder.struct_array_access(
            u32, control,
            builder.add(packet_base, builder.uint_immediate_number(u32, 2))),
        one);
  }

  const auto next_iteration = builder.add(logical_count, one);
  const auto next_segment = builder.mul(next_iteration, segment_words);
  const auto has_next = builder.lt(
      next_iteration, builder.uint_immediate_number(u32, max_iterations));
  const auto next_enabled =
      builder.select(active, builder.select(has_next, one, zero), zero);
  const auto sink_packet = builder.add(
      observation_base,
      builder.uint_immediate_number(u32, kStructuredObservationWords));
  const auto next_packet =
      builder.select(has_next, next_segment, sink_packet);
  builder.store_variable(
      builder.struct_array_access(u32, control, next_packet), next_enabled);
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(next_packet, builder.uint_immediate_number(u32, 1))),
      one);
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.add(next_packet, builder.uint_immediate_number(u32, 2))),
      one);

  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, control, status};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_conditional_controller_spirv(
    const DeviceCapabilityConfig &caps,
    std::uint32_t max_iterations,
    bool has_status) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto control = builder.buffer_argument(u32, 0, 1, "control");
  const auto status = builder.buffer_argument(u32, 0, 2, "status");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto one = builder.uint_immediate_number(u32, 1);
  const auto observation_base =
      builder.uint_immediate_number(u32, max_iterations);
  const auto logical_count_ptr =
      builder.struct_array_access(u32, control, observation_base);
  const auto logical_count = builder.load_variable(logical_count_ptr, u32);
  const auto predicate_value = builder.load_variable(
      builder.struct_array_access(u32, predicate, zero), u32);
  const auto active = builder.ne(predicate_value, zero);
  const auto sink_index = builder.uint_immediate_number(
      u32, max_iterations + kStructuredObservationWords);
  const auto stable_index =
      builder.select(active, logical_count, sink_index);
  builder.store_variable(
      builder.struct_array_access(u32, control, stable_index),
      predicate_value);

  if (has_status) {
    const auto initial_status_ptr = builder.struct_array_access(
        u32, control,
        builder.add(observation_base,
                    builder.uint_immediate_number(u32, 4)));
    const auto initial_status =
        builder.load_variable(initial_status_ptr, u32);
    const auto status_value = builder.load_variable(
        builder.struct_array_access(u32, status, zero), u32);
    builder.store_variable(
        initial_status_ptr,
        builder.select(builder.eq(logical_count, zero), status_value,
                       initial_status));
  }
  builder.store_variable(
      logical_count_ptr,
      builder.add(logical_count, builder.select(active, one, zero)));

  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);
  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, control, status};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::vector<uint32_t> make_coarse_conditional_gate_spirv(
    const DeviceCapabilityConfig &caps,
    std::uint32_t stable_predicate_index) {
  spirv::IRBuilder builder(Arch::vulkan, &caps);
  builder.init_header();
  const auto u32 = builder.u32_type();
  const auto predicate = builder.buffer_argument(u32, 0, 0, "predicate");
  const auto control = builder.buffer_argument(u32, 0, 1, "control");
  const auto function = builder.new_function();
  builder.start_function(function);

  const auto zero = builder.uint_immediate_number(u32, 0);
  const auto predicate_value = builder.load_variable(
      builder.struct_array_access(u32, predicate, zero), u32);
  builder.store_variable(
      builder.struct_array_access(
          u32, control,
          builder.uint_immediate_number(u32, stable_predicate_index)),
      predicate_value);
  builder.make_inst(spv::OpReturn);
  builder.make_inst(spv::OpFunctionEnd);

  std::vector<spirv::Value> interfaces;
  if (caps.get(DeviceCapability::spirv_version) > 0x10300) {
    interfaces = {predicate, control};
  }
  builder.commit_kernel_function(function, "main", std::move(interfaces),
                                 {1, 1, 1});
  return builder.finalize();
}

std::unique_ptr<Pipeline> create_structured_pipeline(
    Device *device,
    const std::vector<uint32_t> &spirv,
    const char *name) {
  PipelineSourceDesc source{PipelineSourceType::spirv_binary, spirv.data(),
                            spirv.size() * sizeof(uint32_t),
                            PipelineStageType::compute};
  auto [pipeline, result] =
      device->create_pipeline_unique(source, name);
  TI_ERROR_IF(result != RhiResult::success || !pipeline,
              "Failed to create Vulkan structured control pipeline '{}': "
              "RhiResult({})",
              name, result);
  return std::move(pipeline);
}

class HostDeviceContextBlitter {
 public:
  HostDeviceContextBlitter(const KernelContextAttributes *ctx_attribs,
                           const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
                               *runtime_array_args,
                           LaunchContextBuilder &host_ctx,
                           Device *device,
                           DeviceAllocation *device_args_buffer,
                           DeviceAllocation *device_ret_buffer)
      : ctx_attribs_(ctx_attribs),
        runtime_array_args_(runtime_array_args),
        host_ctx_(host_ctx),
        device_args_buffer_(device_args_buffer),
        device_ret_buffer_(device_ret_buffer),
        device_(device) {
  }

  void host_to_device(
      const std::unordered_map<std::vector<int>,
                               DeviceAllocation,
                               hashing::Hasher<std::vector<int>>> &ext_arrays,
      const std::unordered_map<std::vector<int>,
                               size_t,
                               hashing::Hasher<std::vector<int>>> &ext_arr_size,
      const std::unordered_map<std::vector<int>,
                               const ArgPack *,
                               hashing::Hasher<std::vector<int>>> &argpacks) {
    if (!ctx_attribs_->has_args()) {
      return;
    }

    void *device_base{nullptr};
    TI_ASSERT(device_->map(*device_args_buffer_, &device_base) ==
              RhiResult::success);

    for (const auto &array_arg : *runtime_array_args_) {
      const auto &indices = array_arg.indices;
      const auto alloc_type = host_ctx_.device_allocation_type[indices];
      if (alloc_type == LaunchContextBuilder::DevAllocType::kNone &&
          ext_arr_size.at(indices)) {
          // Only need to blit ext arrs (host array)
          // Bug B/C fix (forge 2026-05): always blit host→device for ext
          // arrs regardless of ExternalPtrAccess flag. The previous READ-only
          // optimization left WRITE-only device buffers uninitialized with
          // recycled GPU memory; for kernels whose struct-for over a sparse
          // SNode only writes a subset of cells (e.g. tensor_to_ext_arr from
          // to_numpy on bitmasked/pointer fields after deactivate), the
          // unwritten cells leaked stale data back to the user. The host
          // ndarray (e.g. np.zeros from to_numpy) is the user's contract for
          // the initial state of the buffer; device buffer must match it.
          (void)array_arg.access;
          {
            DeviceAllocation buffer = ext_arrays.at(indices);
            void *device_arr_ptr{nullptr};
            TI_ASSERT(device_->map(buffer, &device_arr_ptr) ==
                      RhiResult::success);
            const void *host_ptr =
                host_ctx_.array_ptrs[array_arg.data_ptr_indices];
            std::memcpy(device_arr_ptr, host_ptr, ext_arr_size.at(indices));
            device_->unmap(buffer);
          }
          if (array_arg.grad_access != 0) {
            DeviceAllocation grad_buffer =
                ext_arrays.at(array_arg.grad_ptr_indices);
            void *device_grad_ptr{nullptr};
            TI_ASSERT(device_->map(grad_buffer, &device_grad_ptr) ==
                      RhiResult::success);
            const void *host_grad_ptr =
                host_ctx_.array_ptrs[array_arg.grad_ptr_indices];
            TI_ASSERT(host_grad_ptr != nullptr);
            std::memcpy(device_grad_ptr, host_grad_ptr,
                        ext_arr_size.at(indices));
            device_->unmap(grad_buffer);
          }
        }
        // Substitute in the device address.

        if ((alloc_type == LaunchContextBuilder::DevAllocType::kNone ||
             alloc_type == LaunchContextBuilder::DevAllocType::kNdarray ||
             alloc_type ==
                 LaunchContextBuilder::DevAllocType::kDenseStorage) &&
            device_->get_caps().get(
                DeviceCapability::spirv_has_physical_storage_buffer)) {
          uint64_t addr =
              device_->get_memory_physical_pointer(ext_arrays.at(indices));
          if (alloc_type ==
              LaunchContextBuilder::DevAllocType::kDenseStorage) {
            addr += host_ctx_.get_resolved_dense_storage(indices).byte_offset;
          }
          uint64_t grad_addr = 0;
          if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray &&
              host_ctx_.array_ptrs[array_arg.grad_ptr_indices] != nullptr) {
            auto grad_alloc =
                *(DeviceAllocation *)(
                    host_ctx_.array_ptrs[array_arg.grad_ptr_indices]);
            grad_addr = device_->get_memory_physical_pointer(grad_alloc);
          } else if (alloc_type == LaunchContextBuilder::DevAllocType::kNone &&
                     array_arg.grad_access != 0) {
            grad_addr = device_->get_memory_physical_pointer(
                ext_arrays.at(array_arg.grad_ptr_indices));
          }
          host_ctx_.set_ndarray_ptrs(indices, addr, grad_addr);
        }
    }

    std::memcpy(device_base, host_ctx_.get_context().arg_buffer,
                ctx_attribs_->args_bytes());

    device_->unmap(*device_args_buffer_);
  }

  bool device_to_host(
      CommandList *cmdlist,
      const std::unordered_map<std::vector<int>,
                               DeviceAllocation,
                               hashing::Hasher<std::vector<int>>> &ext_arrays,
      const std::unordered_map<std::vector<int>,
                               size_t,
                               hashing::Hasher<std::vector<int>>>
          &ext_arr_size) {
    if (ctx_attribs_->empty()) {
      return false;
    }

    bool require_sync = ctx_attribs_->rets().size() > 0;
    std::vector<DevicePtr> readback_dev_ptrs;
    std::vector<void *> readback_host_ptrs;
    std::vector<size_t> readback_sizes;

    for (const auto &array_arg : *runtime_array_args_) {
      const auto &indices = array_arg.indices;
      if (host_ctx_.device_allocation_type[indices] ==
              LaunchContextBuilder::DevAllocType::kNone &&
          ext_arr_size.at(indices)) {
        if (array_arg.access & uint32_t(irpass::ExternalPtrAccess::WRITE)) {
          // Only need to blit ext arrs (host array)
          readback_dev_ptrs.push_back(ext_arrays.at(indices).get_ptr(0));
          readback_host_ptrs.push_back(
              host_ctx_.array_ptrs[array_arg.data_ptr_indices]);
          // TODO: readback grad_ptrs as well once ndarray ad is supported
          readback_sizes.push_back(ext_arr_size.at(indices));
          require_sync = true;
        }
        if (array_arg.grad_access &
            uint32_t(irpass::ExternalPtrAccess::WRITE)) {
          readback_dev_ptrs.push_back(
              ext_arrays.at(array_arg.grad_ptr_indices).get_ptr(0));
          readback_host_ptrs.push_back(
              host_ctx_.array_ptrs[array_arg.grad_ptr_indices]);
          readback_sizes.push_back(ext_arr_size.at(indices));
          require_sync = true;
        }
      }
    }

    if (require_sync) {
      if (readback_sizes.size()) {
        StreamSemaphore command_complete_sema =
            device_->get_compute_stream()->submit(cmdlist);

        device_->wait_idle();

        // In this case `readback_data` syncs
        TI_ASSERT(device_->readback_data(
                      readback_dev_ptrs.data(), readback_host_ptrs.data(),
                      readback_sizes.data(), int(readback_sizes.size()),
                      {command_complete_sema}) == RhiResult::success);
      } else {
        device_->get_compute_stream()->submit_synced(cmdlist);
      }

      if (!ctx_attribs_->has_rets()) {
        return true;
      }
    } else {
      return false;
    }

    void *device_base{nullptr};
    TI_ASSERT(device_->map(*device_ret_buffer_, &device_base) ==
              RhiResult::success);

    void *ctx_result_buffer = host_ctx_.get_context().result_buffer;
    std::memcpy(ctx_result_buffer, device_base, ctx_attribs_->rets_bytes());

    device_->unmap(*device_ret_buffer_);

    return true;
  }

  static std::unique_ptr<HostDeviceContextBlitter> maybe_make(
      const KernelContextAttributes *ctx_attribs,
      const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
          *runtime_array_args,
      LaunchContextBuilder &host_ctx,
      Device *device,
      DeviceAllocation *device_args_buffer,
      DeviceAllocation *device_ret_buffer) {
    if (ctx_attribs->empty()) {
      return nullptr;
    }
    return std::make_unique<HostDeviceContextBlitter>(
        ctx_attribs, runtime_array_args, host_ctx, device, device_args_buffer,
        device_ret_buffer);
  }

 private:
  const KernelContextAttributes *const ctx_attribs_;
  const std::vector<CompiledTaichiKernel::RuntimeArrayArg>
      *const runtime_array_args_;
  LaunchContextBuilder &host_ctx_;
  DeviceAllocation *const device_args_buffer_;
  DeviceAllocation *const device_ret_buffer_;
  Device *const device_;
};

uint64_t graph_allocation_generation(DeviceAllocation alloc) {
  return 0;
}

void push_graph_allocation_key(std::vector<uint64_t> &key,
                               DeviceAllocation alloc,
                               uint64_t offset = 0,
                               uint64_t bytes = 0) {
  key.push_back(reinterpret_cast<uint64_t>(alloc.device));
  key.push_back(alloc.alloc_id);
  key.push_back(graph_allocation_generation(alloc));
  key.push_back(offset);
  key.push_back(bytes);
}

void push_graph_allocation_structure_key(std::vector<uint64_t> &key,
                                         DeviceAllocation alloc,
                                         uint64_t ordinal,
                                         uint64_t offset,
                                         uint64_t bytes) {
  key.push_back(reinterpret_cast<uint64_t>(alloc.device));
  key.push_back(ordinal);
  key.push_back(offset);
  key.push_back(bytes);
}

}  // namespace

uint64_t get_graph_replay_slot_saturation_fallbacks() {
  return graph_replay_slot_saturation_fallbacks.load(
      std::memory_order_relaxed);
}

class GraphReplayRegistry {
 public:
  explicit GraphReplayRegistry(GfxRuntime *runtime) : runtime_(runtime) {
  }

  void retire(uint64_t token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (runtime_ != nullptr) {
      runtime_->retire_graph_replay(token);
    }
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    runtime_ = nullptr;
  }

  GraphReplayStats debug_stats(uint64_t replay_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (runtime_ == nullptr) {
      return {};
    }
    return runtime_->debug_graph_replay_stats(replay_key);
  }

  GraphReplayStats snapshot_stats(uint64_t replay_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (runtime_ == nullptr) {
      return {};
    }
    return runtime_->snapshot_graph_replay_stats(replay_key);
  }

  void launch_prepared(uint64_t replay_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(runtime_ == nullptr, "Prepared Vulkan Graph belongs to a closed runtime");
    runtime_->launch_prepared_graph(replay_key);
  }

 private:
  std::mutex mutex_;
  GfxRuntime *runtime_{nullptr};
};

GraphReplayRegistration::GraphReplayRegistration(
    std::shared_ptr<GraphReplayRegistry> registry,
    uint64_t replay_key)
    : registry_(std::move(registry)), replay_key_(replay_key) {
  TI_ASSERT(registry_ != nullptr);
  TI_ASSERT(replay_key_ != 0);
}

GraphReplayRegistration::~GraphReplayRegistration() {
  if (registry_ != nullptr) {
    registry_->retire(replay_key_);
  }
}

GraphReplayStats GraphReplayRegistration::debug_stats() const {
  if (registry_ == nullptr) {
    return {};
  }
  return registry_->debug_stats(replay_key_);
}

GraphReplayStats GraphReplayRegistration::snapshot_stats() const {
  if (registry_ == nullptr) {
    return {};
  }
  return registry_->snapshot_stats(replay_key_);
}

void GraphReplayRegistration::launch_prepared() const {
  registry_->launch_prepared(replay_key_);
}

constexpr size_t kGtmpBufferSize = 1024 * 1024;
constexpr size_t kHashOverflowBufferSize = 6 * sizeof(uint32_t);
constexpr size_t kListGenBufferSize = 32 << 20;
constexpr size_t kListGenMinBufferSize = sizeof(uint32_t);
constexpr size_t kListGenAutoSlackEntries = 1024;
constexpr size_t kListGenBufferAlignment = 4096;

size_t align_up_to(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

bool snode_can_use_spirv_listgen(const SNode *snode) {
  if (snode == nullptr || snode->type == SNodeType::root ||
      snode->is_place() || snode->is_path_all_dense) {
    return false;
  }
  return snode->type == SNodeType::dense ||
         snode->type == SNodeType::bitmasked ||
         snode->type == SNodeType::pointer ||
         snode->type == SNodeType::hash ||
         snode->type == SNodeType::dynamic;
}

size_t estimate_listgen_entries(const CompiledSNodeStructs &compiled_structs) {
  size_t result = 0;
  for (const auto &[sid, desc] : compiled_structs.snode_descriptors) {
    (void)sid;
    if (snode_can_use_spirv_listgen(desc.snode)) {
      size_t entries = desc.total_num_cells_from_root;
      size_t suffix = 1;
      for (auto *sn = desc.snode; sn != nullptr && sn->type != SNodeType::root;
           sn = sn->parent) {
        const auto it = compiled_structs.snode_descriptors.find(sn->id);
        if (it != compiled_structs.snode_descriptors.end() &&
            sn->type == SNodeType::hash && it->second.hash_table_capacity > 0) {
          entries = it->second.hash_table_capacity * suffix;
          break;
        }
        suffix *= static_cast<size_t>(sn->num_cells_per_container);
      }
      result = std::max(result, entries);
    }
  }
  return result;
}

// Info for launching a compiled Taichi kernel, which consists of a series of
// Unified Device API pipelines.

CompiledTaichiKernel::CompiledTaichiKernel(const Params &ti_params)
    : ti_kernel_attribs_(*ti_params.ti_kernel_attribs),
      device_(ti_params.device) {
  input_buffers_[BufferType::GlobalTmps] = ti_params.global_tmps_buffer;
  input_buffers_[BufferType::HashOverflow] = ti_params.hash_overflow_buffer;
  input_buffers_[BufferType::ListGen] = ti_params.listgen_buffer;

  // Compiled_structs can be empty if loading a kernel from an AOT module as
  // the SNode are not re-compiled/structured. In this case, we assume a
  // single root buffer size configured from the AOT module.
  for (int root = 0; root < ti_params.num_snode_trees; ++root) {
    BufferInfo buffer = {BufferType::Root, root};
    input_buffers_[buffer] = ti_params.root_buffers[root];
  }
#if defined(TI_WITH_VULKAN_POINTER)
  // B-3.b (2026-05): 注册独立 NodeAllocatorPool buffer。key 中 root_id[0] = sid。
  // OFF 默认该 vector 为空，与历史 input_buffers_ 内容字节等价。
  for (const auto &[sid, alloc] : ti_params.node_allocator_pool_buffers) {
    BufferInfo buffer = {BufferType::NodeAllocatorPool, sid};
    input_buffers_[buffer] = alloc;
  }
  // C-2.5 (2026-05): 注册 chunked allocator 的全部 chunk DeviceAllocation
  // 列表。dispatch 时按 BufferBind.chunk_count > 0 用 rw_buffer_array(N)
  // 把 N 个 buffer 一并 bind 到单个 descriptor array binding。
  for (const auto &[sid, allocs] : ti_params.node_allocator_chunk_arrays) {
    BufferInfo buffer = {BufferType::NodeAllocatorPool, sid};
    chunk_arrays_[buffer] = allocs;
  }
#endif

  const auto arg_sz = ti_kernel_attribs_.ctx_attribs.args_bytes();
  const auto ret_sz = ti_kernel_attribs_.ctx_attribs.rets_bytes();

  args_buffer_size_ = arg_sz;
  ret_buffer_size_ = ret_sz;

  const auto &ctx_attribs = ti_kernel_attribs_.ctx_attribs;
  for (const auto &kv : ctx_attribs.args()) {
    const auto &indices = kv.first;
    const auto &arg = kv.second;
    if (!arg.is_array) {
      continue;
    }
    RuntimeArrayArg array_arg;
    array_arg.indices = indices;
    array_arg.data_ptr_indices = indices;
    array_arg.data_ptr_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
    array_arg.grad_ptr_indices = indices;
    array_arg.grad_ptr_indices.push_back(TypeFactory::GRAD_PTR_POS_IN_NDARRAY);
    for (const auto &access : ctx_attribs.arr_access) {
      if (access.first == indices) {
        array_arg.access = uint32_t(access.second);
        break;
      }
    }
    for (const auto &access : ctx_attribs.grad_arr_access) {
      if (access.first == indices) {
        array_arg.grad_access = uint32_t(access.second);
        break;
      }
    }
    runtime_array_args_.push_back(std::move(array_arg));
  }

  for (const auto &kv : ctx_attribs.argpack_types()) {
    runtime_argpack_args_.push_back(kv.first);
  }

  const auto &task_attribs = ti_kernel_attribs_.tasks_attribs;
  const auto &spirv_bins = ti_params.spirv_bins;
  TI_ASSERT(task_attribs.size() == spirv_bins.size());
  cached_resource_sets_.resize(task_attribs.size());
  buffer_binding_plans_.resize(task_attribs.size());
  task_uses_listgen_buffer_.resize(task_attribs.size(), false);

  for (int i = 0; i < task_attribs.size(); ++i) {
    if (task_attribs[i].task_type == OffloadedTaskType::listgen) {
      required_listgen_capacity_entries_ = std::max(
          required_listgen_capacity_entries_,
          static_cast<size_t>(
              std::max(task_attribs[i].advisory_total_num_threads, 0)));
    }
    auto &binding_plans = buffer_binding_plans_[i];
    binding_plans.reserve(task_attribs[i].buffer_binds.size());
    for (const auto &bind : task_attribs[i].buffer_binds) {
      if (bind.buffer.type == BufferType::ListGen) {
        task_uses_listgen_buffer_[i] = true;
      }
      BufferBindingPlan plan;
      plan.buffer = bind.buffer;
      plan.binding = bind.binding;
      plan.chunk_count = bind.chunk_count;
      if (bind.binding < 0) {
        plan.kind = BufferBindingKind::Skip;
      } else if (bind.buffer.type == BufferType::ExtArr) {
        plan.kind = BufferBindingKind::ExtArrRw;
      } else if (bind.buffer.type == BufferType::Args) {
        plan.kind = BufferBindingKind::Args;
      } else if (bind.buffer.type == BufferType::ArgPack) {
        plan.kind = BufferBindingKind::ArgPack;
      } else if (bind.buffer.type == BufferType::Rets) {
        plan.kind = BufferBindingKind::RetsRw;
      } else if (bind.buffer.type == BufferType::ListGen) {
        plan.kind = BufferBindingKind::StaticLookupRw;
      } else {
        if (bind.buffer.type == BufferType::NodeAllocatorPool &&
            bind.chunk_count > 0u) {
          auto chunk_it = chunk_arrays_.find(bind.buffer);
          if (chunk_it != chunk_arrays_.end()) {
            plan.kind = BufferBindingKind::ChunkedRwArray;
            plan.chunk_array = &chunk_it->second;
            binding_plans.push_back(plan);
            continue;
          }
        }
        plan.kind = BufferBindingKind::StaticRw;
        auto alloc_it = input_buffers_.find(bind.buffer);
        if (alloc_it != input_buffers_.end()) {
          plan.static_alloc = alloc_it->second;
        }
      }
      binding_plans.push_back(plan);
    }

    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary,
                                   (void *)spirv_bins[i].data(),
                                   spirv_bins[i].size() * sizeof(uint32_t)};
    auto [vp, res] = ti_params.device->create_pipeline_unique(
        source_desc, task_attribs[i].name, ti_params.backend_cache);
    pipelines_.push_back(std::move(vp));
  }
}

const TaichiKernelAttributes &CompiledTaichiKernel::ti_kernel_attribs() const {
  return ti_kernel_attribs_;
}

size_t CompiledTaichiKernel::num_pipelines() const {
  return pipelines_.size();
}

size_t CompiledTaichiKernel::get_args_buffer_size() const {
  return args_buffer_size_;
}

size_t CompiledTaichiKernel::get_ret_buffer_size() const {
  return ret_buffer_size_;
}

Pipeline *CompiledTaichiKernel::get_pipeline(int i) {
  return pipelines_[i].get();
}

ShaderResourceSet *CompiledTaichiKernel::get_cached_resource_set(int i) {
  if (!cached_resource_sets_[i]) {
    cached_resource_sets_[i] = device_->create_resource_set_unique();
  }
  return cached_resource_sets_[i].get();
}

GfxRuntime::GfxRuntime(const Params &params)
    : device_(params.device),
      profiler_(params.profiler),
      listgen_dynamic_size_(params.listgen_dynamic_size),
      listgen_explicit_size_(params.listgen_buffer_MB > 0),
      dispatch_cache_(params.dispatch_cache),
      listgen_reuse_(params.listgen_reuse),
      listgen_reuse_adaptive_(params.listgen_reuse_adaptive),
      ctx_buffer_ring_enabled_(params.ctx_buffer_ring),
      ctx_buffer_ring_size_(
          params.ctx_buffer_ring_size > 0
              ? static_cast<size_t>(params.ctx_buffer_ring_size)
              : size_t{8}),
      cmdlist_lazy_submit_enabled_(params.cmdlist_lazy_submit),
      cmdlist_lazy_submit_min_dispatches_(
          params.cmdlist_max_dispatches > 0
              ? static_cast<size_t>(params.cmdlist_max_dispatches)
              : size_t{0}),
      debug_mode_(params.debug) {
  graph_replay_registry_ = std::make_shared<GraphReplayRegistry>(this);
  TI_ERROR_IF(params.listgen_buffer_MB < 0,
              "vulkan_listgen_buffer_MB must be >= 0, got {}",
              params.listgen_buffer_MB);
  if (listgen_explicit_size_) {
    listgen_initial_buffer_size_ =
        static_cast<size_t>(params.listgen_buffer_MB) * 1024u * 1024u;
  } else if (listgen_dynamic_size_) {
    listgen_initial_buffer_size_ = 0;
  } else {
    listgen_initial_buffer_size_ = kListGenBufferSize;
  }
  current_cmdlist_pending_since_ = high_res_clock::now();
  init_nonroot_buffers();

  // Read pipeline cache from disk if available.
  std::filesystem::path cache_path(get_repo_dir());
  cache_path /= "rhi_cache.bin";
  std::vector<char> cache_data;
  constexpr uintmax_t kMaxPipelineCacheBytes = 256u * 1024u * 1024u;
  std::error_code cache_ec;
  const bool cache_exists = std::filesystem::exists(cache_path, cache_ec);
  if (cache_exists && !cache_ec) {
    const uintmax_t cache_size =
        std::filesystem::file_size(cache_path, cache_ec);
    if (cache_ec) {
      TI_WARN("Ignoring unreadable Vulkan pipeline cache at {}: {}",
              cache_path.generic_string(), cache_ec.message());
    } else if (cache_size > kMaxPipelineCacheBytes) {
      TI_WARN("Ignoring oversized Vulkan pipeline cache at {} ({} bytes)",
              cache_path.generic_string(), cache_size);
    } else if (cache_size != 0) {
      TI_TRACE("Loading pipeline cache from {}", cache_path.generic_string());
      std::ifstream cache_file(cache_path, std::ios::binary);
      if (cache_file) {
        cache_data.resize(static_cast<size_t>(cache_size));
        cache_file.read(cache_data.data(),
                        static_cast<std::streamsize>(cache_data.size()));
        if (!cache_file ||
            cache_file.gcount() !=
                static_cast<std::streamsize>(cache_data.size())) {
          TI_WARN("Ignoring truncated Vulkan pipeline cache at {}",
                  cache_path.generic_string());
          cache_data.clear();
        }
      }
    }
  } else if (!cache_ec) {
    TI_TRACE("Pipeline cache not found at {}", cache_path.generic_string());
  } else {
    TI_WARN("Failed to inspect Vulkan pipeline cache at {}: {}",
            cache_path.generic_string(), cache_ec.message());
  }
  auto cache_result = device_->create_pipeline_cache_unique(
      cache_data.size(), cache_data.empty() ? nullptr : cache_data.data());
  if (cache_result.second != RhiResult::success && !cache_data.empty()) {
    TI_WARN("Discarding incompatible Vulkan pipeline cache at {}",
            cache_path.generic_string());
    cache_data.clear();
    cache_result = device_->create_pipeline_cache_unique();
  }
  if (cache_result.second == RhiResult::success) {
    backend_cache_ = std::move(cache_result.first);
  }
}

GfxRuntime::~GfxRuntime() {
  // Close the registry before touching device state. A late graph-cache
  // destructor may retain the registry object, but it can no longer call back
  // into this runtime after close() returns.
  graph_replay_registry_->close();
  if (device_->backend_calls_safe()) {
    try {
      synchronize_impl(/*check_hash_overflow=*/false);
    } catch (const BackendRuntimeError &error) {
      // Destructors are noexcept. The reporter preserves the first fatal
      // backend error; teardown below only destroys host-owned wrappers and
      // Vulkan handles, without issuing another wait or submission.
      device_->report_backend_error(error);
    } catch (const std::exception &error) {
      TI_WARN("GfxRuntime teardown synchronization failed: {}", error.what());
    } catch (...) {
      TI_WARN("GfxRuntime teardown synchronization failed");
    }
  }
  graph_replay_states_.clear();
  graph_replay_retirement_head_ = 0;
  graph_replay_retirement_tail_ = 0;
  graph_replay_retirement_count_ = 0;

  // Write pipeline cache back to disk.
  if (backend_cache_) {
    uint8_t *cache_data = nullptr;
    size_t cache_size = 0;
    if (device_->backend_calls_safe()) {
      cache_data = (uint8_t *)backend_cache_->data();
      cache_size = backend_cache_->size();
    }
    if (cache_data) {
      // C4 (2026-04-26): atomic write — write to <path>.tmp then rename.
      // Avoids leaving a half-written rhi_cache.bin on the disk if the
      // process is killed mid-write, which would cause the next startup
      // to fail in create_pipeline_cache_unique() with a corrupted blob.
      // (Periodic in-flight flush and cross-process file locks were
      // considered and dropped as low-ROI; see compile_doc/优化总规划.md
      // §7.2 row 4.)
      std::filesystem::path cache_path =
          std::filesystem::path(get_repo_dir()) / "rhi_cache.bin";
      std::filesystem::path tmp_path = cache_path;
      tmp_path += ".tmp";
      bool wrote_ok = false;
      {
        std::ofstream cache_file(tmp_path,
                                 std::ios::binary | std::ios::trunc);
        if (cache_file) {
          cache_file.write(reinterpret_cast<const char *>(cache_data),
                           static_cast<std::streamsize>(cache_size));
          cache_file.flush();
          wrote_ok = static_cast<bool>(cache_file);
        }
      }
      if (wrote_ok) {
        std::error_code ec;
        std::filesystem::rename(tmp_path, cache_path, ec);
        if (ec) {
          // Fallback: remove dest then rename. Some filesystems on Windows
          // refuse cross-existing rename without explicit replace.
          std::filesystem::remove(cache_path, ec);
          std::filesystem::rename(tmp_path, cache_path, ec);
          if (ec) {
            TI_TRACE("Failed to atomically install rhi_cache.bin: {}",
                     ec.message());
            std::filesystem::remove(tmp_path, ec);
          }
        }
      } else {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
      }
    }
    backend_cache_.reset();
  }

  {
    decltype(ti_kernels_) tmp;
    tmp.swap(ti_kernels_);
  }
  ti_kernel_snode_tree_ids_.clear();
  global_tmps_buffer_.reset();
  listgen_buffer_.reset();
}

int64 GfxRuntime::get_sparse_list_version(int snode_id) const {
  auto it = sparse_list_states_.find(snode_id);
  if (it == sparse_list_states_.end()) {
    return 0;
  }
  return it->second.version;
}

bool GfxRuntime::sparse_list_task_is_current(
    const TaskAttributes &attribs) {
  const bool is_listgen =
      attribs.sparse_list_op == TaskAttributes::kSparseListOpListgen &&
      attribs.sparse_list_snode_id >= 0 &&
      attribs.sparse_list_parent_snode_id >= 0;
  SparseListgenNodeStatistics *telemetry = nullptr;
  if (sparse_listgen_telemetry_enabled_ && is_listgen) {
    telemetry = &sparse_listgen_telemetry_[attribs.sparse_list_snode_id];
    telemetry->snode_id = attribs.sparse_list_snode_id;
    telemetry->parent_snode_id = attribs.sparse_list_parent_snode_id;
    ++telemetry->requests;
    if (!telemetry->resident_evictions.available) {
      telemetry->resident_evictions = {0, true};
    }
  }
  if (!is_listgen) {
    return false;
  }
  if (!listgen_reuse_) {
    if (telemetry != nullptr) {
      telemetry->last_rebuild_reason = "reuse_disabled";
    }
    return false;
  }
  if (resident_sparse_list_snode_id_ != attribs.sparse_list_snode_id) {
    if (telemetry != nullptr) {
      const bool cold =
          sparse_list_states_.find(attribs.sparse_list_snode_id) ==
          sparse_list_states_.end();
      telemetry->last_rebuild_reason =
          cold ? "cold" : "resident_list_evicted";
      if (!cold) {
        ++telemetry->resident_evictions.value;
      }
    }
    return false;
  }

  auto it = sparse_list_states_.find(attribs.sparse_list_snode_id);
  if (it == sparse_list_states_.end()) {
    if (telemetry != nullptr) {
      telemetry->last_rebuild_reason = "cold";
    }
    if (listgen_reuse_adaptive_) {
      auto &state = sparse_list_states_[attribs.sparse_list_snode_id];
      record_sparse_list_reuse_sample(state, /*would_skip=*/false);
    }
    return false;
  }
  auto &state = it->second;
  const bool would_skip = state.clean_epoch == state.dirty_epoch &&
                 state.global_dirty_seen ==
                   sparse_list_global_dirty_epoch_ &&
                 state.clean_parent_version ==
                   get_sparse_list_version(
                     attribs.sparse_list_parent_snode_id);
  record_sparse_list_reuse_sample(state, would_skip);
  if (telemetry != nullptr) {
    if (would_skip && !state.adaptive_disabled) {
      ++telemetry->reuse_hits;
    } else if (state.clean_epoch != state.dirty_epoch) {
      telemetry->last_rebuild_reason = "topology_dirty";
    } else if (state.global_dirty_seen != sparse_list_global_dirty_epoch_) {
      telemetry->last_rebuild_reason = "global_topology_dirty";
    } else if (state.clean_parent_version !=
               get_sparse_list_version(
                   attribs.sparse_list_parent_snode_id)) {
      telemetry->last_rebuild_reason = "parent_version_changed";
    } else if (state.adaptive_disabled) {
      telemetry->last_rebuild_reason = "adaptive_reuse_disabled";
    } else {
      telemetry->last_rebuild_reason = "not_current";
    }
  }
  return would_skip && !state.adaptive_disabled;
}

void GfxRuntime::record_sparse_list_reuse_sample(SparseListState &state,
                                                 bool would_skip) const {
  if (!listgen_reuse_adaptive_) {
    return;
  }
  constexpr int kWindow = 64;
  constexpr int kDisablePercent = 10;
  constexpr int kEnablePercent = 15;
  const int hit = would_skip ? 1 : 0;
  if (state.adaptive_window_size < kWindow) {
    state.adaptive_window_size++;
  } else {
    state.adaptive_hit_count -=
        int((state.adaptive_window_bits >> (kWindow - 1)) & 1u);
  }
  state.adaptive_window_bits = (state.adaptive_window_bits << 1) |
                               static_cast<std::uint64_t>(hit);
  state.adaptive_hit_count += hit;

  if (state.adaptive_window_size < kWindow) {
    return;
  }
  if (!state.adaptive_disabled &&
      state.adaptive_hit_count * 100 < kDisablePercent * kWindow) {
    state.adaptive_disabled = true;
  } else if (state.adaptive_disabled &&
             state.adaptive_hit_count * 100 >= kEnablePercent * kWindow) {
    state.adaptive_disabled = false;
  }
}

void GfxRuntime::mark_sparse_list_task_launched(
    const TaskAttributes &attribs) {
  if (attribs.task_type != OffloadedTaskType::listgen) {
    return;
  }
  if (sparse_listgen_telemetry_enabled_ &&
      attribs.sparse_list_snode_id >= 0) {
    auto &telemetry =
        sparse_listgen_telemetry_[attribs.sparse_list_snode_id];
    telemetry.snode_id = attribs.sparse_list_snode_id;
    telemetry.parent_snode_id = attribs.sparse_list_parent_snode_id;
    ++telemetry.rebuilds;
    const std::uint64_t candidates = static_cast<std::uint64_t>(
        std::max(attribs.advisory_total_num_threads, 0));
    if (!telemetry.candidate_slots_dispatched.available) {
      telemetry.candidate_slots_dispatched = {0, true};
    }
    telemetry.candidate_slots_dispatched.value += candidates;
  }
  if (!listgen_reuse_) {
    return;
  }
  if (attribs.sparse_list_op != TaskAttributes::kSparseListOpListgen ||
      attribs.sparse_list_snode_id < 0 ||
      attribs.sparse_list_parent_snode_id < 0) {
    resident_sparse_list_snode_id_ = -1;
    return;
  }

  auto &state = sparse_list_states_[attribs.sparse_list_snode_id];
  if (state.parent_snode_id != attribs.sparse_list_parent_snode_id) {
    if (state.parent_snode_id >= 0) {
      auto old_children = child_lists_by_parent_.find(state.parent_snode_id);
      if (old_children != child_lists_by_parent_.end()) {
        old_children->second.erase(attribs.sparse_list_snode_id);
        if (old_children->second.empty()) {
          child_lists_by_parent_.erase(old_children);
        }
      }
    }
    child_lists_by_parent_[attribs.sparse_list_parent_snode_id].insert(
        attribs.sparse_list_snode_id);
    state.parent_snode_id = attribs.sparse_list_parent_snode_id;
  }
  state.clean_epoch = state.dirty_epoch;
  state.clean_parent_version =
      get_sparse_list_version(attribs.sparse_list_parent_snode_id);
  state.global_dirty_seen = sparse_list_global_dirty_epoch_;
  state.version++;
  resident_sparse_list_snode_id_ = attribs.sparse_list_snode_id;
}

void GfxRuntime::invalidate_sparse_list_cache(int sparse_mutation_snode_id) {
  if (!listgen_reuse_) {
    return;
  }
  if (sparse_mutation_snode_id < 0) {
    sparse_list_global_dirty_epoch_++;
    if (sparse_listgen_telemetry_enabled_) {
      for (const auto &[snode_id, state] : sparse_list_states_) {
        (void)state;
        auto &telemetry = sparse_listgen_telemetry_[snode_id];
        telemetry.snode_id = snode_id;
        ++telemetry.invalidations;
      }
    }
    return;
  }

  std::unordered_set<int> affected;
  std::vector<int> stack;
  stack.push_back(sparse_mutation_snode_id);
  while (!stack.empty()) {
    const int snode_id = stack.back();
    stack.pop_back();
    if (!affected.insert(snode_id).second) {
      continue;
    }
    auto children = child_lists_by_parent_.find(snode_id);
    if (children != child_lists_by_parent_.end()) {
      for (int child_snode_id : children->second) {
        stack.push_back(child_snode_id);
      }
    }
  }
  for (int snode_id : affected) {
    auto it = sparse_list_states_.find(snode_id);
    if (it != sparse_list_states_.end()) {
      it->second.dirty_epoch++;
      if (sparse_listgen_telemetry_enabled_) {
        auto &telemetry = sparse_listgen_telemetry_[snode_id];
        telemetry.snode_id = snode_id;
        ++telemetry.invalidations;
      }
    }
  }
}

void GfxRuntime::clear_sparse_list_cache_resident() {
  sparse_list_states_.clear();
  child_lists_by_parent_.clear();
  sparse_list_global_dirty_epoch_ = 0;
  resident_sparse_list_snode_id_ = -1;
}

void GfxRuntime::register_hash_overflow_checks(
    int root_id,
    const CompiledSNodeStructs &compiled_structs) {
  for (const auto &[sid, desc] : compiled_structs.snode_descriptors) {
    if (desc.snode == nullptr || desc.snode->type != SNodeType::hash ||
        desc.hash_table_capacity == 0 ||
        desc.snode->parent == nullptr ||
        desc.snode->parent->type != SNodeType::root) {
      continue;
    }
    size_t base_offset = 0;
    for (const SNode *sn = desc.snode;
         sn != nullptr && sn->type != SNodeType::root; sn = sn->parent) {
      const auto it = compiled_structs.snode_descriptors.find(sn->id);
      TI_ASSERT_INFO(it != compiled_structs.snode_descriptors.end(),
                     "Hash SNode {} is missing from compiled descriptors",
                     sn->id);
      base_offset += it->second.mem_offset_in_parent_cell;
    }
    const size_t no_offset = static_cast<size_t>(-1);
    hash_overflow_watches_.push_back(
        {root_id, sid, base_offset + desc.hash_overflow_count_offset,
         base_offset + desc.hash_active_count_offset,
         desc.hash_tombstone_count_offset == no_offset
             ? no_offset
             : base_offset + desc.hash_tombstone_count_offset});
  }
}

void GfxRuntime::check_hash_overflow_counters() {
  if (hash_overflow_error_reported_) {
    return;
  }

  std::vector<DevicePtr> ptrs;
  std::vector<uint32_t> values;
  std::vector<void *> host_ptrs;
  std::vector<size_t> sizes;
  std::vector<HashOverflowWatch> live_watches;
  ptrs.reserve(hash_overflow_watches_.size());
  values.resize(hash_overflow_watches_.size());
  host_ptrs.reserve(hash_overflow_watches_.size());
  sizes.reserve(hash_overflow_watches_.size());
  live_watches.reserve(hash_overflow_watches_.size());

  for (const auto &watch : hash_overflow_watches_) {
    if (watch.root_id < 0 ||
        watch.root_id >= static_cast<int>(root_buffers_.size()) ||
        !root_buffers_[watch.root_id]) {
      continue;
    }
    ptrs.push_back(
        root_buffers_[watch.root_id]->get_ptr(watch.overflow_byte_offset));
    host_ptrs.push_back(&values[ptrs.size() - 1]);
    sizes.push_back(sizeof(uint32_t));
    live_watches.push_back(watch);
  }
  int first_overflow = -1;
  if (!ptrs.empty()) {
    auto status = device_->readback_data(ptrs.data(), host_ptrs.data(),
                                         sizes.data(), int(ptrs.size()));
    TI_ERROR_IF(status != RhiResult::success,
                "Failed to read Hash SNode overflow counters.");

    for (int i = 0; i < static_cast<int>(live_watches.size()); ++i) {
      if (values[i] != 0) {
        first_overflow = i;
        break;
      }
    }
  }

  if (first_overflow >= 0) {
    const auto &watch = live_watches[first_overflow];
    uint32_t active_count = 0;
    uint32_t tombstone_count = 0;
    if (watch.root_id >= 0 &&
        watch.root_id < static_cast<int>(root_buffers_.size()) &&
        root_buffers_[watch.root_id]) {
      std::vector<DevicePtr> diag_ptrs;
      std::vector<void *> diag_host_ptrs;
      std::vector<size_t> diag_sizes;
      diag_ptrs.push_back(
          root_buffers_[watch.root_id]->get_ptr(watch.active_byte_offset));
      diag_host_ptrs.push_back(&active_count);
      diag_sizes.push_back(sizeof(uint32_t));
      if (watch.tombstone_byte_offset != static_cast<size_t>(-1)) {
        diag_ptrs.push_back(
            root_buffers_[watch.root_id]->get_ptr(watch.tombstone_byte_offset));
        diag_host_ptrs.push_back(&tombstone_count);
        diag_sizes.push_back(sizeof(uint32_t));
      }
      auto diag_status = device_->readback_data(
          diag_ptrs.data(), diag_host_ptrs.data(), diag_sizes.data(),
          int(diag_ptrs.size()));
      if (diag_status != RhiResult::success) {
        active_count = 0;
        tombstone_count = 0;
      }
    }
    hash_overflow_error_reported_ = true;
    TI_ERROR(
        "Hash SNode table overflow on root {} SNode {}. Increase "
        "capacity=... or expected_active=... before materialization, or "
        "rebuild the SNode tree after high churn. active_count={}, "
        "tombstone_count={}.",
        watch.root_id, watch.snode_id, active_count, tombstone_count);
  }

  if (!hash_overflow_buffer_) {
    return;
  }
  std::array<uint32_t, 6> aggregate{};
  std::array<void *, 1> aggregate_host_ptrs{aggregate.data()};
  std::array<DevicePtr, 1> aggregate_ptrs{
      hash_overflow_buffer_->get_ptr(0)};
  std::array<size_t, 1> aggregate_sizes{aggregate.size() * sizeof(uint32_t)};
  auto aggregate_status = device_->readback_data(
      aggregate_ptrs.data(), aggregate_host_ptrs.data(),
      aggregate_sizes.data(), int(aggregate_ptrs.size()));
  TI_ERROR_IF(aggregate_status != RhiResult::success,
              "Failed to read sparse SNode overflow diagnostics.");
  if (aggregate[0] == 0) {
    return;
  }
  const auto overflow_kind =
      static_cast<spirv::SparseOverflowKind>(aggregate[5]);
  if (overflow_kind == spirv::SparseOverflowKind::Pointer) {
    hash_overflow_error_reported_ = true;
    TI_ERROR(
        "Pointer SNode pool overflow on root {} SNode {}. The configured "
        "capacity is {} cells. Increase vk_max_active or remove the reduced "
        "capacity hint before materialization.",
        aggregate[1], aggregate[2], aggregate[3]);
  }
  if (overflow_kind == spirv::SparseOverflowKind::Dynamic) {
    hash_overflow_error_reported_ = true;
    TI_ERROR(
        "Dynamic SNode capacity overflow on root {} SNode {}. The configured "
        "capacity is {} cells.",
        aggregate[1], aggregate[2], aggregate[3]);
  }
  uint32_t active_count = 0;
  uint32_t tombstone_count = 0;
  active_count = aggregate[3];
  tombstone_count = aggregate[4];
  hash_overflow_error_reported_ = true;
  TI_ERROR(
      "Hash SNode table overflow on root {} SNode {}. Increase "
      "capacity=... or expected_active=... before materialization, or rebuild "
      "the SNode tree after high churn. active_count={}, tombstone_count={}.",
      aggregate[1], aggregate[2], active_count, tombstone_count);
}

GfxRuntime::KernelHandle GfxRuntime::register_taichi_kernel(
    GfxRuntime::RegisterParams reg_params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  CompiledTaichiKernel::Params params;
  params.ti_kernel_attribs = &(reg_params.kernel_attribs);
  params.num_snode_trees = reg_params.num_snode_trees;
  params.device = device_;
  params.root_buffers = {};
  for (int root = 0; root < root_buffers_.size(); ++root) {
    params.root_buffers.push_back(root_buffers_[root].get());
  }
  params.global_tmps_buffer = global_tmps_buffer_.get();
  params.hash_overflow_buffer = hash_overflow_buffer_.get();
  params.listgen_buffer = listgen_buffer_.get();
#if defined(TI_WITH_VULKAN_POINTER)
  // B-3.b (2026-05): 枚举所有独立 pool buffer。allocator->independent_pool_alloc()
  // 返回 nullptr 表示该 SNode 仍走 root_buffer 子区间（在 vector 中跳过）。
  // C-2.2 (2026-05): independent_pool_alloc() 已升级为 DeviceNodeAllocator 基类
  // 虚方法，BumpOnly 与 Chunked skeleton 都能透出 chunk[0] DeviceAllocation。
  for (const auto &[rid, sid_to_alloc] : node_allocators_) {
    (void)rid;
    for (const auto &[sid, allocator_ptr] : sid_to_alloc) {
      if (allocator_ptr == nullptr) {
        continue;
      }
      DeviceAllocation *indep = allocator_ptr->independent_pool_alloc();
      if (indep != nullptr) {
        params.node_allocator_pool_buffers.emplace_back(sid, indep);
      }
      // C-2.5 (2026-05): chunked allocator 在 rw_buffer_array 路径下需要
      // 全部 chunk 的 DeviceAllocation 列表。
      // §13.4 (2026-05-02): 任意 chunk 数（含 1）都走 chunk_arrays 路径，
      // 与 spirv_codegen.cpp::lookup_chunked_pool_contract 同步——后者已
      // 移除 max_chunks>1 限制；codegen 与 runtime 必须对 chunked 是否
      // 走 descriptor array 给出相同答案，避免 binding 不匹配 (§13.4
      // ptr_to_chunk_idx_ vs ptr_to_buffers_ 不变量)。
      auto chunk_list = allocator_ptr->chunks();
      if (!chunk_list.empty() && allocator_ptr->is_chunked()) {
        params.node_allocator_chunk_arrays.emplace_back(
            sid, std::move(chunk_list));
      }
    }
  }
#endif
  params.backend_cache = backend_cache_.get();

  for (int i = 0; i < reg_params.task_spirv_source_codes.size(); ++i) {
    const auto &spirv_src = reg_params.task_spirv_source_codes[i];

    // If we can reach here, we have succeeded. Otherwise
    // std::optional::value() would have killed us.
    params.spirv_bins.push_back(std::move(spirv_src));
  }
  KernelHandle res;
  TI_ERROR_IF(next_ti_kernel_id_ == std::numeric_limits<int>::max(),
              "GFX kernel registration ID space exhausted; call ti.reset().");
  const int launch_id = next_ti_kernel_id_;
  auto compiled_kernel = std::make_unique<CompiledTaichiKernel>(params);
  const auto [kernel_iter, kernel_inserted] =
      ti_kernels_.emplace(launch_id, std::move(compiled_kernel));
  TI_ASSERT(kernel_inserted);
  try {
    const bool tree_was_inserted =
        ti_kernel_snode_tree_ids_
            .emplace(launch_id, std::move(reg_params.snode_tree_ids))
            .second;
    TI_ASSERT(tree_was_inserted);
  } catch (...) {
    ti_kernels_.erase(kernel_iter);
    throw;
  }
  ++next_ti_kernel_id_;
  res.set_launch_id(launch_id);
  return res;
}

void GfxRuntime::retire_snode_tree_kernels(int tree_id) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  // Graph replay executables cache raw CompiledTaichiKernel pointers. The
  // Program destroy transaction synchronized the device before entering here,
  // so dropping every replay recording is safe. Registrations remain alive;
  // unrelated graphs simply record again on their next run.
  graph_replay_states_.clear();
  graph_replay_retirement_head_ = 0;
  graph_replay_retirement_tail_ = 0;
  graph_replay_retirement_count_ = 0;
  TI_ASSERT(ti_kernel_snode_tree_ids_.size() == ti_kernels_.size());
  bool retired_any = false;
  std::vector<int> retired_sparse_snode_ids;
  for (auto iter = ti_kernel_snode_tree_ids_.begin();
       iter != ti_kernel_snode_tree_ids_.end();) {
    const auto &tree_ids = iter->second;
    if (!std::binary_search(tree_ids.begin(), tree_ids.end(), tree_id)) {
      ++iter;
      continue;
    }
    const int launch_id = iter->first;
    const auto kernel_it = ti_kernels_.find(launch_id);
    TI_ASSERT(kernel_it != ti_kernels_.end());
    for (const auto &attribs :
         kernel_it->second->ti_kernel_attribs().tasks_attribs) {
      if (attribs.sparse_list_snode_id >= 0) {
        retired_sparse_snode_ids.push_back(
            attribs.sparse_list_snode_id);
      }
    }
    TI_ASSERT(ti_kernels_.erase(launch_id) == 1);
    iter = ti_kernel_snode_tree_ids_.erase(iter);
    retired_any = true;
  }
  if (retired_any) {
    clear_sparse_list_cache_resident();
    for (int snode_id : retired_sparse_snode_ids) {
      sparse_listgen_telemetry_.erase(snode_id);
    }
  }
}

std::size_t GfxRuntime::debug_registered_kernel_count() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  return ti_kernels_.size();
}

void GfxRuntime::debug_reset_sparse_listgen_statistics() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  sparse_listgen_telemetry_.clear();
  sparse_listgen_telemetry_enabled_ = true;
}

SparseSNodeTreeListgenStatistics
GfxRuntime::debug_sparse_listgen_statistics(
    const std::vector<int> &snode_ids) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  SparseSNodeTreeListgenStatistics result;
  result.available = sparse_listgen_telemetry_enabled_;
  if (!result.available) {
    return result;
  }
  for (const auto &[snode_id, telemetry] : sparse_listgen_telemetry_) {
    if (std::binary_search(snode_ids.begin(), snode_ids.end(), snode_id)) {
      result.nodes.push_back(telemetry);
    }
  }
  std::sort(result.nodes.begin(), result.nodes.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.snode_id < rhs.snode_id;
            });
  return result;
}

void GfxRuntime::launch_kernel(KernelHandle handle,
                               LaunchContextBuilder &host_ctx) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  auto kernel_iter = ti_kernels_.find(handle.get_launch_id());
  TI_ASSERT(handle.get_launch_id() >= 0 && kernel_iter != ti_kernels_.end());
  auto *ti_kernel = kernel_iter->second.get();

#if defined(__APPLE__)
  if (profiler_) {
    const int apple_max_query_pool_count = 32;
    int task_count = ti_kernel->ti_kernel_attribs().tasks_attribs.size();
    if (task_count > apple_max_query_pool_count) {
      TI_WARN(
          "Cannot concurrently profile more than 32 tasks in a single Taichi "
          "kernel. Profiling aborted.");
      profiler_ = nullptr;
    } else if (device_->profiler_get_sampler_count() + task_count >
               apple_max_query_pool_count) {
      flush();
      device_->profiler_sync();
    }
  }
#endif

  ensure_listgen_capacity_for_kernel(*ti_kernel);

  std::unique_ptr<DeviceAllocationGuard> args_buffer{nullptr},
      ret_buffer{nullptr};

  if (ti_kernel->get_args_buffer_size()) {
    if (auto pooled = acquire_ctx_buffer(ti_kernel->get_args_buffer_size(),
                                         AllocUsage::Uniform)) {
      args_buffer = std::move(pooled);
    } else {
      auto [buf, res] = device_->allocate_memory_unique(
          {ti_kernel->get_args_buffer_size(),
           /*host_write=*/true, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Uniform});
      TI_ASSERT_INFO(res == RhiResult::success,
                     "Failed to allocate args buffer");
      args_buffer = std::move(buf);
    }
  }

  if (ti_kernel->get_ret_buffer_size()) {
    if (auto pooled = acquire_ctx_buffer(ti_kernel->get_ret_buffer_size(),
                                         AllocUsage::Storage)) {
      ret_buffer = std::move(pooled);
    } else {
      auto [buf, res] = device_->allocate_memory_unique(
          {ti_kernel->get_ret_buffer_size(),
           /*host_write=*/false, /*host_read=*/true,
           /*export_sharing=*/false, AllocUsage::Storage});
      TI_ASSERT_INFO(res == RhiResult::success, "Failed to allocate ret buffer");
      ret_buffer = std::move(buf);
    }
  }

  // Create context blitter
  auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
      &ti_kernel->ti_kernel_attribs().ctx_attribs,
      &ti_kernel->runtime_array_args(), host_ctx, device_, args_buffer.get(),
      ret_buffer.get());

  // `any_arrays` contain both external arrays and NDArrays
  std::unordered_map<std::vector<int>, DeviceAllocation,
                     hashing::Hasher<std::vector<int>>>
      any_arrays;
  // `ext_array_size` only holds the size of external arrays (host arrays)
  // As buffer size information is only needed when it needs to be allocated
  // and transferred by the host
  std::unordered_map<std::vector<int>, size_t,
                     hashing::Hasher<std::vector<int>>>
      ext_array_size;
  std::unordered_map<std::vector<int>, DeviceAllocation,
                     hashing::Hasher<std::vector<int>>>
      textures;
  // `argpacks` holds argpacks that passed to this kernel.
  std::unordered_map<std::vector<int>, const ArgPack *,
                     hashing::Hasher<std::vector<int>>>
      argpacks;
  std::vector<DeviceAllocation> argpack_allocations;

  // Prepare context buffers & arrays
  if (ctx_blitter) {
    TI_ASSERT(ti_kernel->get_args_buffer_size() ||
              ti_kernel->get_ret_buffer_size());

    for (const auto &array_arg : ti_kernel->runtime_array_args()) {
      const auto &indices = array_arg.indices;
      const auto alloc_type = host_ctx.device_allocation_type[indices];
      if (alloc_type != LaunchContextBuilder::DevAllocType::kNone) {
          if (alloc_type ==
              LaunchContextBuilder::DevAllocType::kAccelerationStructure) {
            continue;
          }
          DeviceAllocation devalloc = kDeviceNullAllocation;
          // NDArray
          if (host_ctx.array_ptrs.count(array_arg.data_ptr_indices)) {
            devalloc =
                *(DeviceAllocation *)(
                    host_ctx.array_ptrs[array_arg.data_ptr_indices]);
          }
          // Texture
          if (host_ctx.array_ptrs.count(indices)) {
            devalloc = *(DeviceAllocation *)(host_ctx.array_ptrs[indices]);
          }

          if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray ||
              alloc_type ==
                  LaunchContextBuilder::DevAllocType::kDenseStorage) {
            if (alloc_type ==
                LaunchContextBuilder::DevAllocType::kDenseStorage) {
              devalloc =
                  host_ctx.get_resolved_dense_storage(indices).allocation;
            }
            any_arrays[indices] = devalloc;
            any_arrays[array_arg.data_ptr_indices] = devalloc;
            ndarrays_in_use_.insert(devalloc.alloc_id);
            if (alloc_type ==
                LaunchContextBuilder::DevAllocType::kNdarray) {
              auto grad_it =
                  host_ctx.array_ptrs.find(array_arg.grad_ptr_indices);
              if (grad_it != host_ctx.array_ptrs.end() &&
                  grad_it->second != nullptr) {
                auto grad_alloc = *(DeviceAllocation *)(grad_it->second);
                any_arrays[array_arg.grad_ptr_indices] = grad_alloc;
                ndarrays_in_use_.insert(grad_alloc.alloc_id);
              }
            }
          } else if (alloc_type == LaunchContextBuilder::DevAllocType::kTexture) {
            textures[indices] = devalloc;
          } else if (alloc_type ==
                     LaunchContextBuilder::DevAllocType::kRWTexture) {
            textures[indices] = devalloc;
          } else {
            TI_NOT_IMPLEMENTED;
          }
        } else {
          ext_array_size[indices] = host_ctx.array_runtime_sizes[indices];
          uint32_t access = array_arg.access;
          // Alloc ext arr
          size_t alloc_size = std::max(size_t(32), ext_array_size.at(indices));
          // Bug B/C fix (forge 2026-05): host_write must be true regardless
          // of the kernel's READ/WRITE access pattern, because we now always
          // blit host→device for ext arrs (see host_to_device above) and the
          // buffer therefore must be host-visible. The previous code keyed
          // host_write off the READ flag, but mis-named it host_write — it's
          // really "host-can-map".
          (void)access;
          bool host_write = true;
          auto [allocated, res] = device_->allocate_memory_unique(
              {alloc_size, host_write, false, /*export_sharing=*/false,
               AllocUsage::Storage});
          TI_ASSERT_INFO(res == RhiResult::success,
                         "Failed to allocate ext arr buffer");
          any_arrays[indices] = *allocated.get();
          any_arrays[array_arg.data_ptr_indices] = *allocated.get();
          ctx_buffers_.push_back(std::move(allocated));
          if (array_arg.grad_access != 0) {
            auto grad_it =
                host_ctx.array_ptrs.find(array_arg.grad_ptr_indices);
            TI_ERROR_IF(
                grad_it == host_ctx.array_ptrs.end() ||
                    grad_it->second == nullptr,
                "External ndarray gradient storage is required by this "
                "GFX kernel but no gradient pointer was supplied");
            auto [grad_allocated, grad_res] =
                device_->allocate_memory_unique(
                    {alloc_size, /*host_write=*/true, /*host_read=*/false,
                     /*export_sharing=*/false, AllocUsage::Storage});
            TI_ASSERT_INFO(grad_res == RhiResult::success,
                           "Failed to allocate external ndarray gradient buffer");
            any_arrays[array_arg.grad_ptr_indices] = *grad_allocated.get();
            ctx_buffers_.push_back(std::move(grad_allocated));
          }
        }
    }

    for (const auto &indices : ti_kernel->runtime_argpack_args()) {
      TI_ASSERT(host_ctx.device_allocation_type[indices] ==
                LaunchContextBuilder::DevAllocType::kArgPack);
      TI_ASSERT(host_ctx.argpack_ptrs.count(indices));
      const ArgPack *argpack = host_ctx.argpack_ptrs[indices];
      DeviceAllocation devalloc = argpack->get_device_allocation();
      argpacks_in_use_.insert(devalloc.alloc_id);
      argpacks[indices] = argpack;
      argpack_allocations.push_back(devalloc);
    }

    ctx_blitter->host_to_device(any_arrays, ext_array_size, argpacks);
  }

  ensure_current_cmdlist();

  // Record commands
  const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;
  bool argpack_barriers_inserted = false;

  auto mark_storage_buffer_write = [&](const BufferBind &bind) {
    if (!bind.may_write()) {
      return;
    }
    switch (bind.buffer.type) {
      case BufferType::Args:
      case BufferType::ArgPack:
        return;
      case BufferType::ExtArr:
        add_pending_dispatch_barrier(any_arrays.at(bind.buffer.root_id));
        return;
      case BufferType::Rets:
        if (ret_buffer) {
          add_pending_dispatch_barrier(*ret_buffer);
        }
        return;
      case BufferType::NodeAllocatorPool:
        if (bind.chunk_count > 0u) {
          if (bind.chunk_count > 1u) {
            pending_dispatch_global_barrier_ = true;
            return;
          }
          if (auto *chunks = ti_kernel->get_chunk_array(bind.buffer)) {
            for (const auto &chunk : *chunks) {
              add_pending_dispatch_barrier(chunk);
            }
            return;
          }
        }
        break;
      default:
        break;
    }
    DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
    if (alloc != nullptr) {
      add_pending_dispatch_barrier(*alloc);
    }
  };
  auto mark_task_writes = [&](const TaskAttributes &attribs) {
    if (!dispatch_cache_) {
      return;
    }
    for (const auto &bind : attribs.texture_binds) {
      if (bind.is_storage) {
        pending_dispatch_global_barrier_ = true;
        return;
      }
    }
    for (const auto &bind : attribs.buffer_binds) {
      mark_storage_buffer_write(bind);
    }
  };

  for (int i = 0; i < task_attribs.size(); ++i) {
    const auto &attribs = task_attribs[i];
    if (sparse_list_task_is_current(attribs)) {
      TI_TRACE("Skipping current Vulkan sparse list kernel {}", attribs.name);
      continue;
    }
    if (ti_kernel->task_uses_listgen_buffer(i)) {
      // VS-1.3 hardening: any dispatched task that binds the shared listgen
      // buffer (writer listgen or reader struct_for) makes a later grow/replace
      // require synchronization. This is deliberately more conservative than
      // tracking only OffloadedTaskType::listgen and protects future AOT / sparse
      // reader paths from replacing an in-flight listgen allocation.
      listgen_buffer_used_ = true;
    }
    insert_pending_dispatch_barriers();
    if (!argpack_barriers_inserted) {
      for (DeviceAllocation alloc : argpack_allocations) {
        current_cmdlist_->buffer_barrier(alloc);
      }
      argpack_barriers_inserted = true;
    }
    auto vp = ti_kernel->get_pipeline(i);
    const int group_x = (attribs.advisory_total_num_threads +
                         attribs.advisory_num_threads_per_group - 1) /
                        attribs.advisory_num_threads_per_group;
    std::unique_ptr<ShaderResourceSet> one_shot_bindings;
    ShaderResourceSet *bindings = nullptr;
    if (dispatch_cache_) {
      bindings = ti_kernel->get_cached_resource_set(i);
    } else {
      one_shot_bindings = device_->create_resource_set_unique();
      bindings = one_shot_bindings.get();
    }
    for (const auto &bind : ti_kernel->buffer_binding_plan(i)) {
      if (bind.binding < 0) {
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::StaticRw) {
        bindings->rw_buffer(bind.binding,
                            bind.static_alloc ? *bind.static_alloc
                                              : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind ==
          CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray) {
        TI_ASSERT_INFO(
            bind.chunk_array->size() == bind.chunk_count,
            "C-2.5: chunk_count mismatch sid={}, attribs={} runtime={}",
            bind.buffer.root_id, bind.chunk_count, bind.chunk_array->size());
        bindings->rw_buffer_array(bind.binding, *bind.chunk_array);
        continue;
      }
      if (bind.kind ==
          CompiledTaichiKernel::BufferBindingKind::StaticLookupRw) {
        DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
        bindings->rw_buffer(bind.binding,
                            alloc ? *alloc : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::ExtArrRw) {
        if (host_ctx.device_allocation_type[bind.buffer.root_id] ==
            LaunchContextBuilder::DevAllocType::kDenseStorage) {
          const auto &binding =
              host_ctx.get_resolved_dense_storage(bind.buffer.root_id);
          if (binding.byte_size == 0) {
            bindings->rw_buffer(bind.binding, binding.allocation);
          } else {
            bindings->rw_buffer(bind.binding, binding.device_ptr(),
                                binding.byte_size);
          }
        } else {
          bindings->rw_buffer(bind.binding,
                              any_arrays.at(bind.buffer.root_id));
        }
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::Args) {
        bindings->buffer(bind.binding,
                         args_buffer ? *args_buffer : kDeviceNullAllocation);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::ArgPack) {
        DeviceAllocation alloc =
            argpacks.at(bind.buffer.root_id)->get_device_allocation();
        bindings->buffer(bind.binding, alloc);
        continue;
      }
      if (bind.kind == CompiledTaichiKernel::BufferBindingKind::RetsRw) {
        bindings->rw_buffer(bind.binding,
                            ret_buffer ? *ret_buffer : kDeviceNullAllocation);
        continue;
      }
      TI_NOT_IMPLEMENTED;
    }

    for (auto &bind : attribs.texture_binds) {
      DeviceAllocation texture = textures.at(bind.arg_id);
      if (bind.is_storage) {
        transition_image(texture, ImageLayout::shader_read_write);
        bindings->rw_image(bind.binding, texture, 0);
      } else {
        transition_image(texture, ImageLayout::shader_read);
        const auto sampler = image_sampler_configs_.find(texture.alloc_id);
        bindings->image(bind.binding, texture,
                        sampler == image_sampler_configs_.end()
                            ? ImageSamplerConfig{}
                            : sampler->second);
      }
    }

    for (const auto &bind : attribs.acceleration_structure_binds) {
      const int arg_offset = host_ctx.args_type->get_element_offset(bind.arg_id);
      const auto found = std::find_if(
          host_ctx.acceleration_structure_ptrs.begin(),
          host_ctx.acceleration_structure_ptrs.end(),
          [arg_offset](const auto &ref) {
            return ref.arg_offset == arg_offset;
          });
      TI_ERROR_IF(found == host_ctx.acceleration_structure_ptrs.end() ||
                      found->owner == nullptr || found->handle == 0,
                  "Vulkan acceleration-structure kernel argument is unbound");
      found->owner->vulkan_bind_ray_kernel_resource(
          found->handle, bindings, bind.binding, current_cmdlist_.get());
    }

    if (attribs.task_type == OffloadedTaskType::listgen) {
      for (auto &bind : attribs.buffer_binds) {
        if (bind.buffer.type == BufferType::ListGen) {
          // Bug D fix (forge 2026-05): only zero the count slot (4 bytes
          // at offset 0), not the entire 32MB listgen buffer. The listgen
          // kernel writes count via atomic_add starting from 0 and writes
          // entries to listgen[1 + slot] for each unique slot it claims;
          // the consuming struct_for only reads listgen[1..count], so any
          // stale data past `count` is unreachable. Filling the full 32MB
          // every dispatch was costing ~0.15ms per listgen task on
          // discrete GPUs (sparse warm 0.785 -> 0.539 ms expected).
          // FIXME: properly support multiple lists at distinct offsets.
          current_cmdlist_->buffer_fill(
              ti_kernel->get_buffer_bind(bind.buffer)->get_ptr(0),
              /*size=*/sizeof(uint32_t),
              /*data=*/0);
          if (dispatch_cache_) {
            current_cmdlist_->buffer_barrier(
                ti_kernel->get_buffer_bind(bind.buffer)->get_ptr(0),
                sizeof(uint32_t));
          } else {
            current_cmdlist_->buffer_barrier(
                *ti_kernel->get_buffer_bind(bind.buffer));
          }
        }
      }
    }

    current_cmdlist_->bind_pipeline(vp);
    RhiResult status = current_cmdlist_->bind_shader_resources(bindings);
    TI_ERROR_IF(status != RhiResult::success,
                "Resource binding error : RhiResult({})", status);

    std::string trace_name;
    std::optional<ScopedExternalProfilerAnnotation> annotation;
    if (!host_ctx.dispatch_label().empty()) {
      trace_name = make_labeled_task_name(
          attribs.name, attribs.task_id, host_ctx.dispatch_label());
      annotation.emplace(trace_name);
    }
    if (profiler_) {
      current_cmdlist_->begin_profiler_scope(
          trace_name.empty() ? attribs.name : trace_name);
    }

    status = current_cmdlist_->dispatch(group_x);

    if (profiler_) {
      current_cmdlist_->end_profiler_scope();
    }

    TI_ERROR_IF(status != RhiResult::success, "Dispatch error : RhiResult({})",
                status);
    ++current_cmdlist_dispatch_count_;
    mark_sparse_list_task_launched(attribs);
    if (attribs.may_mutate_sparse_topology) {
      invalidate_sparse_list_cache(attribs.sparse_mutation_snode_id);
    }
    if (dispatch_cache_) {
      mark_task_writes(attribs);
    } else {
      current_cmdlist_->memory_barrier();
    }
  }

  // Keep context buffers used in this dispatch
  if (ti_kernel->get_args_buffer_size()) {
    if (ctx_buffer_pool_enabled()) {
      pending_pool_.push_back({std::move(args_buffer),
                               ti_kernel->get_args_buffer_size(),
                               AllocUsage::Uniform});
    } else {
      ctx_buffers_.push_back(std::move(args_buffer));
    }
  }
  if (ti_kernel->get_ret_buffer_size()) {
    if (ctx_buffer_pool_enabled()) {
      pending_pool_.push_back({std::move(ret_buffer),
                               ti_kernel->get_ret_buffer_size(),
                               AllocUsage::Storage});
    } else {
      ctx_buffers_.push_back(std::move(ret_buffer));
    }
  }

  // If we need to host sync, sync and remove in-flight references
  if (ctx_blitter) {
    insert_pending_dispatch_barriers();
    if (ctx_blitter->device_to_host(current_cmdlist_.get(), any_arrays,
                                    ext_array_size)) {
      current_cmdlist_ = nullptr;
      current_cmdlist_dispatch_count_ = 0;
      ctx_buffers_.clear();
      // R2.a: device_to_host internally syncs (wait_idle / submit_synced),
      // so all pool buffers are safe to recycle here.
      if (ctx_buffer_pool_enabled()) {
        recycle_pools_to_free();
      }
    }
  }

  submit_current_cmdlist_if_timeout();
}

void GfxRuntime::GraphReplayExecutable::reset() {
  cached_prepared.clear();
  cached_prepare_key.clear();
  slots.clear();
  next_slot = 0;
  device = nullptr;
}

void GfxRuntime::GraphReplayExecutable::bind_device(Device *new_device) {
  if (device != new_device) {
    reset();
    device = new_device;
  }
}

bool GfxRuntime::GraphReplayExecutable::refresh_prepared_cache(
    const std::vector<uint64_t> &key,
    std::vector<PreparedDispatch> &prepared) {
  const bool cache_hit =
      cached_prepare_key == key && cached_prepared.size() == prepared.size();
  if (!cache_hit) {
    cached_prepared.clear();
    cached_prepared.resize(prepared.size());
    for (size_t i = 0; i < prepared.size(); ++i) {
      auto &cache = cached_prepared[i];
      auto &pd = prepared[i];
      cache.any_arrays.clear();
      for (const auto &array_arg : pd.kernel->runtime_array_args()) {
        const auto alloc_type =
            pd.host_ctx->device_allocation_type[array_arg.indices];
        DeviceAllocation devalloc = kDeviceNullAllocation;
        if (alloc_type == LaunchContextBuilder::DevAllocType::kDenseStorage) {
          devalloc =
              pd.host_ctx->get_resolved_dense_storage(array_arg.indices)
                  .allocation;
        } else {
          TI_ASSERT(alloc_type ==
                    LaunchContextBuilder::DevAllocType::kNdarray);
          auto data_it =
              pd.host_ctx->array_ptrs.find(array_arg.data_ptr_indices);
          TI_ASSERT(data_it != pd.host_ctx->array_ptrs.end() &&
                    data_it->second != nullptr);
          devalloc = *static_cast<DeviceAllocation *>(data_it->second);
        }
        cache.any_arrays[array_arg.indices] = devalloc;
        cache.any_arrays[array_arg.data_ptr_indices] = devalloc;

        if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray) {
          auto grad_it =
              pd.host_ctx->array_ptrs.find(array_arg.grad_ptr_indices);
          if (grad_it != pd.host_ctx->array_ptrs.end() &&
              grad_it->second != nullptr) {
            DeviceAllocation grad_alloc =
                *static_cast<DeviceAllocation *>(grad_it->second);
            cache.any_arrays[array_arg.grad_ptr_indices] = grad_alloc;
          }
        }
      }
    }
    cached_prepare_key = key;
  }
  for (size_t i = 0; i < prepared.size(); ++i) {
    prepared[i].any_arrays = &cached_prepared[i].any_arrays;
  }
  return cache_hit;
}

GfxRuntime::GraphReplayExecutable::Slot *
GfxRuntime::GraphReplayExecutable::acquire_ready_slot(
    const std::vector<uint64_t> &key,
    const std::vector<uint64_t> &structure_key,
    bool wait_on_saturation) {
  if (slots.empty()) {
    slots.resize(kReplaySlots);
  }

  auto ready = [](const Slot &slot) {
    return !slot.completion || slot.completion->is_ready();
  };
  auto select = [&](int preference) -> std::optional<size_t> {
    for (size_t i = 0; i < slots.size(); ++i) {
      const size_t candidate = (next_slot + i) % slots.size();
      const auto &slot = slots[candidate];
      if (!ready(slot)) {
        continue;
      }
      const bool matches =
          preference == 0
              ? slot.recorded && slot.key == key
              : preference == 1
                    ? slot.recorded && slot.structure_key == structure_key
                    : preference == 2 ? !slot.recorded : true;
      if (matches) {
        return candidate;
      }
    }
    return std::nullopt;
  };

  // Prefer exact replay, then descriptor-only patch, then an unused slot.
  // Re-recording a structurally unrelated ready slot is the last resort.
  for (int preference = 0; preference != 4; ++preference) {
    if (auto candidate = select(preference)) {
      next_slot = (*candidate + 1) % slots.size();
      return &slots[*candidate];
    }
  }

  if (wait_on_saturation) {
    // A compound structured submission can consume all fixed replay slots.
    // Waiting at the allocation boundary prevents a later invocation from
    // publishing a prefix of its chunks and then failing halfway through.
    // Queue execution does not require host_api_mutex_, so the oldest
    // round-robin slot can make progress while this host call is blocked.
    auto &oldest = slots[next_slot];
    if (oldest.completion && oldest.completion->wait()) {
      for (int preference = 0; preference != 4; ++preference) {
        if (auto candidate = select(preference)) {
          next_slot = (*candidate + 1) % slots.size();
          return &slots[*candidate];
        }
      }
    }
  }

  graph_replay_slot_saturation_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
  return nullptr;
}

bool GfxRuntime::GraphReplayExecutable::ready_for_retirement() const {
  for (const auto &slot : slots) {
    if (slot.completion && !slot.completion->is_ready()) {
      return false;
    }
  }
  return true;
}

uint64_t
GfxRuntime::GraphReplayExecutable::known_persistent_argument_bytes() const {
  uint64_t bytes = 0;
  for (const auto &slot : slots) {
    for (size_t size : slot.args_buffer_sizes) {
      bytes += size;
    }
    bytes += slot.structured_control_bytes;
    bytes += slot.structured_observation_bytes;
  }
  return bytes;
}

void GfxRuntime::GraphReplayState::reset() {
  fixed_submit = {};
  fixed_argument_bytes = 0;
  fixed_secondary = false;
  executable.reset();
  attempts = 0;
  recorded = 0;
  replayed = 0;
  patched = 0;
  fallbacks = 0;
  structural_fallbacks = 0;
  runtime_mode_fallbacks = 0;
  slot_saturation_fallbacks = 0;
  effect_reads = 0;
  effect_writes = 0;
  dependency_barriers = 0;
  exit_barriers = 0;
  barrier_deferrals = 0;
  rar_elisions = 0;
  last_path = GraphReplayLastPath::none;
  last_fallback_reason = GraphReplayFallbackReason::none;
  diagnostics_enabled = false;
  retirement_requested = false;
  retirement_next = 0;
}

std::unique_ptr<GraphReplayRegistration> GfxRuntime::register_graph_replay(
    uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ASSERT(replay_token != 0);
  const uint64_t replay_key = next_graph_replay_registration_id_++;
  TI_ASSERT(replay_key != 0);
  return std::unique_ptr<GraphReplayRegistration>(
      new GraphReplayRegistration(graph_replay_registry_, replay_key));
}

bool GfxRuntime::owns_graph_replay_registration(
    const GraphReplayRegistration &registration) const {
  return registration.registry_.get() == graph_replay_registry_.get();
}

void GfxRuntime::collect_ready_graph_replays(std::size_t max_polls) {
  const std::size_t poll_count =
      std::min(max_polls, graph_replay_retirement_count_);
  for (std::size_t i = 0; i < poll_count; ++i) {
    const uint64_t replay_token = graph_replay_retirement_head_;
    TI_ASSERT(replay_token != 0);
    auto state = graph_replay_states_.find(replay_token);
    TI_ASSERT(state != graph_replay_states_.end());
    TI_ASSERT(state->second.retirement_requested);
    const uint64_t next = state->second.retirement_next;
    // Query before mutating the queue. If a backend readiness check throws,
    // the retirement remains discoverable and can be retried or released by
    // synchronize/reset instead of becoming a permanently orphaned state.
    if (!state->second.executable.ready_for_retirement()) {
      if (graph_replay_retirement_count_ > 1) {
        TI_ASSERT(next != 0 && graph_replay_retirement_tail_ != replay_token);
        auto tail = graph_replay_states_.find(graph_replay_retirement_tail_);
        TI_ASSERT(tail != graph_replay_states_.end());
        TI_ASSERT(tail->second.retirement_next == 0);
        tail->second.retirement_next = replay_token;
        graph_replay_retirement_head_ = next;
        graph_replay_retirement_tail_ = replay_token;
        state->second.retirement_next = 0;
      }
      continue;
    }
    graph_replay_retirement_head_ = next;
    --graph_replay_retirement_count_;
    if (graph_replay_retirement_count_ == 0) {
      graph_replay_retirement_head_ = 0;
      graph_replay_retirement_tail_ = 0;
    } else {
      TI_ASSERT(graph_replay_retirement_head_ != 0);
    }
    graph_replay_states_.erase(state);
  }
}

void GfxRuntime::retire_graph_replay(uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  auto active = graph_replay_states_.find(replay_token);
  if (active == graph_replay_states_.end()) {
    return;
  }
  if (active->second.retirement_requested) {
    collect_ready_graph_replays(kGraphReplayRetirementPollBudget);
    return;
  }
  // Registration destruction reaches this path. Linking the state itself
  // avoids allocation and therefore remains safe under host-memory pressure.
  active->second.retirement_requested = true;
  active->second.retirement_next = 0;
  if (graph_replay_retirement_tail_ == 0) {
    TI_ASSERT(graph_replay_retirement_head_ == 0 &&
              graph_replay_retirement_count_ == 0);
    graph_replay_retirement_head_ = replay_token;
  } else {
    auto tail = graph_replay_states_.find(graph_replay_retirement_tail_);
    TI_ASSERT(tail != graph_replay_states_.end());
    TI_ASSERT(tail->second.retirement_next == 0);
    tail->second.retirement_next = replay_token;
  }
  graph_replay_retirement_tail_ = replay_token;
  ++graph_replay_retirement_count_;
  collect_ready_graph_replays(kGraphReplayRetirementPollBudget);
}

GraphReplayStats GfxRuntime::debug_graph_replay_stats(
    uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  // A report can opt in before the first launch. Materialize only the tiny
  // host state here (not replay slots or GPU resources) so the next launch
  // observes diagnostics_enabled from its first attempt.
  auto [it, inserted] = graph_replay_states_.try_emplace(replay_token);
  (void)inserted;
  GraphReplayState &state = it->second;
  GraphReplayStats result{
      state.attempts,
      state.recorded,
      state.replayed,
      state.patched,
      state.fallbacks,
      state.structural_fallbacks,
      state.runtime_mode_fallbacks,
      state.slot_saturation_fallbacks,
      state.fixed_submit ? state.fixed_argument_bytes
                         : state.executable.known_persistent_argument_bytes(),
      state.effect_reads,
      state.effect_writes,
      state.dependency_barriers,
      state.exit_barriers,
      state.barrier_deferrals,
      state.rar_elisions,
      state.last_path,
      state.last_fallback_reason,
      state.fixed_secondary,
  };
  state.diagnostics_enabled = true;
  return result;
}

GraphReplayStats GfxRuntime::snapshot_graph_replay_stats(
    uint64_t replay_token) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  const auto found = graph_replay_states_.find(replay_token);
  if (found == graph_replay_states_.end()) {
    return {};
  }
  const GraphReplayState &state = found->second;
  return GraphReplayStats{
      state.attempts,
      state.recorded,
      state.replayed,
      state.patched,
      state.fallbacks,
      state.structural_fallbacks,
      state.runtime_mode_fallbacks,
      state.slot_saturation_fallbacks,
      state.fixed_submit ? state.fixed_argument_bytes
                         : state.executable.known_persistent_argument_bytes(),
      state.effect_reads,
      state.effect_writes,
      state.dependency_barriers,
      state.exit_barriers,
      state.barrier_deferrals,
      state.rar_elisions,
      state.last_path,
      state.last_fallback_reason,
      state.fixed_secondary,
  };
}

namespace {
bool bind_graph_task(GfxRuntime::GraphReplayExecutable::PreparedDispatch &pd,
                     int task_index,
                     ShaderResourceSet *bindings,
                     bool patch_existing) {
  for (const auto &bind : pd.kernel->buffer_binding_plan(task_index)) {
    if (bind.binding < 0) {
      continue;
    }
    switch (bind.kind) {
      case CompiledTaichiKernel::BufferBindingKind::Skip:
        break;
      case CompiledTaichiKernel::BufferBindingKind::StaticRw:
        bindings->rw_buffer(bind.binding, bind.static_alloc
                                              ? *bind.static_alloc
                                              : kDeviceNullAllocation);
        break;
      case CompiledTaichiKernel::BufferBindingKind::StaticLookupRw: {
        auto *allocation = pd.kernel->get_buffer_bind(bind.buffer);
        bindings->rw_buffer(bind.binding,
                            allocation ? *allocation : kDeviceNullAllocation);
        break;
      }
      case CompiledTaichiKernel::BufferBindingKind::ExtArrRw:
        if (pd.host_ctx->device_allocation_type[bind.buffer.root_id] ==
            LaunchContextBuilder::DevAllocType::kDenseStorage) {
          const auto &binding =
              pd.host_ctx->get_resolved_dense_storage(bind.buffer.root_id);
          if (binding.byte_size == 0) {
            bindings->rw_buffer(bind.binding, binding.allocation);
          } else {
            bindings->rw_buffer(bind.binding, binding.device_ptr(),
                                binding.byte_size);
          }
        } else {
          bindings->rw_buffer(bind.binding,
                              pd.any_arrays->at(bind.buffer.root_id));
        }
        break;
      case CompiledTaichiKernel::BufferBindingKind::Args:
        bindings->buffer(bind.binding, pd.args_buffer ? *pd.args_buffer
                                                      : kDeviceNullAllocation);
        break;
      case CompiledTaichiKernel::BufferBindingKind::ArgPack:
      case CompiledTaichiKernel::BufferBindingKind::RetsRw:
      case CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray:
        TI_NOT_IMPLEMENTED;
    }
  }
  return bindings->prepare_for_replay(patch_existing) == RhiResult::success;
}
}  // namespace

std::unique_ptr<GraphReplayRegistration> GfxRuntime::prepare_fixed_graph(
    const std::vector<GraphRecordingOperation> &operations,
    std::vector<std::shared_ptr<void>> owners) {
  // On a cold failure release the host API lock before registration retirement
  // (registry -> runtime is also the order used by reset and late destruction).
  auto registration = register_graph_replay(1);
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(
      operations.empty() || profiler_ || dispatch_cache_,
      "Prepared Vulkan Graph requires nonempty uninstrumented commands");
  // Build transactionally before publishing a registration. No mathematical
  // dispatch or queue submission occurs during argument/descriptor preparation.
  GraphReplayState state;
  auto &executable = state.executable;
  executable.bind_device(device_);
  auto payload = std::make_shared<GraphReplayExecutable::Slot>();
  auto &slot = *payload;
  slot.retained_owners = std::move(owners);
  std::vector<GraphReplayExecutable::PreparedDispatch> prepared;
  for (const auto &operation : operations) {
    if (operation.external) {
      continue;
    }
    const auto &dispatch = operation.dispatch;
    auto found = ti_kernels_.find(dispatch.handle.get_launch_id());
    TI_ERROR_IF(!dispatch.host_ctx || found == ti_kernels_.end() ||
                    dispatch.indirect_dispatch != kDeviceNullPtr,
                "Prepared Vulkan Graph requires fixed kernel dispatches");
    auto *kernel = found->second.get();
    TI_ERROR_IF(kernel->get_ret_buffer_size() ||
                    !kernel->runtime_argpack_args().empty(),
                "Prepared Vulkan Graph cannot retain host returns or ArgPacks");
    for (const auto &array : kernel->runtime_array_args()) {
      const auto kind =
          dispatch.host_ctx->device_allocation_type[array.indices];
      TI_ERROR_IF(kind != LaunchContextBuilder::DevAllocType::kNdarray &&
                      kind != LaunchContextBuilder::DevAllocType::kDenseStorage,
                  "Prepared Vulkan Graph requires owned device arrays");
    }
    const auto &tasks = kernel->ti_kernel_attribs().tasks_attribs;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      TI_ERROR_IF(
          !tasks[i].texture_binds.empty() ||
              !tasks[i].acceleration_structure_binds.empty() ||
              tasks[i].task_type == OffloadedTaskType::listgen ||
              tasks[i].may_mutate_sparse_topology ||
              kernel->task_uses_listgen_buffer(i),
          "Prepared Vulkan Graph requires topology-stable buffer tasks");
      for (const auto &bind : kernel->buffer_binding_plan(i)) {
        TI_ERROR_IF(
            bind.kind == CompiledTaichiKernel::BufferBindingKind::ArgPack ||
                bind.kind == CompiledTaichiKernel::BufferBindingKind::RetsRw ||
                bind.kind ==
                    CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray,
            "Unsupported prepared Vulkan Graph resource binding");
      }
    }
    prepared.push_back({kernel, dispatch.host_ctx});
  }
  executable.refresh_prepared_cache({1}, prepared);
  slot.args_buffers.resize(prepared.size());
  slot.args_buffer_sizes.resize(prepared.size());
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    auto &pd = prepared[i];
    const auto size = pd.kernel->get_args_buffer_size();
    if (size) {
      auto [buffer, status] = device_->allocate_memory_unique(
          {size, true, false, false, AllocUsage::Uniform});
      TI_ERROR_IF(status != RhiResult::success,
                  "Prepared Vulkan Graph argument allocation failed");
      slot.args_buffers[i] = std::move(buffer);
      slot.args_buffer_sizes[i] = size;
    }
    pd.args_buffer = slot.args_buffers[i].get();
    auto blitter = HostDeviceContextBlitter::maybe_make(
        &pd.kernel->ti_kernel_attribs().ctx_attribs,
        &pd.kernel->runtime_array_args(), *pd.host_ctx, device_, pd.args_buffer,
        nullptr);
    if (blitter) {
      blitter->host_to_device(*pd.any_arrays, {}, {});
    }
  }
  std::unique_ptr<CommandList> commands;
  if (std::all_of(operations.begin(), operations.end(),
                  [](const auto &operation) {
                    return !operation.external || operation.inline_recording;
                  })) {
    commands = device_->get_compute_stream()->new_secondary_command_list();
  }
  state.fixed_secondary = bool(commands);
  if (!commands) {
    auto [primary, status] =
        device_->get_compute_stream()->new_command_list_unique();
    TI_ERROR_IF(status != RhiResult::success,
                "Prepared Vulkan Graph command allocation failed");
    commands = std::move(primary);
  }
  commands->memory_barrier();
  std::size_t kernel_index = 0;
  for (const auto &operation : operations) {
    if (operation.external) {
      operation.external(device_, commands.get());
      commands->memory_barrier();
      continue;
    }
    auto &pd = prepared[kernel_index++];
    const auto &tasks = pd.kernel->ti_kernel_attribs().tasks_attribs;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      auto resources = device_->create_resource_set_unique();
      TI_ERROR_IF(!bind_graph_task(pd, i, resources.get(), false),
                  "Prepared Vulkan Graph descriptor preparation failed");
      commands->bind_pipeline(pd.kernel->get_pipeline(i));
      TI_ERROR_IF(commands->bind_shader_resources(resources.get()) !=
                      RhiResult::success,
                  "Prepared Vulkan Graph resource binding failed");
      const auto &task = tasks[i];
      const int groups = (task.advisory_total_num_threads +
                          task.advisory_num_threads_per_group - 1) /
                         task.advisory_num_threads_per_group;
      TI_ERROR_IF(commands->dispatch(groups) != RhiResult::success,
                  "Prepared Vulkan Graph dispatch recording failed");
      commands->memory_barrier();
      slot.resource_sets.push_back(std::move(resources));
    }
  }
  for (auto bytes : slot.args_buffer_sizes) {
    state.fixed_argument_bytes += bytes;
  }
  if (state.fixed_secondary) {
    // payload.cmdlist deliberately stays empty: commands retain the payload,
    // never the reverse. This makes close-before-submission safe without
    // cycles.
    auto append = commands->finalize_secondary(std::move(payload));
    TI_ERROR_IF(!append, "Secondary Graph command finalization is unsupported");
    // The enclosing primary command owns the secondary and its complete
    // resource payload until retirement. No per-replay owner allocation,
    // fence poll, queue submit or semaphore is needed for this append.
    state.fixed_submit = [this, append = std::move(append)] {
      ensure_current_cmdlist();
      insert_pending_dispatch_barriers();
      append(current_cmdlist_.get());
    };
  } else {
    slot.cmdlist = std::move(commands);
    slot.recorded = true;
    executable.slots.push_back(std::move(slot));
    auto *primary_slot = &executable.slots.front();
    state.fixed_submit = [this, primary_slot] {
      flush_if_pending();
      primary_slot->completion =
          device_->get_compute_stream()->submit(primary_slot->cmdlist.get());
      latest_compute_completion_ = primary_slot->completion;
    };
  }
  state.last_path = GraphReplayLastPath::record;
  graph_replay_states_.emplace(registration->replay_key(), std::move(state));
  return registration;
}

void GfxRuntime::launch_prepared_graph(std::uint64_t replay_key) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  // Fixed argument images, commands and owners: no signature building,
  // descriptor updates, per-dispatch preparation, observation or ready-slot
  // poll. Same ordered queue permits simultaneous reuse of the immutable
  // command list.
  graph_replay_states_.at(replay_key).fixed_submit();
}

bool GfxRuntime::try_launch_graph(
    const std::vector<GraphDispatch> &dispatches,
    uint64_t replay_key,
    RuntimeStatistics *statistics,
    const GraphStructuredControl *structured_control,
    GraphStructuredResult *structured_result,
    const GraphNestedStructuredControl *nested_control,
    GraphNestedStructuredResult *nested_result) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  collect_ready_graph_replays(kGraphReplayRetirementPollBudget);
  TI_ASSERT(replay_key != 0);
  GraphReplayState &state = graph_replay_states_[replay_key];
  TI_ASSERT(!state.retirement_requested);
  GraphReplayExecutable &executable = state.executable;
  if (state.diagnostics_enabled) {
    ++state.attempts;
  }
  state.last_path = GraphReplayLastPath::none;
  state.last_fallback_reason = GraphReplayFallbackReason::none;
  if (structured_result != nullptr) {
    *structured_result = {};
  }
  if (nested_result != nullptr) {
    *nested_result = {};
  }
  if (structured_control == nullptr && structured_result != nullptr) {
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason =
        GraphReplayFallbackReason::structural_unsupported;
    return false;
  }
  if (nested_control == nullptr && nested_result != nullptr) {
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason =
        GraphReplayFallbackReason::structural_unsupported;
    return false;
  }
  if (structured_control != nullptr && nested_control != nullptr) {
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason =
        GraphReplayFallbackReason::structural_unsupported;
    return false;
  }
  if (dispatches.empty() || profiler_ || dispatch_cache_) {
    if (state.diagnostics_enabled) {
      ++state.fallbacks;
      ++state.runtime_mode_fallbacks;
    }
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason = GraphReplayFallbackReason::runtime_mode;
    return false;
  }

  using PreparedDispatch = GraphReplayExecutable::PreparedDispatch;
  using SizeMap = std::unordered_map<std::vector<int>,
                                     size_t,
                                     hashing::Hasher<std::vector<int>>>;
  using ArgPackMap =
      std::unordered_map<std::vector<int>,
                         const ArgPack *,
                         hashing::Hasher<std::vector<int>>>;

  std::vector<uint64_t> key;
  std::vector<uint64_t> structure_key;
  key.reserve(dispatches.size() * 16);
  structure_key.reserve(dispatches.size() * 12);
  std::unordered_map<DeviceAllocationId, uint64_t> allocation_ordinals;
  auto push_runtime_structure = [&](DeviceAllocation allocation,
                                    uint64_t offset, uint64_t bytes) {
    auto [it, inserted] = allocation_ordinals.try_emplace(
        allocation.alloc_id, allocation_ordinals.size() + 1);
    (void)inserted;
    push_graph_allocation_structure_key(structure_key, allocation,
                                        it->second, offset, bytes);
  };
  std::vector<PreparedDispatch> prepared;
  prepared.reserve(dispatches.size());
  size_t total_tasks = 0;
  std::vector<std::size_t> dispatch_task_offsets;
  dispatch_task_offsets.reserve(dispatches.size() + 1);
  dispatch_task_offsets.push_back(0);
  bool has_indirect_dispatch = false;
  GraphStructuredStrategy structured_strategy =
      GraphStructuredStrategy::automatic;

  auto reject = [&](GraphReplayFallbackReason reason =
                        GraphReplayFallbackReason::structural_unsupported) {
    if (state.diagnostics_enabled) {
      ++state.fallbacks;
      if (reason == GraphReplayFallbackReason::structural_unsupported) {
        ++state.structural_fallbacks;
      }
    }
    state.last_path = GraphReplayLastPath::fallback;
    state.last_fallback_reason = reason;
    return false;
  };

  executable.bind_device(device_);
  if (structured_control != nullptr) {
    const auto valid_scalar = [&](DevicePtr ptr) {
      return ptr.device == device_ && ptr.alloc_id != 0 &&
             ptr.offset % alignof(std::uint32_t) == 0;
    };
    const bool valid_strategy =
        structured_control->strategy == GraphStructuredStrategy::automatic ||
        structured_control->strategy == GraphStructuredStrategy::compact ||
        structured_control->strategy == GraphStructuredStrategy::chained ||
        structured_control->strategy ==
            GraphStructuredStrategy::conditional ||
        structured_control->strategy ==
            GraphStructuredStrategy::coarse_conditional;
    if (!valid_scalar(structured_control->predicate) ||
        !valid_scalar(structured_control->counter) ||
        (structured_control->has_status &&
         !valid_scalar(structured_control->status)) ||
        structured_control->initial_dispatch_count == 0 ||
        structured_control->initial_dispatch_count >= dispatches.size() ||
        !valid_strategy) {
      return reject();
    }
    key.insert(key.end(),
               {0xC000u,
                structured_control->initial_dispatch_count,
                structured_control->max_iterations,
                structured_control->has_status ? 1u : 0u,
                structured_result != nullptr ? 1u : 0u,
                static_cast<std::uint32_t>(structured_control->strategy),
                structured_control->predicate.alloc_id,
                structured_control->predicate.offset,
                structured_control->counter.alloc_id,
                structured_control->counter.offset,
                structured_control->status.alloc_id,
                structured_control->status.offset});
    structure_key.insert(
        structure_key.end(),
        {0xC000u, structured_control->initial_dispatch_count,
         structured_control->max_iterations,
         structured_control->has_status ? 1u : 0u,
         structured_result != nullptr ? 1u : 0u,
         static_cast<std::uint32_t>(structured_control->strategy)});
    push_runtime_structure(
        DeviceAllocation{structured_control->predicate.device,
                         structured_control->predicate.alloc_id},
        structured_control->predicate.offset, sizeof(std::uint32_t));
    push_runtime_structure(
        DeviceAllocation{structured_control->counter.device,
                         structured_control->counter.alloc_id},
        structured_control->counter.offset, sizeof(std::uint32_t));
    if (structured_control->has_status) {
      push_runtime_structure(
          DeviceAllocation{structured_control->status.device,
                           structured_control->status.alloc_id},
          structured_control->status.offset, sizeof(std::uint32_t));
    }
  } else if (nested_control != nullptr) {
    const auto valid_scalar = [&](DevicePtr ptr) {
      return ptr.device == device_ && ptr.alloc_id != 0 &&
             ptr.offset % alignof(std::uint32_t) == 0;
    };
    std::vector<DevicePtr> control_ptrs{
        nested_control->outer_predicate, nested_control->outer_counter};
    if (nested_control->outer_has_status) {
      control_ptrs.push_back(nested_control->outer_status);
    }
    for (const auto &inner : nested_control->inner_controls) {
      control_ptrs.push_back(inner.inner_predicate);
      control_ptrs.push_back(inner.inner_counter);
      if (inner.inner_has_status) {
        control_ptrs.push_back(inner.inner_status);
      }
    }
    bool controls_alias = false;
    for (std::size_t i = 0; i < control_ptrs.size(); ++i) {
      for (std::size_t j = i + 1; j < control_ptrs.size(); ++j) {
        if (control_ptrs[i].alloc_id == control_ptrs[j].alloc_id &&
            control_ptrs[i].offset == control_ptrs[j].offset) {
          controls_alias = true;
          break;
        }
      }
      if (controls_alias) {
        break;
      }
    }
    bool inner_valid = !nested_control->inner_controls.empty() &&
                       nested_control->inner_controls.size() <= 8;
    std::size_t boundary_cursor =
        nested_control->outer_condition_dispatch_count;
    for (const auto &inner : nested_control->inner_controls) {
      inner_valid =
          inner_valid && valid_scalar(inner.inner_predicate) &&
          valid_scalar(inner.inner_counter) &&
          (!inner.inner_has_status || valid_scalar(inner.inner_status)) &&
          boundary_cursor <= inner.inner_condition_dispatch_begin &&
          inner.inner_condition_dispatch_begin <
              inner.inner_body_dispatch_begin &&
          inner.inner_body_dispatch_begin < inner.inner_dispatch_end &&
          inner.inner_dispatch_end < dispatches.size() &&
          inner.inner_max_iterations > 0 &&
          inner.inner_max_iterations <= 64 && inner.inner_chunk_size > 0 &&
          inner.inner_chunk_size <= inner.inner_max_iterations &&
          inner.inner_chunk_size <= 64;
      boundary_cursor = inner.inner_dispatch_end;
    }
    if (!device_->supports_conditional_commands() ||
        !valid_scalar(nested_control->outer_predicate) ||
        !valid_scalar(nested_control->outer_counter) ||
        (nested_control->outer_has_status &&
         !valid_scalar(nested_control->outer_status)) ||
        controls_alias || !inner_valid ||
        nested_control->outer_condition_dispatch_count == 0 ||
        nested_control->outer_max_iterations == 0 ||
        nested_control->outer_max_iterations >
            kNestedStructuredMaximumOuterIterations) {
      return reject();
    }
    key.insert(
        key.end(),
        {0xC100u,
         nested_control->outer_condition_dispatch_count,
         nested_control->outer_max_iterations,
         nested_control->outer_has_status ? 1u : 0u,
         nested_control->inner_controls.size(),
         nested_control->outer_predicate.alloc_id,
         nested_control->outer_predicate.offset,
         nested_control->outer_counter.alloc_id,
         nested_control->outer_counter.offset,
         nested_control->outer_status.alloc_id,
         nested_control->outer_status.offset});
    structure_key.insert(
        structure_key.end(),
        {0xC100u,
         nested_control->outer_condition_dispatch_count,
         nested_control->outer_max_iterations,
         nested_control->outer_has_status ? 1u : 0u,
         nested_control->inner_controls.size()});
    const auto push_control_structure = [&](DevicePtr ptr) {
      push_runtime_structure(
          DeviceAllocation{ptr.device, ptr.alloc_id}, ptr.offset,
          sizeof(std::uint32_t));
    };
    push_control_structure(nested_control->outer_predicate);
    push_control_structure(nested_control->outer_counter);
    if (nested_control->outer_has_status) {
      push_control_structure(nested_control->outer_status);
    }
    for (const auto &inner : nested_control->inner_controls) {
      key.insert(key.end(),
                 {inner.inner_condition_dispatch_begin,
                  inner.inner_body_dispatch_begin,
                  inner.inner_dispatch_end, inner.inner_max_iterations,
                  inner.inner_chunk_size,
                  inner.inner_has_status ? 1u : 0u,
                  inner.inner_predicate.alloc_id,
                  inner.inner_predicate.offset,
                  inner.inner_counter.alloc_id,
                  inner.inner_counter.offset,
                  inner.inner_status.alloc_id,
                  inner.inner_status.offset});
      structure_key.insert(
          structure_key.end(),
          {inner.inner_condition_dispatch_begin,
           inner.inner_body_dispatch_begin, inner.inner_dispatch_end,
           inner.inner_max_iterations, inner.inner_chunk_size,
           inner.inner_has_status ? 1u : 0u});
      push_control_structure(inner.inner_predicate);
      push_control_structure(inner.inner_counter);
      if (inner.inner_has_status) {
        push_control_structure(inner.inner_status);
      }
    }
  }

  for (const GraphDispatch &dispatch : dispatches) {
    if (dispatch.host_ctx == nullptr) {
      return reject();
    }
    auto kernel_iter = ti_kernels_.find(dispatch.handle.get_launch_id());
    if (kernel_iter == ti_kernels_.end()) {
      return reject();
    }
    auto *ti_kernel = kernel_iter->second.get();
    if (ti_kernel->get_ret_buffer_size() != 0 ||
        !ti_kernel->runtime_argpack_args().empty()) {
      return reject();
    }

    PreparedDispatch prepared_dispatch;
    prepared_dispatch.kernel = ti_kernel;
    prepared_dispatch.host_ctx = dispatch.host_ctx;
    prepared_dispatch.indirect_dispatch = dispatch.indirect_dispatch;
    const bool is_indirect =
        dispatch.indirect_dispatch != kDeviceNullPtr;
    if (is_indirect) {
      if (structured_control != nullptr || nested_control != nullptr ||
          dispatch.indirect_dispatch.device != device_ ||
          dispatch.indirect_dispatch.alloc_id == 0 ||
          dispatch.indirect_dispatch.offset %
                  alignof(std::uint32_t) !=
              0) {
        return reject();
      }
      has_indirect_dispatch = true;
      DeviceAllocation packet_allocation{
          dispatch.indirect_dispatch.device,
          dispatch.indirect_dispatch.alloc_id};
      ndarrays_in_use_.insert(packet_allocation.alloc_id);
      key.push_back(0xD000u);
      push_graph_allocation_key(
          key, packet_allocation,
          dispatch.indirect_dispatch.offset,
          3 * sizeof(std::uint32_t));
      // vkCmdDispatchIndirect bakes the buffer handle and offset into the
      // recorded command, so allocation changes require re-recording rather
      // than descriptor-only structural patching.
      structure_key.push_back(0xD000u);
      push_graph_allocation_key(
          structure_key, packet_allocation,
          dispatch.indirect_dispatch.offset,
          3 * sizeof(std::uint32_t));
    } else {
      key.push_back(0xD001u);
      structure_key.push_back(0xD001u);
    }

    key.push_back(dispatch.handle.get_launch_id());
    key.push_back(ti_kernel->get_args_buffer_size());
    key.push_back(ti_kernel->num_pipelines());
    structure_key.push_back(dispatch.handle.get_launch_id());
    structure_key.push_back(ti_kernel->get_args_buffer_size());
    structure_key.push_back(ti_kernel->num_pipelines());

    for (const auto &array_arg : ti_kernel->runtime_array_args()) {
      const auto alloc_type =
          dispatch.host_ctx->device_allocation_type[array_arg.indices];
      const auto runtime_size =
          dispatch.host_ctx->array_runtime_sizes.find(array_arg.indices);
      structure_key.push_back(
          runtime_size == dispatch.host_ctx->array_runtime_sizes.end()
              ? 0
              : runtime_size->second);
      if (alloc_type == LaunchContextBuilder::DevAllocType::kDenseStorage) {
        const auto &binding =
            dispatch.host_ctx->get_resolved_dense_storage(array_arg.indices);
        ndarrays_in_use_.insert(binding.allocation.alloc_id);
        key.push_back(0xA004u);
        structure_key.push_back(0xA004u);
        push_graph_allocation_key(key, binding.allocation,
                                  binding.byte_offset, binding.byte_size);
        push_runtime_structure(binding.allocation, binding.byte_offset,
                               binding.byte_size);
        key.push_back(binding.runtime_signature);
        key.push_back(0xA003u);
        structure_key.push_back(0xA003u);
      } else if (alloc_type == LaunchContextBuilder::DevAllocType::kNdarray) {
        auto data_it =
            dispatch.host_ctx->array_ptrs.find(array_arg.data_ptr_indices);
        if (data_it == dispatch.host_ctx->array_ptrs.end() ||
            data_it->second == nullptr) {
          return reject();
        }
        const int arg_offset =
            dispatch.host_ctx->args_type->get_element_offset(
                array_arg.indices);
        const auto resource_ref = std::find_if(
            dispatch.host_ctx->ndarray_ptrs.begin(),
            dispatch.host_ctx->ndarray_ptrs.end(),
            [arg_offset](const auto &ref) {
              return ref.arg_offset == arg_offset;
            });
        const bool has_registered_resource =
            resource_ref != dispatch.host_ctx->ndarray_ptrs.end();
        if (has_registered_resource && !resource_ref->data_handle) {
          return reject();
        }
        DeviceAllocation devalloc =
            *static_cast<DeviceAllocation *>(data_it->second);
        ndarrays_in_use_.insert(devalloc.alloc_id);
        key.push_back(0xA001u);
        structure_key.push_back(0xA001u);
        push_graph_allocation_key(key, devalloc);
        push_runtime_structure(devalloc, 0, 0);
        // DeviceAllocationId may be reused after an Ndarray is retired.
        // Keep the generation-qualified runtime-resource identity in the
        // exact replay key so an ABA allocation cannot replay descriptors
        // recorded for the previous high-level resource. The structural key
        // deliberately omits this identity, allowing descriptor patching.
        if (has_registered_resource) {
          key.insert(
              key.end(),
              {0xA005u, resource_ref->data_handle.domain,
               static_cast<std::uint64_t>(resource_ref->data_handle.kind),
               static_cast<std::uint64_t>(resource_ref->data_handle.index),
               static_cast<std::uint64_t>(
                   resource_ref->data_handle.generation)});
        }

        auto grad_it =
            dispatch.host_ctx->array_ptrs.find(array_arg.grad_ptr_indices);
        if (grad_it != dispatch.host_ctx->array_ptrs.end() &&
            grad_it->second != nullptr) {
          if (has_registered_resource && !resource_ref->grad_handle) {
            return reject();
          }
          DeviceAllocation grad_alloc =
              *static_cast<DeviceAllocation *>(grad_it->second);
          ndarrays_in_use_.insert(grad_alloc.alloc_id);
          key.push_back(0xA002u);
          structure_key.push_back(0xA002u);
          push_graph_allocation_key(key, grad_alloc);
          push_runtime_structure(grad_alloc, 0, 0);
          if (has_registered_resource) {
            key.insert(
                key.end(),
                {0xA006u, resource_ref->grad_handle.domain,
                 static_cast<std::uint64_t>(resource_ref->grad_handle.kind),
                 static_cast<std::uint64_t>(
                     resource_ref->grad_handle.index),
                 static_cast<std::uint64_t>(
                     resource_ref->grad_handle.generation)});
          }
        } else {
          key.push_back(0xA003u);
          structure_key.push_back(0xA003u);
        }
      } else {
        return reject();
      }
    }

    const auto &task_attribs = ti_kernel->ti_kernel_attribs().tasks_attribs;
    if (is_indirect && task_attribs.size() != 1) {
      return reject();
    }
    for (int task_index = 0; task_index < task_attribs.size(); ++task_index) {
      const auto &attribs = task_attribs[task_index];
      if (!attribs.texture_binds.empty() ||
          !attribs.acceleration_structure_binds.empty() ||
          attribs.task_type == OffloadedTaskType::listgen ||
          attribs.may_mutate_sparse_topology ||
          ti_kernel->task_uses_listgen_buffer(task_index)) {
        return reject();
      }
      key.push_back(0xB000u);
      key.push_back(static_cast<uint64_t>(task_index));
      structure_key.push_back(0xB000u);
      structure_key.push_back(static_cast<uint64_t>(task_index));
      for (const auto &bind : ti_kernel->buffer_binding_plan(task_index)) {
        if (bind.binding < 0) {
          continue;
        }
        switch (bind.kind) {
          case CompiledTaichiKernel::BufferBindingKind::Skip:
          case CompiledTaichiKernel::BufferBindingKind::Args:
          case CompiledTaichiKernel::BufferBindingKind::ExtArrRw:
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticRw:
            key.push_back(0xB101u);
            key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(key, bind.static_alloc
                                               ? *bind.static_alloc
                                               : kDeviceNullAllocation);
            structure_key.push_back(0xB101u);
            structure_key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(
                structure_key, bind.static_alloc ? *bind.static_alloc
                                                 : kDeviceNullAllocation);
            break;
          case CompiledTaichiKernel::BufferBindingKind::StaticLookupRw: {
            if (bind.buffer.type == BufferType::ListGen) {
              return reject();
            }
            DeviceAllocation *alloc = ti_kernel->get_buffer_bind(bind.buffer);
            key.push_back(0xB102u);
            key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(key,
                                      alloc ? *alloc : kDeviceNullAllocation);
            structure_key.push_back(0xB102u);
            structure_key.push_back(static_cast<uint64_t>(bind.binding));
            push_graph_allocation_key(
                structure_key, alloc ? *alloc : kDeviceNullAllocation);
            break;
          }
          case CompiledTaichiKernel::BufferBindingKind::ArgPack:
          case CompiledTaichiKernel::BufferBindingKind::RetsRw:
          case CompiledTaichiKernel::BufferBindingKind::ChunkedRwArray:
            return reject();
        }
      }
    }
    total_tasks += task_attribs.size();
    dispatch_task_offsets.push_back(total_tasks);
    prepared.push_back(std::move(prepared_dispatch));
  }
  std::size_t structured_initial_tasks = 0;
  std::vector<uint32_t> structured_group_counts;
  struct NestedTaskRange {
    std::size_t condition_begin{0};
    std::size_t body_begin{0};
    std::size_t end{0};
    std::vector<std::uint32_t> group_counts;
    std::size_t packet_word_offset{0};
    std::size_t observation_word_offset{0};
  };
  std::size_t nested_outer_condition_task_end = 0;
  std::vector<NestedTaskRange> nested_task_ranges;
  if (structured_control != nullptr) {
    for (std::size_t i = 0;
         i < structured_control->initial_dispatch_count; ++i) {
      structured_initial_tasks +=
          prepared[i].kernel->ti_kernel_attribs().tasks_attribs.size();
    }
    for (std::size_t i = structured_control->initial_dispatch_count;
         i < prepared.size(); ++i) {
      for (const auto &attribs :
           prepared[i].kernel->ti_kernel_attribs().tasks_attribs) {
        if (attribs.advisory_num_threads_per_group <= 0) {
          return reject();
        }
        const int group_x =
            (attribs.advisory_total_num_threads +
             attribs.advisory_num_threads_per_group - 1) /
            attribs.advisory_num_threads_per_group;
        if (group_x < 0) {
          return reject();
        }
        structured_group_counts.push_back(
            static_cast<std::uint32_t>(group_x));
      }
    }
    const std::size_t encoded_actions =
        static_cast<std::size_t>(structured_control->max_iterations) *
        (structured_group_counts.size() + 1);
    if (structured_initial_tasks == 0 ||
        structured_group_counts.empty() ||
        structured_group_counts.size() > kStructuredMaximumPackets) {
      return reject();
    }
    const std::size_t chained_control_bytes =
        (static_cast<std::size_t>(structured_control->max_iterations) *
             (structured_group_counts.size() + 1) *
             kStructuredPacketWords +
         kStructuredObservationWords + kStructuredChainedSinkWords) *
        sizeof(std::uint32_t);
    const bool chained_eligible =
        structured_control->max_iterations > 0 &&
        encoded_actions <= kStructuredChainedMaximumDispatches &&
        chained_control_bytes <= kStructuredChainedMaximumControlBytes;
    const bool conditional_eligible =
        structured_control->max_iterations > 0 &&
        encoded_actions <= kStructuredMaximumEncodedActions &&
        device_->supports_conditional_commands();
    const bool coarse_conditional_eligible =
        structured_control->max_iterations > 0 &&
        encoded_actions <= kStructuredMaximumEncodedActions &&
        device_->supports_conditional_commands();
    structured_strategy = structured_control->strategy;
    if (structured_strategy == GraphStructuredStrategy::automatic) {
      // Chained dispatch remains an explicit qualification path until a
      // device-specific performance table proves that eliminating controller
      // invocations offsets its extra indirect commands and control arena.
      structured_strategy = GraphStructuredStrategy::compact;
    }
    if ((structured_strategy == GraphStructuredStrategy::chained &&
         !chained_eligible) ||
        (structured_strategy == GraphStructuredStrategy::conditional &&
         !conditional_eligible) ||
        (structured_strategy ==
             GraphStructuredStrategy::coarse_conditional &&
         !coarse_conditional_eligible) ||
        (structured_strategy == GraphStructuredStrategy::compact &&
         encoded_actions > kStructuredMaximumEncodedActions)) {
      return reject();
    }
    key.insert(key.end(),
               {0xC001u,
                static_cast<std::uint32_t>(structured_strategy),
                static_cast<std::uint32_t>(
                    structured_control->execute_initial_dispatches)});
    structure_key.insert(
        structure_key.end(),
        {0xC001u, static_cast<std::uint32_t>(structured_strategy),
         static_cast<std::uint32_t>(
             structured_control->execute_initial_dispatches)});
  } else if (nested_control != nullptr) {
    nested_outer_condition_task_end =
        dispatch_task_offsets[nested_control
                                  ->outer_condition_dispatch_count];
    if (nested_outer_condition_task_end == 0 ||
        nested_outer_condition_task_end >= total_tasks) {
      return reject();
    }
    nested_task_ranges.reserve(nested_control->inner_controls.size());
    std::size_t task_cursor = nested_outer_condition_task_end;
    std::uint64_t repeated_payload_tasks = 0;
    std::uint64_t single_copy_payload_tasks = 0;
    std::uint64_t inner_controller_actions = 0;
    for (const auto &inner : nested_control->inner_controls) {
      NestedTaskRange range;
      range.condition_begin =
          dispatch_task_offsets[inner.inner_condition_dispatch_begin];
      range.body_begin =
          dispatch_task_offsets[inner.inner_body_dispatch_begin];
      range.end = dispatch_task_offsets[inner.inner_dispatch_end];
      if (task_cursor > range.condition_begin ||
          range.condition_begin >= range.body_begin ||
          range.body_begin >= range.end || range.end > total_tasks) {
        return reject();
      }
      for (std::size_t i = range.body_begin; i < range.end; ++i) {
        const auto dispatch_index = static_cast<std::size_t>(
            std::upper_bound(dispatch_task_offsets.begin(),
                             dispatch_task_offsets.end(), i) -
            dispatch_task_offsets.begin() - 1);
        const auto task_index =
            static_cast<int>(i - dispatch_task_offsets[dispatch_index]);
        const auto &attribs =
            prepared[dispatch_index]
                .kernel->ti_kernel_attribs()
                .tasks_attribs[task_index];
        if (attribs.advisory_num_threads_per_group <= 0) {
          return reject();
        }
        const int group_x =
            (attribs.advisory_total_num_threads +
             attribs.advisory_num_threads_per_group - 1) /
            attribs.advisory_num_threads_per_group;
        if (group_x < 0) {
          return reject();
        }
        range.group_counts.push_back(static_cast<std::uint32_t>(group_x));
      }
      if (range.group_counts.empty() ||
          range.group_counts.size() > kStructuredMaximumPackets) {
        return reject();
      }
      const std::uint64_t payload_tasks = range.end - range.body_begin;
      single_copy_payload_tasks += payload_tasks;
      repeated_payload_tasks +=
          static_cast<std::uint64_t>(inner.inner_max_iterations) *
          payload_tasks;
      inner_controller_actions +=
          2 + static_cast<std::uint64_t>(inner.inner_max_iterations);
      task_cursor = range.end;
      nested_task_ranges.push_back(std::move(range));
    }
    const std::uint64_t static_tasks =
        total_tasks - nested_outer_condition_task_end -
        single_copy_payload_tasks;
    const std::uint64_t per_outer_actions =
        1 + static_tasks + repeated_payload_tasks +
        inner_controller_actions;
    const std::uint64_t encoded_actions =
        nested_outer_condition_task_end + 2 +
        static_cast<std::uint64_t>(
            nested_control->outer_max_iterations) *
            per_outer_actions;
    if (encoded_actions > kStructuredMaximumEncodedActions) {
      return reject();
    }
    structured_strategy = GraphStructuredStrategy::compact;
    structured_group_counts = nested_task_ranges.front().group_counts;
    key.insert(key.end(),
               {0xC101u, static_cast<std::uint64_t>(encoded_actions)});
    structure_key.insert(
        structure_key.end(),
        {0xC101u, static_cast<std::uint64_t>(encoded_actions)});
  }
  const bool composed_single_task =
      total_tasks == 1 && dispatches.size() == 1 &&
      dispatches.front().source_dispatch_count > 1;
  if (total_tasks == 0 ||
      (total_tasks == 1 && !has_indirect_dispatch &&
       !composed_single_task)) {
    return reject(GraphReplayFallbackReason::insufficient_tasks);
  }

  executable.refresh_prepared_cache(key, prepared);

  // Ordinary replay may fall back to individually launched tasks when the
  // fixed slot ring is busy. Indirect dispatch cannot: its device-written
  // packet is consumed by the recorded command, so applying bounded
  // backpressure is the only semantics-preserving fixed-memory path.
  GfxRuntime::GraphReplayExecutable::Slot *slot =
      executable.acquire_ready_slot(
          key, structure_key,
          has_indirect_dispatch ||
          (structured_control != nullptr && structured_result == nullptr) ||
              (nested_control != nullptr && nested_result == nullptr));
  if (slot == nullptr) {
    if (state.diagnostics_enabled) {
      ++state.slot_saturation_fallbacks;
    }
    if (statistics != nullptr) {
      statistics->record_graph_slot_saturation_fallback();
    }
    return reject(GraphReplayFallbackReason::slot_saturated);
  }

  const bool is_recapture = slot->recorded || slot->cmdlist != nullptr;
  slot->args_buffers.resize(prepared.size());
  slot->args_buffer_sizes.resize(prepared.size(), 0);
  SizeMap empty_ext_sizes;
  ArgPackMap empty_argpacks;
  for (size_t i = 0; i < prepared.size(); ++i) {
    auto &pd = prepared[i];
    const size_t args_size = pd.kernel->get_args_buffer_size();
    if (args_size > 0 &&
        (!slot->args_buffers[i] || slot->args_buffer_sizes[i] != args_size)) {
      auto [buf, res] = device_->allocate_memory_unique(
          {args_size,
           /*host_write=*/true, /*host_read=*/false,
           /*export_sharing=*/false, AllocUsage::Uniform});
      TI_ASSERT_INFO(res == RhiResult::success,
                     "Failed to allocate Vulkan graph args buffer");
      slot->args_buffers[i] = std::move(buf);
      slot->args_buffer_sizes[i] = args_size;
      slot->recorded = false;
      slot->cmdlist = nullptr;
    }
    pd.args_buffer = slot->args_buffers[i].get();
    auto ctx_blitter = HostDeviceContextBlitter::maybe_make(
        &pd.kernel->ti_kernel_attribs().ctx_attribs,
        &pd.kernel->runtime_array_args(), *pd.host_ctx, device_,
        pd.args_buffer, nullptr);
    if (ctx_blitter) {
      ctx_blitter->host_to_device(*pd.any_arrays, empty_ext_sizes,
                                  empty_argpacks);
    }
  }

  flush_if_pending();

  if (slot->resource_sets.size() < total_tasks) {
    const size_t old_size = slot->resource_sets.size();
    slot->resource_sets.resize(total_tasks);
    for (size_t i = old_size; i < total_tasks; ++i) {
      slot->resource_sets[i] = device_->create_resource_set_unique();
    }
  }

  auto update_task_bindings = [&](PreparedDispatch &pd, int task_index,
                                  ShaderResourceSet *bindings,
                                  bool patch_existing) {
    return bind_graph_task(pd, task_index, bindings, patch_existing);
  };

  auto patch_all_task_bindings = [&]() {
    size_t resource_set_index = 0;
    for (auto &pd : prepared) {
      const auto &task_attribs = pd.kernel->ti_kernel_attribs().tasks_attribs;
      for (int task_index = 0; task_index < task_attribs.size();
           ++task_index) {
        if (!update_task_bindings(
                pd, task_index,
                slot->resource_sets[resource_set_index++].get(),
                /*patch_existing=*/true)) {
          return false;
        }
      }
    }
    return true;
  };

  const bool nested_structured = nested_control != nullptr;
  std::size_t structured_observation_word_offset = 0;
  std::size_t nested_outer_gate_word_offset = 0;
  std::size_t nested_header_word_offset = 0;
  std::size_t nested_trace_word_offset = 0;
  std::size_t nested_trace_row_words = 0;
  std::size_t structured_control_bytes = 0;
  if (!structured_group_counts.empty()) {
    if (nested_structured) {
      std::size_t child_control_words = 0;
      for (auto &range : nested_task_ranges) {
        range.packet_word_offset = child_control_words;
        range.observation_word_offset =
            child_control_words +
            range.group_counts.size() * kStructuredPacketWords;
        child_control_words =
            range.observation_word_offset + kStructuredObservationWords;
      }
      nested_outer_gate_word_offset = child_control_words;
      nested_header_word_offset = nested_outer_gate_word_offset + 1;
      nested_trace_word_offset =
          nested_header_word_offset + kNestedStructuredHeaderWords;
      nested_trace_row_words =
          nested_task_ranges.size() * kNestedStructuredTraceWords;
      structured_control_bytes =
          (nested_trace_word_offset +
           static_cast<std::size_t>(
               nested_control->outer_max_iterations) *
               nested_trace_row_words) *
          sizeof(std::uint32_t);
    } else {
      structured_observation_word_offset =
          structured_strategy == GraphStructuredStrategy::conditional
              ? structured_control->max_iterations
              : structured_strategy == GraphStructuredStrategy::chained
                    ? static_cast<std::size_t>(
                          structured_control->max_iterations) *
                          (structured_group_counts.size() + 1) *
                          kStructuredPacketWords
                    : structured_group_counts.size() *
                          kStructuredPacketWords;
      structured_control_bytes =
          (structured_observation_word_offset +
           kStructuredObservationWords +
           (structured_strategy == GraphStructuredStrategy::chained
                ? kStructuredChainedSinkWords
                : structured_strategy ==
                          GraphStructuredStrategy::conditional
                      ? 1
                      : structured_strategy ==
                                GraphStructuredStrategy::coarse_conditional
                            ? 1
                            : 0)) *
          sizeof(std::uint32_t);
    }
  }
  if (structured_control_bytes >
      kStructuredChainedMaximumControlBytes) {
    return reject();
  }
  // Nested replay uses the same device-side snapshot buffer for its gate
  // controller even when the caller does not request an immediate host
  // observation. Keeping that buffer internal lets asynchronous submission
  // return after queueing while preserving exactly the same encoded control
  // semantics as the synchronous diagnostic path.
  const bool structured_terminal_observation =
      (structured_control != nullptr && structured_result != nullptr) ||
      nested_control != nullptr;
  const std::size_t structured_observation_bytes =
      !structured_terminal_observation
          ? 0
          : nested_structured
                ? (kNestedStructuredHeaderWords +
                   static_cast<std::size_t>(
                       nested_control->outer_max_iterations) *
                       nested_trace_row_words) *
                      sizeof(std::uint32_t)
                : kStructuredObservationWords * sizeof(std::uint32_t);
  if (structured_control != nullptr || nested_control != nullptr) {
    const bool inner_has_status =
        !nested_structured && structured_control->has_status;
    const bool outer_has_status =
        nested_structured && nested_control->outer_has_status;
    std::vector<std::vector<std::uint32_t>> nested_group_counts;
    std::vector<bool> nested_has_status;
    if (nested_structured) {
      nested_group_counts.reserve(nested_task_ranges.size());
      nested_has_status.reserve(nested_task_ranges.size());
      for (std::size_t child = 0; child < nested_task_ranges.size(); ++child) {
        nested_group_counts.push_back(
            nested_task_ranges[child].group_counts);
        nested_has_status.push_back(
            nested_control->inner_controls[child].inner_has_status);
      }
    }
    const bool rebuild_control =
        !slot->structured_control_buffer ||
        slot->structured_control_bytes != structured_control_bytes ||
        slot->structured_observation_bytes !=
            structured_observation_bytes ||
        slot->structured_group_counts != structured_group_counts ||
        slot->nested_group_counts != nested_group_counts ||
        slot->nested_has_status != nested_has_status ||
        slot->structured_has_status != inner_has_status ||
        slot->structured_nested != nested_structured ||
        slot->structured_outer_has_status != outer_has_status ||
        slot->structured_strategy != structured_strategy;
    if (rebuild_control) {
      auto [control_buffer, control_result] =
          device_->allocate_memory_unique(
              {structured_control_bytes,
               /*host_write=*/false, /*host_read=*/false,
               /*export_sharing=*/false,
               nested_structured
                   ? AllocUsage::Storage | AllocUsage::Indirect |
                         AllocUsage::Conditional
               : structured_strategy == GraphStructuredStrategy::conditional
                   ? AllocUsage::Storage | AllocUsage::Conditional
                   : structured_strategy ==
                             GraphStructuredStrategy::coarse_conditional
                         ? AllocUsage::Storage | AllocUsage::Indirect |
                               AllocUsage::Conditional
                         : AllocUsage::Storage | AllocUsage::Indirect});
      if (control_result != RhiResult::success || !control_buffer) {
        return reject();
      }
      std::unique_ptr<DeviceAllocationGuard> observation_buffer;
      if (structured_terminal_observation) {
        auto [allocated, observation_result] =
            device_->allocate_memory_unique(
                {structured_observation_bytes,
                 /*host_write=*/false, /*host_read=*/true,
                 /*export_sharing=*/false, AllocUsage::None});
        if (observation_result != RhiResult::success || !allocated) {
          return reject();
        }
        observation_buffer = std::move(allocated);
      }
      slot->structured_control_buffer = std::move(control_buffer);
      slot->structured_observation_buffer =
          std::move(observation_buffer);
      slot->structured_control_bytes = structured_control_bytes;
      slot->structured_observation_bytes =
          structured_observation_bytes;
      slot->structured_group_counts = structured_group_counts;
      slot->nested_group_counts = nested_group_counts;
      slot->nested_has_status = nested_has_status;
      slot->structured_has_status = inner_has_status;
      slot->structured_nested = nested_structured;
      slot->structured_outer_has_status = outer_has_status;
      slot->structured_strategy = structured_strategy;
      if (nested_structured) {
        slot->structured_controller_pipeline.reset();
        slot->structured_controller_resources.reset();
        slot->nested_inner_controller_pipelines.clear();
        slot->nested_inner_controller_resources.clear();
        for (std::size_t child = 0; child < nested_task_ranges.size(); ++child) {
          slot->nested_inner_controller_pipelines.push_back(
              create_structured_pipeline(
                  device_,
                  make_compact_controller_spirv(
                      device_->get_caps(),
                      nested_task_ranges[child].group_counts,
                      nested_control->inner_controls[child].inner_has_status,
                      nested_task_ranges[child].packet_word_offset),
                  "vulkan_nested_compact_controller"));
          slot->nested_inner_controller_resources.push_back(
              device_->create_resource_set_unique());
        }
      } else {
        slot->nested_inner_controller_pipelines.clear();
        slot->nested_inner_controller_resources.clear();
        slot->structured_controller_pipeline = create_structured_pipeline(
            device_,
            structured_strategy == GraphStructuredStrategy::conditional
                ? make_conditional_controller_spirv(
                      device_->get_caps(),
                      structured_control->max_iterations,
                      inner_has_status)
                : structured_strategy == GraphStructuredStrategy::chained
                      ? make_chained_controller_spirv(
                            device_->get_caps(), structured_group_counts,
                            structured_control->max_iterations,
                            inner_has_status)
                      : make_compact_controller_spirv(
                            device_->get_caps(), structured_group_counts,
                            inner_has_status),
            structured_strategy == GraphStructuredStrategy::conditional
                ? "vulkan_conditional_controller"
                : structured_strategy == GraphStructuredStrategy::chained
                      ? "vulkan_chained_controller"
                      : "vulkan_compact_controller");
        slot->structured_controller_resources =
            device_->create_resource_set_unique();
      }
      if (nested_structured ||
          structured_strategy ==
              GraphStructuredStrategy::coarse_conditional) {
        const std::uint32_t stable_predicate_index =
            nested_structured
                ? static_cast<std::uint32_t>(
                      nested_outer_gate_word_offset)
                : static_cast<std::uint32_t>(
                      structured_observation_word_offset +
                      kStructuredObservationWords);
        slot->structured_gate_pipeline = create_structured_pipeline(
            device_,
            make_coarse_conditional_gate_spirv(
                device_->get_caps(), stable_predicate_index),
            nested_structured ? "vulkan_nested_outer_gate"
                              : "vulkan_coarse_conditional_gate");
        slot->structured_gate_resources =
            device_->create_resource_set_unique();
      } else {
        slot->structured_gate_pipeline.reset();
        slot->structured_gate_resources.reset();
      }
      if (!nested_structured && structured_terminal_observation) {
        slot->structured_terminal_pipeline = create_structured_pipeline(
            device_,
            make_structured_terminal_spirv(
                device_->get_caps(), structured_observation_word_offset,
                inner_has_status),
            "vulkan_structured_terminal");
        slot->structured_terminal_resources =
            device_->create_resource_set_unique();
      } else {
        slot->structured_terminal_pipeline.reset();
        slot->structured_terminal_resources.reset();
      }
      if (nested_structured) {
        slot->nested_outer_initial_pipeline = create_structured_pipeline(
            device_,
            make_nested_outer_snapshot_spirv(
                device_->get_caps(), nested_header_word_offset,
                outer_has_status, /*terminal=*/false),
            "vulkan_nested_outer_initial");
        slot->nested_inner_initial_pipelines.clear();
        slot->nested_inner_terminal_pipelines.clear();
        slot->nested_inner_initial_resources.clear();
        slot->nested_inner_terminal_resources.clear();
        for (std::size_t child = 0; child < nested_task_ranges.size(); ++child) {
          const bool child_has_status =
              nested_control->inner_controls[child].inner_has_status;
          const std::size_t child_trace_offset =
              nested_trace_word_offset +
              child * kNestedStructuredTraceWords;
          slot->nested_inner_initial_pipelines.push_back(
              create_structured_pipeline(
                  device_,
                  make_nested_inner_initial_spirv(
                      device_->get_caps(), nested_header_word_offset,
                      child_trace_offset, nested_trace_row_words,
                      child_has_status, child == 0),
                  "vulkan_nested_inner_initial"));
          slot->nested_inner_terminal_pipelines.push_back(
              create_structured_pipeline(
                  device_,
                  make_nested_inner_terminal_spirv(
                      device_->get_caps(),
                      nested_task_ranges[child].observation_word_offset,
                      nested_header_word_offset, child_trace_offset,
                      nested_trace_row_words, child_has_status),
                  "vulkan_nested_inner_terminal"));
          slot->nested_inner_initial_resources.push_back(
              device_->create_resource_set_unique());
          slot->nested_inner_terminal_resources.push_back(
              device_->create_resource_set_unique());
        }
        slot->nested_outer_terminal_pipeline =
            create_structured_pipeline(
                device_,
                make_nested_outer_snapshot_spirv(
                    device_->get_caps(), nested_header_word_offset,
                    outer_has_status, /*terminal=*/true),
                "vulkan_nested_outer_terminal");
        slot->nested_outer_initial_resources =
            device_->create_resource_set_unique();
        slot->nested_outer_terminal_resources =
            device_->create_resource_set_unique();
      } else {
        slot->nested_outer_initial_pipeline.reset();
        slot->nested_inner_initial_pipelines.clear();
        slot->nested_inner_terminal_pipelines.clear();
        slot->nested_outer_terminal_pipeline.reset();
        slot->nested_outer_initial_resources.reset();
        slot->nested_inner_initial_resources.clear();
        slot->nested_inner_terminal_resources.clear();
        slot->nested_outer_terminal_resources.reset();
      }
      slot->recorded = false;
      slot->cmdlist.reset();
    }
  }

  auto update_structured_bindings = [&](bool patch_existing) {
    if (structured_control == nullptr && nested_control == nullptr) {
      return true;
    }
    if (nested_control != nullptr) {
      auto *outer_gate = slot->structured_gate_resources.get();
      auto *outer_initial =
          slot->nested_outer_initial_resources.get();
      auto *outer_terminal =
          slot->nested_outer_terminal_resources.get();
      if (outer_gate == nullptr || outer_initial == nullptr ||
          outer_terminal == nullptr ||
          slot->nested_inner_controller_resources.size() !=
              nested_control->inner_controls.size() ||
          slot->nested_inner_initial_resources.size() !=
              nested_control->inner_controls.size() ||
          slot->nested_inner_terminal_resources.size() !=
              nested_control->inner_controls.size() ||
          !slot->structured_control_buffer) {
        return false;
      }
      const DevicePtr outer_status =
          nested_control->outer_has_status
              ? nested_control->outer_status
              : nested_control->outer_counter;
      outer_gate->rw_buffer(0, nested_control->outer_predicate,
                            sizeof(std::uint32_t));
      outer_gate->rw_buffer(1, *slot->structured_control_buffer);
      outer_initial->rw_buffer(0, nested_control->outer_predicate,
                               sizeof(std::uint32_t));
      outer_initial->rw_buffer(1, nested_control->outer_counter,
                               sizeof(std::uint32_t));
      outer_initial->rw_buffer(2, outer_status,
                               sizeof(std::uint32_t));
      outer_initial->rw_buffer(3,
                               *slot->structured_control_buffer);
      bool inner_ready = true;
      for (std::size_t child = 0;
           child < nested_control->inner_controls.size(); ++child) {
        const auto &inner = nested_control->inner_controls[child];
        auto *controller =
            slot->nested_inner_controller_resources[child].get();
        auto *inner_initial =
            slot->nested_inner_initial_resources[child].get();
        auto *inner_terminal =
            slot->nested_inner_terminal_resources[child].get();
        if (controller == nullptr || inner_initial == nullptr ||
            inner_terminal == nullptr) {
          return false;
        }
        const DevicePtr inner_status =
            inner.inner_has_status ? inner.inner_status
                                   : inner.inner_counter;
        controller->rw_buffer(0, inner.inner_predicate,
                              sizeof(std::uint32_t));
        controller->rw_buffer(1, *slot->structured_control_buffer);
        controller->rw_buffer(2, inner_status, sizeof(std::uint32_t));
        inner_initial->rw_buffer(0, inner.inner_counter,
                                 sizeof(std::uint32_t));
        inner_initial->rw_buffer(1, inner_status,
                                 sizeof(std::uint32_t));
        inner_initial->rw_buffer(2, *slot->structured_control_buffer);
        inner_terminal->rw_buffer(0, inner.inner_predicate,
                                  sizeof(std::uint32_t));
        inner_terminal->rw_buffer(1, inner.inner_counter,
                                  sizeof(std::uint32_t));
        inner_terminal->rw_buffer(2, inner_status,
                                  sizeof(std::uint32_t));
        inner_terminal->rw_buffer(3, *slot->structured_control_buffer);
        inner_ready =
            inner_ready &&
            controller->prepare_for_replay(patch_existing) ==
                RhiResult::success &&
            inner_initial->prepare_for_replay(patch_existing) ==
                RhiResult::success &&
            inner_terminal->prepare_for_replay(patch_existing) ==
                RhiResult::success;
      }
      outer_terminal->rw_buffer(0, nested_control->outer_predicate,
                                sizeof(std::uint32_t));
      outer_terminal->rw_buffer(1, nested_control->outer_counter,
                                sizeof(std::uint32_t));
      outer_terminal->rw_buffer(2, outer_status,
                                sizeof(std::uint32_t));
      outer_terminal->rw_buffer(3,
                                *slot->structured_control_buffer);
      return inner_ready &&
             outer_gate->prepare_for_replay(patch_existing) ==
                 RhiResult::success &&
             outer_initial->prepare_for_replay(patch_existing) ==
                 RhiResult::success &&
             outer_terminal->prepare_for_replay(patch_existing) ==
                 RhiResult::success;
    }
    auto *controller = slot->structured_controller_resources.get();
    auto *gate = slot->structured_gate_resources.get();
    auto *terminal = slot->structured_terminal_resources.get();
    if (controller == nullptr ||
        (structured_terminal_observation && terminal == nullptr) ||
        !slot->structured_control_buffer) {
      return false;
    }
    controller->rw_buffer(0, structured_control->predicate,
                          sizeof(std::uint32_t));
    controller->rw_buffer(1, *slot->structured_control_buffer);
    controller->rw_buffer(
        2,
        structured_control->has_status ? structured_control->status
                                       : structured_control->counter,
        sizeof(std::uint32_t));
    if (structured_terminal_observation) {
      terminal->rw_buffer(0, structured_control->predicate,
                          sizeof(std::uint32_t));
      terminal->rw_buffer(1, structured_control->counter,
                          sizeof(std::uint32_t));
      terminal->rw_buffer(
          2,
          structured_control->has_status ? structured_control->status
                                         : structured_control->counter,
          sizeof(std::uint32_t));
      terminal->rw_buffer(3, *slot->structured_control_buffer);
    }
    bool gate_ready = true;
    if (structured_strategy ==
        GraphStructuredStrategy::coarse_conditional) {
      gate_ready = gate != nullptr;
      if (gate_ready) {
        gate->rw_buffer(0, structured_control->predicate,
                        sizeof(std::uint32_t));
        gate->rw_buffer(1, *slot->structured_control_buffer);
        gate_ready = gate->prepare_for_replay(patch_existing) ==
                     RhiResult::success;
      }
    }
    return gate_ready &&
           controller->prepare_for_replay(patch_existing) ==
               RhiResult::success &&
           (!structured_terminal_observation ||
            terminal->prepare_for_replay(patch_existing) ==
                RhiResult::success);
  };

  auto finish_structured_submission = [&]() {
    if ((structured_control == nullptr || structured_result == nullptr) &&
        (nested_control == nullptr || nested_result == nullptr)) {
      return true;
    }
    // Reaching this point means submit() has already transferred ownership of
    // the command list to the queue. Any subsequent observation failure is
    // therefore not a capability miss: falling back would execute the same
    // side-effecting Graph a second time.
    if (structured_result != nullptr) {
      structured_result->submitted = true;
    }
    if (nested_result != nullptr) {
      nested_result->submitted = true;
    }
    const auto fail_committed_observation =
        [](const char *detail) -> void {
      throw TaichiRuntimeError(fmt::format(
          "Vulkan structured Graph submission committed, but terminal "
          "observation {}. Portable fallback is disabled after queue "
          "submission.",
          detail));
    };
    if (!slot->completion) {
      fail_committed_observation("did not receive a completion semaphore");
    }
    if (!slot->completion->wait()) {
      fail_committed_observation("completion wait failed");
    }
    // Internal deterministic fault injection for the no-double-execution
    // regression. It is intentionally checked only after completion, so the
    // device-side side effects are observable by the test.
    if (get_environ_config(
            "TI_INTERNAL_TEST_VULKAN_GRAPH_FAIL_POST_SUBMIT_OBSERVATION",
            0) != 0) {
      fail_committed_observation("was rejected by fault injection");
    }
    void *mapped = nullptr;
    const RhiResult map_result =
        device_->map(*slot->structured_observation_buffer, &mapped);
    if (map_result != RhiResult::success) {
      fail_committed_observation("mapping failed");
    }
    if (nested_control != nullptr && nested_result != nullptr) {
      std::vector<std::uint32_t> words(
          structured_observation_bytes / sizeof(std::uint32_t));
      std::memcpy(words.data(), mapped, structured_observation_bytes);
      device_->unmap(*slot->structured_observation_buffer);
      if (words.size() < kNestedStructuredHeaderWords) {
        fail_committed_observation("contained a truncated nested header");
      }
      nested_result->outer_logical_iterations =
          std::min(words[0], nested_control->outer_max_iterations);
      nested_result->outer_encoded_iterations =
          nested_control->outer_max_iterations;
      nested_result->outer_initial_predicate =
          static_cast<std::int32_t>(words[2]);
      nested_result->outer_initial_counter =
          static_cast<std::int32_t>(words[3]);
      nested_result->outer_initial_status =
          static_cast<std::int32_t>(words[4]);
      nested_result->outer_final_predicate =
          static_cast<std::int32_t>(words[5]);
      nested_result->outer_final_counter =
          static_cast<std::int32_t>(words[6]);
      nested_result->outer_final_status =
          static_cast<std::int32_t>(words[7]);
      nested_result->inner_region_count = static_cast<std::uint32_t>(
          nested_control->inner_controls.size());
      const std::uint32_t logical_outer =
          nested_result->outer_logical_iterations;
      const std::size_t inner_sample_count =
          static_cast<std::size_t>(logical_outer) *
          nested_control->inner_controls.size();
      nested_result->inner_logical_iterations.reserve(inner_sample_count);
      nested_result->inner_encoded_iterations.reserve(inner_sample_count);
      nested_result->inner_initial_counters.reserve(inner_sample_count);
      nested_result->inner_final_counters.reserve(inner_sample_count);
      nested_result->inner_final_predicates.reserve(inner_sample_count);
      nested_result->inner_initial_statuses.reserve(inner_sample_count);
      nested_result->inner_final_statuses.reserve(inner_sample_count);
      std::uint64_t active_payload_dispatches = 0;
      for (std::uint32_t i = 0; i < logical_outer; ++i) {
        for (std::size_t child = 0;
             child < nested_control->inner_controls.size(); ++child) {
          const std::size_t row =
              kNestedStructuredHeaderWords +
              static_cast<std::size_t>(i) * nested_trace_row_words +
              child * kNestedStructuredTraceWords;
          if (row + kNestedStructuredTraceWords > words.size()) {
            fail_committed_observation("contained a truncated nested trace");
          }
          const auto &inner = nested_control->inner_controls[child];
          const std::uint32_t inner_logical =
              std::min(words[row], inner.inner_max_iterations);
          nested_result->inner_logical_iterations.push_back(inner_logical);
          nested_result->inner_encoded_iterations.push_back(
              inner.inner_max_iterations);
          nested_result->inner_initial_counters.push_back(
              static_cast<std::int32_t>(words[row + 1]));
          nested_result->inner_final_counters.push_back(
              static_cast<std::int32_t>(words[row + 2]));
          nested_result->inner_final_predicates.push_back(
              static_cast<std::int32_t>(words[row + 3]));
          nested_result->inner_initial_statuses.push_back(
              static_cast<std::int32_t>(words[row + 4]));
          nested_result->inner_final_statuses.push_back(
              static_cast<std::int32_t>(words[row + 5]));
          active_payload_dispatches +=
              static_cast<std::uint64_t>(inner_logical) *
              nested_task_ranges[child].group_counts.size();
        }
      }
      std::uint64_t encoded_inner_dispatches = 0;
      std::uint64_t inner_controller_count = 0;
      for (std::size_t child = 0;
           child < nested_control->inner_controls.size(); ++child) {
        encoded_inner_dispatches +=
            static_cast<std::uint64_t>(
                nested_control->outer_max_iterations) *
            nested_control->inner_controls[child].inner_max_iterations *
            nested_task_ranges[child].group_counts.size();
        inner_controller_count +=
            nested_control->inner_controls[child].inner_max_iterations;
      }
      nested_result->controller_dispatches =
          static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(
                  nested_control->outer_max_iterations) *
              (inner_controller_count + 1));
      nested_result->controller_invocations =
          static_cast<std::uint32_t>(
              static_cast<std::uint64_t>(
                  nested_control->outer_max_iterations) +
              static_cast<std::uint64_t>(logical_outer) *
                  inner_controller_count);
      nested_result->indirect_dispatches =
          static_cast<std::uint32_t>(encoded_inner_dispatches);
      nested_result->zero_dispatches =
          static_cast<std::uint32_t>(
              encoded_inner_dispatches -
              active_payload_dispatches);
      nested_result->control_bytes =
          static_cast<std::uint32_t>(structured_control_bytes);
      nested_result->observation_bytes =
          static_cast<std::uint32_t>(
              structured_observation_bytes);
      return true;
    }
    std::array<std::uint32_t, kStructuredObservationWords> words{};
    std::memcpy(words.data(), mapped, sizeof(words));
    device_->unmap(*slot->structured_observation_buffer);
    structured_result->strategy = structured_strategy;
    structured_result->logical_iterations = words[0];
    structured_result->predicate =
        static_cast<std::int32_t>(words[1]);
    structured_result->counter =
        static_cast<std::int32_t>(words[2]);
    structured_result->status = static_cast<std::int32_t>(words[3]);
    structured_result->initial_status =
        static_cast<std::int32_t>(words[4]);
    structured_result->encoded_iterations =
        structured_control->max_iterations;
    structured_result->controller_dispatches =
        structured_control->max_iterations;
    const std::uint32_t payload_dispatches =
        structured_control->max_iterations *
        static_cast<std::uint32_t>(structured_group_counts.size());
    const std::uint32_t bounded_logical =
        std::min(words[0], structured_control->max_iterations);
    const std::uint32_t active_payload_dispatches =
        bounded_logical *
        static_cast<std::uint32_t>(structured_group_counts.size());
    if (structured_strategy == GraphStructuredStrategy::conditional) {
      structured_result->controller_invocations =
          structured_control->max_iterations;
      structured_result->indirect_dispatches = 0;
      structured_result->zero_dispatches = 0;
    } else if (structured_strategy ==
               GraphStructuredStrategy::coarse_conditional) {
      structured_result->controller_invocations =
          bounded_logical == 0 ? 0 : structured_control->max_iterations;
      structured_result->indirect_dispatches = payload_dispatches;
      structured_result->zero_dispatches =
          payload_dispatches - active_payload_dispatches;
    } else if (structured_strategy == GraphStructuredStrategy::chained) {
      structured_result->controller_invocations = std::min(
          structured_control->max_iterations,
          bounded_logical +
              (bounded_logical < structured_control->max_iterations ? 1u
                                                                     : 0u));
      structured_result->indirect_dispatches =
          payload_dispatches + structured_control->max_iterations;
      structured_result->zero_dispatches =
          structured_result->indirect_dispatches -
          active_payload_dispatches -
          structured_result->controller_invocations;
    } else {
      structured_result->controller_invocations =
          structured_control->max_iterations;
      structured_result->indirect_dispatches = payload_dispatches;
      structured_result->zero_dispatches =
          payload_dispatches - active_payload_dispatches;
    }
    structured_result->control_bytes =
        static_cast<std::uint32_t>(structured_control_bytes);
    structured_result->observation_bytes =
        static_cast<std::uint32_t>(structured_observation_bytes);
    return true;
  };

  if (slot->recorded && slot->cmdlist && slot->key == key) {
    slot->completion =
        device_->get_compute_stream()->submit(slot->cmdlist.get());
    if (state.diagnostics_enabled) {
      ++state.replayed;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    state.last_path = GraphReplayLastPath::replay;
    return finish_structured_submission();
  }

  const bool structural_patch_enabled =
      get_environ_config("TI_VULKAN_GRAPH_STRUCTURAL_PATCH", 1) != 0;
  if (structural_patch_enabled && slot->recorded && slot->cmdlist &&
      slot->structure_key == structure_key && patch_all_task_bindings() &&
      update_structured_bindings(/*patch_existing=*/true)) {
    slot->key = key;
    slot->completion =
        device_->get_compute_stream()->submit(slot->cmdlist.get());
    if (state.diagnostics_enabled) {
      ++state.replayed;
      ++state.patched;
    }
    if (statistics != nullptr) {
      statistics->record_graph_replay();
    }
    state.last_path = GraphReplayLastPath::patched_replay;
    return finish_structured_submission();
  }

  auto [recorded_cmdlist, cmd_res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT_INFO(cmd_res == RhiResult::success,
                 "Failed to allocate Vulkan graph command list");
  auto *cmdlist = recorded_cmdlist.get();

  if (structured_control != nullptr || nested_control != nullptr) {
    struct StructuredBufferAccess {
      DeviceAllocation allocation{kDeviceNullAllocation};
      bool read{false};
      bool write{false};
    };
    using StructuredBufferAccessMap =
        std::unordered_map<DeviceAllocationId, StructuredBufferAccess>;
    struct StructuredTask {
      PreparedDispatch *dispatch{nullptr};
      int task_index{0};
      ShaderResourceSet *resources{nullptr};
      std::uint32_t group_x{0};
      StructuredBufferAccessMap accesses;
      bool global_read{false};
      bool global_write{false};
    };
    std::vector<StructuredTask> structured_tasks;
    structured_tasks.reserve(total_tasks);
    std::size_t resource_set_index = 0;
    for (auto &pd : prepared) {
      const auto &task_attribs =
          pd.kernel->ti_kernel_attribs().tasks_attribs;
      for (int task_index = 0; task_index < task_attribs.size();
           ++task_index) {
        auto *resources =
            slot->resource_sets[resource_set_index++].get();
        TI_ERROR_IF(
            !update_task_bindings(pd, task_index, resources,
                                  /*patch_existing=*/false),
            "Vulkan structured Graph descriptor preparation failed");
        const auto &attribs = task_attribs[task_index];
        const int group_x =
            (attribs.advisory_total_num_threads +
             attribs.advisory_num_threads_per_group - 1) /
            attribs.advisory_num_threads_per_group;
        StructuredTask task;
        task.dispatch = &pd;
        task.task_index = task_index;
        task.resources = resources;
        task.group_x = static_cast<std::uint32_t>(group_x);
        for (const auto &bind : attribs.buffer_binds) {
          const bool may_read =
              (bind.access &
               TaskAttributes::BufferBind::kAccessRead) != 0;
          const bool may_write = bind.may_write();
          if (!may_read && !may_write) {
            continue;
          }
          if (bind.buffer.type == BufferType::Args ||
              bind.buffer.type == BufferType::ArgPack) {
            continue;
          }
          if (state.diagnostics_enabled) {
            state.effect_reads += may_read ? 1 : 0;
            state.effect_writes += may_write ? 1 : 0;
          }
          if (bind.buffer.type == BufferType::Rets ||
              bind.buffer.type == BufferType::NodeAllocatorPool) {
            task.global_read = task.global_read || may_read;
            task.global_write = task.global_write || may_write;
            continue;
          }
          DeviceAllocation allocation = kDeviceNullAllocation;
          if (bind.buffer.type == BufferType::ExtArr) {
            allocation = pd.any_arrays->at(bind.buffer.root_id);
          } else {
            DeviceAllocation *resolved =
                pd.kernel->get_buffer_bind(bind.buffer);
            if (resolved != nullptr) {
              allocation = *resolved;
            }
          }
          if (allocation == kDeviceNullAllocation) {
            task.global_read = task.global_read || may_read;
            task.global_write = task.global_write || may_write;
            continue;
          }
          auto [it, inserted] = task.accesses.try_emplace(
              allocation.alloc_id,
              StructuredBufferAccess{allocation, may_read, may_write});
          if (!inserted) {
            it->second.read = it->second.read || may_read;
            it->second.write = it->second.write || may_write;
          }
        }
        structured_tasks.push_back(std::move(task));
      }
    }
    TI_ERROR_IF(
        !update_structured_bindings(/*patch_existing=*/false),
        "Vulkan structured control descriptor preparation failed");

    const bool structured_hazard_planner_enabled =
        get_environ_config("TI_VULKAN_STRUCTURED_HAZARD_PLANNER", 1) != 0;
    StructuredBufferAccessMap pending_accesses;
    bool pending_global_read = false;
    bool pending_global_write = false;
    auto count_structured_barrier = [&](bool exit_boundary) {
      if (!state.diagnostics_enabled) {
        return;
      }
      if (exit_boundary) {
        ++state.exit_barriers;
      } else {
        ++state.dependency_barriers;
      }
    };
    auto flush_structured_pending = [&](bool exit_boundary) {
      if (pending_global_read || pending_global_write ||
          !pending_accesses.empty()) {
        cmdlist->memory_barrier();
        count_structured_barrier(exit_boundary);
      }
      pending_global_read = false;
      pending_global_write = false;
      pending_accesses.clear();
    };
    auto prepare_structured_task = [&](const StructuredTask &task) {
      if (!structured_hazard_planner_enabled) {
        return;
      }
      bool hazard = false;
      const bool current_has_access =
          task.global_read || task.global_write || !task.accesses.empty();
      if ((pending_global_read || pending_global_write) &&
          current_has_access) {
        const bool only_reads =
            pending_global_read && !pending_global_write &&
            !task.global_write &&
            std::all_of(task.accesses.begin(), task.accesses.end(),
                        [](const auto &item) {
                          return item.second.read &&
                                 !item.second.write;
                        });
        hazard = !only_reads;
        if (only_reads && state.diagnostics_enabled) {
          ++state.rar_elisions;
        }
      }
      if ((task.global_read || task.global_write) &&
          !pending_accesses.empty()) {
        const bool pending_only_reads =
            std::all_of(pending_accesses.begin(),
                        pending_accesses.end(), [](const auto &item) {
                          return item.second.read &&
                                 !item.second.write;
                        });
        const bool only_reads =
            task.global_read && !task.global_write &&
            pending_only_reads;
        hazard = hazard || !only_reads;
        if (only_reads && state.diagnostics_enabled) {
          ++state.rar_elisions;
        }
      }
      for (const auto &[id, current] : task.accesses) {
        auto previous = pending_accesses.find(id);
        if (previous == pending_accesses.end()) {
          continue;
        }
        const bool current_hazard =
            (previous->second.write &&
             (current.read || current.write)) ||
            (previous->second.read && current.write);
        hazard = hazard || current_hazard;
        if (!current_hazard && state.diagnostics_enabled &&
            previous->second.read && current.read) {
          ++state.rar_elisions;
        }
      }
      if (hazard) {
        flush_structured_pending(/*exit_boundary=*/false);
      } else if (state.diagnostics_enabled) {
        for (const auto &[id, access] : pending_accesses) {
          (void)access;
          if (task.accesses.find(id) == task.accesses.end()) {
            ++state.barrier_deferrals;
          }
        }
      }
    };
    auto merge_structured_task = [&](const StructuredTask &task) {
      if (!structured_hazard_planner_enabled) {
        return;
      }
      for (const auto &[id, access] : task.accesses) {
        auto [it, inserted] = pending_accesses.try_emplace(id, access);
        if (!inserted) {
          it->second.read = it->second.read || access.read;
          it->second.write = it->second.write || access.write;
        }
      }
      pending_global_read = pending_global_read || task.global_read;
      pending_global_write = pending_global_write || task.global_write;
    };
    auto dispatch_task = [&](const StructuredTask &task,
                             const DevicePtr *indirect) {
      prepare_structured_task(task);
      cmdlist->bind_pipeline(
          task.dispatch->kernel->get_pipeline(task.task_index));
      RhiResult status =
          cmdlist->bind_shader_resources(task.resources);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured Graph resource binding error: "
                  "RhiResult({})",
                  status);
      status = indirect == nullptr
                   ? cmdlist->dispatch(task.group_x)
                   : cmdlist->dispatch_indirect(*indirect);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured Graph dispatch error: RhiResult({})",
                  status);
      if (structured_hazard_planner_enabled) {
        merge_structured_task(task);
      } else {
        cmdlist->memory_barrier();
        count_structured_barrier(/*exit_boundary=*/false);
      }
    };

    if (nested_structured) {
      auto dispatch_nested_controller =
          [&](Pipeline *pipeline, ShaderResourceSet *resources,
              const char *role) {
            flush_structured_pending(/*exit_boundary=*/false);
            cmdlist->bind_pipeline(pipeline);
            RhiResult status =
                cmdlist->bind_shader_resources(resources);
            TI_ERROR_IF(
                status != RhiResult::success,
                "Vulkan nested structured {} resource binding "
                "failed: RhiResult({})",
                role, status);
            status = cmdlist->dispatch(1);
            TI_ERROR_IF(
                status != RhiResult::success,
                "Vulkan nested structured {} dispatch failed: "
                "RhiResult({})",
                role, status);
            cmdlist->memory_barrier();
            count_structured_barrier(/*exit_boundary=*/false);
          };
      const std::size_t inner_control_bytes =
          nested_header_word_offset * sizeof(std::uint32_t);
      const DevicePtr outer_stable_predicate =
          slot->structured_control_buffer->get_ptr(
              nested_outer_gate_word_offset * sizeof(std::uint32_t));
      cmdlist->buffer_fill(slot->structured_control_buffer->get_ptr(),
                           structured_control_bytes, 0);
      cmdlist->buffer_transition(
          slot->structured_control_buffer->get_ptr(),
          structured_control_bytes,
          {BufferBarrierStage::Transfer,
           BufferBarrierAccess::TransferWrite,
           BufferBarrierStage::Compute |
               BufferBarrierStage::IndirectCommand,
           BufferBarrierAccess::ShaderRead |
               BufferBarrierAccess::ShaderWrite |
               BufferBarrierAccess::IndirectCommandRead});

      for (std::size_t i = 0;
           i < nested_outer_condition_task_end; ++i) {
        dispatch_task(structured_tasks[i], nullptr);
      }
      dispatch_nested_controller(
          slot->nested_outer_initial_pipeline.get(),
          slot->nested_outer_initial_resources.get(), "outer initial");

      for (std::uint32_t outer_iteration = 0;
           outer_iteration <
           nested_control->outer_max_iterations;
           ++outer_iteration) {
        flush_structured_pending(/*exit_boundary=*/false);
        cmdlist->buffer_transition(
            slot->structured_control_buffer->get_ptr(),
            inner_control_bytes,
            {BufferBarrierStage::Compute |
                 BufferBarrierStage::IndirectCommand |
                 BufferBarrierStage::ConditionalCommand,
             BufferBarrierAccess::ShaderRead |
                 BufferBarrierAccess::ShaderWrite |
                 BufferBarrierAccess::IndirectCommandRead |
                 BufferBarrierAccess::ConditionalCommandRead,
             BufferBarrierStage::Transfer,
             BufferBarrierAccess::TransferWrite});
        cmdlist->buffer_fill(
            slot->structured_control_buffer->get_ptr(),
            inner_control_bytes, 0);
        cmdlist->buffer_transition(
            slot->structured_control_buffer->get_ptr(),
            inner_control_bytes,
            {BufferBarrierStage::Transfer,
             BufferBarrierAccess::TransferWrite,
             BufferBarrierStage::Compute |
                 BufferBarrierStage::IndirectCommand,
             BufferBarrierAccess::ShaderRead |
                 BufferBarrierAccess::ShaderWrite |
                 BufferBarrierAccess::IndirectCommandRead});
        dispatch_nested_controller(
            slot->structured_gate_pipeline.get(),
            slot->structured_gate_resources.get(), "outer gate");
        cmdlist->buffer_transition(
            outer_stable_predicate,
            sizeof(std::uint32_t),
            {BufferBarrierStage::Compute,
             BufferBarrierAccess::ShaderWrite,
             BufferBarrierStage::ConditionalCommand,
             BufferBarrierAccess::ConditionalCommandRead});
        RhiResult status = cmdlist->begin_conditional(
            outer_stable_predicate);
        TI_ERROR_IF(
            status != RhiResult::success,
            "Vulkan nested outer conditional begin failed: "
            "RhiResult({})",
            status);

        std::size_t task_cursor = nested_outer_condition_task_end;
        for (std::size_t child = 0;
             child < nested_task_ranges.size(); ++child) {
          const auto &range = nested_task_ranges[child];
          const auto &inner = nested_control->inner_controls[child];
          for (std::size_t i = task_cursor; i < range.condition_begin; ++i) {
            dispatch_task(structured_tasks[i], nullptr);
          }
          dispatch_nested_controller(
              slot->nested_inner_initial_pipelines[child].get(),
              slot->nested_inner_initial_resources[child].get(),
              "inner initial");
          for (std::size_t i = range.condition_begin;
               i < range.body_begin; ++i) {
            dispatch_task(structured_tasks[i], nullptr);
          }

          const std::size_t packet_bytes =
              range.group_counts.size() * kStructuredPacketWords *
              sizeof(std::uint32_t);
          const DevicePtr packet_base =
              slot->structured_control_buffer->get_ptr(
                  range.packet_word_offset * sizeof(std::uint32_t));
          for (std::uint32_t inner_iteration = 0;
               inner_iteration < inner.inner_max_iterations;
               ++inner_iteration) {
            flush_structured_pending(/*exit_boundary=*/false);
            cmdlist->buffer_transition(
                inner.inner_predicate, sizeof(std::uint32_t),
                {BufferBarrierStage::Compute,
                 BufferBarrierAccess::ShaderWrite,
                 BufferBarrierStage::Compute,
                 BufferBarrierAccess::ShaderRead});
            cmdlist->bind_pipeline(
                slot->nested_inner_controller_pipelines[child].get());
            status = cmdlist->bind_shader_resources(
                slot->nested_inner_controller_resources[child].get());
            TI_ERROR_IF(
                status != RhiResult::success,
                "Vulkan nested inner controller resource binding "
                "failed: RhiResult({})",
                status);
            status = cmdlist->dispatch(1);
            TI_ERROR_IF(
                status != RhiResult::success,
                "Vulkan nested inner controller dispatch failed: "
                "RhiResult({})",
                status);
            cmdlist->memory_barrier();
            count_structured_barrier(/*exit_boundary=*/false);
            cmdlist->buffer_transition(
                packet_base, packet_bytes,
                {BufferBarrierStage::Compute,
                 BufferBarrierAccess::ShaderWrite,
                 BufferBarrierStage::IndirectCommand,
                 BufferBarrierAccess::IndirectCommandRead});
            for (std::size_t i = range.body_begin; i < range.end; ++i) {
              const std::size_t packet = i - range.body_begin;
              const DevicePtr indirect =
                  slot->structured_control_buffer->get_ptr(
                      (range.packet_word_offset +
                       packet * kStructuredPacketWords) *
                      sizeof(std::uint32_t));
              dispatch_task(structured_tasks[i], &indirect);
            }
            cmdlist->buffer_transition(
                packet_base, packet_bytes,
                {BufferBarrierStage::IndirectCommand,
                 BufferBarrierAccess::IndirectCommandRead,
                 BufferBarrierStage::Compute,
                 BufferBarrierAccess::ShaderRead |
                     BufferBarrierAccess::ShaderWrite});
          }

          dispatch_nested_controller(
              slot->nested_inner_terminal_pipelines[child].get(),
              slot->nested_inner_terminal_resources[child].get(),
              "inner terminal");
          task_cursor = range.end;
        }
        for (std::size_t i = task_cursor;
             i < structured_tasks.size(); ++i) {
          dispatch_task(structured_tasks[i], nullptr);
        }
        flush_structured_pending(/*exit_boundary=*/false);
        status = cmdlist->end_conditional();
        TI_ERROR_IF(
            status != RhiResult::success,
            "Vulkan nested outer conditional end failed: "
            "RhiResult({})",
            status);
      }

      dispatch_nested_controller(
          slot->nested_outer_terminal_pipeline.get(),
          slot->nested_outer_terminal_resources.get(),
          "outer terminal");
      const std::size_t observation_offset =
          nested_header_word_offset * sizeof(std::uint32_t);
      cmdlist->buffer_transition(
          slot->structured_control_buffer->get_ptr(
              observation_offset),
          structured_observation_bytes,
          {BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderWrite,
           BufferBarrierStage::Transfer,
           BufferBarrierAccess::TransferRead});
      cmdlist->buffer_copy(
          slot->structured_observation_buffer->get_ptr(),
          slot->structured_control_buffer->get_ptr(
              observation_offset),
          structured_observation_bytes);
      cmdlist->buffer_transition(
          slot->structured_observation_buffer->get_ptr(),
          structured_observation_bytes,
          {BufferBarrierStage::Transfer,
           BufferBarrierAccess::TransferWrite,
           BufferBarrierStage::Host,
           BufferBarrierAccess::HostRead});
      flush_structured_pending(/*exit_boundary=*/true);

      slot->key = std::move(key);
      slot->structure_key = std::move(structure_key);
      slot->cmdlist = std::move(recorded_cmdlist);
      slot->recorded = true;
      slot->completion =
          device_->get_compute_stream()->submit(
              slot->cmdlist.get());
      if (state.diagnostics_enabled) {
        ++state.recorded;
      }
      if (statistics != nullptr) {
        statistics->record_graph_capture();
        if (is_recapture) {
          statistics->record_graph_recapture();
        }
      }
      state.last_path = GraphReplayLastPath::record;
      return finish_structured_submission();
    }

    if (structured_strategy == GraphStructuredStrategy::chained) {
      cmdlist->buffer_fill(
          slot->structured_control_buffer->get_ptr(
              kStructuredPacketWords * sizeof(std::uint32_t)),
          structured_control_bytes -
              kStructuredPacketWords * sizeof(std::uint32_t),
          0);
      cmdlist->buffer_fill(slot->structured_control_buffer->get_ptr(),
                           kStructuredPacketWords * sizeof(std::uint32_t),
                           1);
    } else {
      cmdlist->buffer_fill(slot->structured_control_buffer->get_ptr(),
                           structured_control_bytes, 0);
    }
    const std::size_t structured_packet_bytes =
        structured_observation_word_offset * sizeof(std::uint32_t);
    cmdlist->buffer_transition(
        slot->structured_control_buffer->get_ptr(),
        structured_packet_bytes,
        {BufferBarrierStage::Transfer,
         BufferBarrierAccess::TransferWrite,
         structured_strategy == GraphStructuredStrategy::chained
             ? BufferBarrierStage::IndirectCommand |
                   BufferBarrierStage::Compute
             : BufferBarrierStage::Compute,
         structured_strategy == GraphStructuredStrategy::chained
             ? BufferBarrierAccess::IndirectCommandRead |
                   BufferBarrierAccess::ShaderRead |
                   BufferBarrierAccess::ShaderWrite
             : BufferBarrierAccess::ShaderRead |
                   BufferBarrierAccess::ShaderWrite});
    cmdlist->buffer_transition(
        slot->structured_control_buffer->get_ptr(structured_packet_bytes),
        structured_control_bytes - structured_packet_bytes,
        {BufferBarrierStage::Transfer,
         BufferBarrierAccess::TransferWrite,
         BufferBarrierStage::Compute,
         BufferBarrierAccess::ShaderRead |
             BufferBarrierAccess::ShaderWrite});
    if (structured_control->execute_initial_dispatches) {
      for (std::size_t i = 0; i < structured_initial_tasks; ++i) {
        dispatch_task(structured_tasks[i], nullptr);
      }
    }
    flush_structured_pending(/*exit_boundary=*/false);

    bool coarse_conditional_open = false;
    if (structured_strategy ==
        GraphStructuredStrategy::coarse_conditional) {
      cmdlist->buffer_transition(
          structured_control->predicate, sizeof(std::uint32_t),
          {BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderWrite,
           BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderRead});
      cmdlist->bind_pipeline(slot->structured_gate_pipeline.get());
      RhiResult status = cmdlist->bind_shader_resources(
          slot->structured_gate_resources.get());
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured coarse gate resource binding error: "
                  "RhiResult({})",
                  status);
      status = cmdlist->dispatch(1);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured coarse gate dispatch error: "
                  "RhiResult({})",
                  status);
      const DevicePtr stable_predicate =
          slot->structured_control_buffer->get_ptr(
              (structured_observation_word_offset +
               kStructuredObservationWords) *
              sizeof(std::uint32_t));
      cmdlist->buffer_transition(
          stable_predicate, sizeof(std::uint32_t),
          {BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderWrite,
           BufferBarrierStage::ConditionalCommand,
           BufferBarrierAccess::ConditionalCommandRead});
      status = cmdlist->begin_conditional(stable_predicate);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured coarse conditional begin failed: "
                  "RhiResult({})",
                  status);
      coarse_conditional_open = true;
    }

    for (std::uint32_t iteration = 0;
         iteration < structured_control->max_iterations; ++iteration) {
      flush_structured_pending(/*exit_boundary=*/false);
      cmdlist->buffer_transition(
          structured_control->predicate, sizeof(std::uint32_t),
          {BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderWrite,
           BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderRead});
      cmdlist->bind_pipeline(
          slot->structured_controller_pipeline.get());
      RhiResult status = cmdlist->bind_shader_resources(
          slot->structured_controller_resources.get());
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured controller resource binding error: "
                  "RhiResult({})",
                  status);
      if (structured_strategy == GraphStructuredStrategy::chained) {
        const std::size_t segment_words =
            (structured_group_counts.size() + 1) *
            kStructuredPacketWords;
        const DevicePtr controller_packet =
            slot->structured_control_buffer->get_ptr(
                iteration * segment_words * sizeof(std::uint32_t));
        status = cmdlist->dispatch_indirect(controller_packet);
      } else {
        status = cmdlist->dispatch(1);
      }
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured controller dispatch error: RhiResult({})",
                  status);
      if (structured_strategy == GraphStructuredStrategy::conditional) {
        const DevicePtr stable_predicate =
            slot->structured_control_buffer->get_ptr(
                iteration * sizeof(std::uint32_t));
        cmdlist->buffer_transition(
            stable_predicate, sizeof(std::uint32_t),
            {BufferBarrierStage::Compute,
             BufferBarrierAccess::ShaderWrite,
             BufferBarrierStage::ConditionalCommand,
             BufferBarrierAccess::ConditionalCommandRead});
        status = cmdlist->begin_conditional(stable_predicate);
        TI_ERROR_IF(status != RhiResult::success,
                    "Vulkan conditional compute begin failed: RhiResult({})",
                    status);
        for (std::size_t packet = 0;
             packet < structured_group_counts.size(); ++packet) {
          dispatch_task(
              structured_tasks[structured_initial_tasks + packet],
              nullptr);
        }
        status = cmdlist->end_conditional();
        TI_ERROR_IF(status != RhiResult::success,
                    "Vulkan conditional compute end failed: RhiResult({})",
                    status);
      } else {
        cmdlist->buffer_transition(
            slot->structured_control_buffer->get_ptr(),
            structured_packet_bytes,
            {BufferBarrierStage::Compute,
             BufferBarrierAccess::ShaderWrite,
             BufferBarrierStage::IndirectCommand,
             BufferBarrierAccess::IndirectCommandRead});

        for (std::size_t packet = 0;
             packet < structured_group_counts.size(); ++packet) {
          const std::size_t packet_word_offset =
              structured_strategy == GraphStructuredStrategy::chained
                  ? (static_cast<std::size_t>(iteration) *
                         (structured_group_counts.size() + 1) +
                     1 + packet) *
                        kStructuredPacketWords
                  : packet * kStructuredPacketWords;
          const DevicePtr indirect =
              slot->structured_control_buffer->get_ptr(
                  packet_word_offset * sizeof(std::uint32_t));
          dispatch_task(
              structured_tasks[structured_initial_tasks + packet],
              &indirect);
        }
        cmdlist->buffer_transition(
            slot->structured_control_buffer->get_ptr(),
            structured_packet_bytes,
            {BufferBarrierStage::IndirectCommand,
             BufferBarrierAccess::IndirectCommandRead,
             structured_strategy == GraphStructuredStrategy::chained
                 ? BufferBarrierStage::IndirectCommand |
                       BufferBarrierStage::Compute
                 : BufferBarrierStage::Compute,
             structured_strategy == GraphStructuredStrategy::chained
                 ? BufferBarrierAccess::IndirectCommandRead |
                       BufferBarrierAccess::ShaderRead |
                       BufferBarrierAccess::ShaderWrite
                 : BufferBarrierAccess::ShaderRead |
                       BufferBarrierAccess::ShaderWrite});
      }
    }

    if (coarse_conditional_open) {
      const RhiResult status = cmdlist->end_conditional();
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan structured coarse conditional end failed: "
                  "RhiResult({})",
                  status);
    }

    if (structured_terminal_observation) {
      flush_structured_pending(/*exit_boundary=*/false);
      cmdlist->bind_pipeline(slot->structured_terminal_pipeline.get());
      RhiResult terminal_status = cmdlist->bind_shader_resources(
          slot->structured_terminal_resources.get());
      TI_ERROR_IF(terminal_status != RhiResult::success,
                  "Vulkan structured terminal resource binding error: "
                  "RhiResult({})",
                  terminal_status);
      terminal_status = cmdlist->dispatch(1);
      TI_ERROR_IF(terminal_status != RhiResult::success,
                  "Vulkan structured terminal dispatch error: "
                  "RhiResult({})",
                  terminal_status);

      const std::size_t observation_offset =
          structured_observation_word_offset * sizeof(std::uint32_t);
      cmdlist->buffer_transition(
          slot->structured_control_buffer->get_ptr(observation_offset),
          structured_observation_bytes,
          {BufferBarrierStage::Compute,
           BufferBarrierAccess::ShaderWrite,
           BufferBarrierStage::Transfer,
           BufferBarrierAccess::TransferRead});
      cmdlist->buffer_copy(
          slot->structured_observation_buffer->get_ptr(),
          slot->structured_control_buffer->get_ptr(observation_offset),
          structured_observation_bytes);
      cmdlist->buffer_transition(
          slot->structured_observation_buffer->get_ptr(),
          structured_observation_bytes,
          {BufferBarrierStage::Transfer,
           BufferBarrierAccess::TransferWrite,
           BufferBarrierStage::Host,
           BufferBarrierAccess::HostRead});
    }
    flush_structured_pending(/*exit_boundary=*/true);

    slot->key = std::move(key);
    slot->structure_key = std::move(structure_key);
    slot->cmdlist = std::move(recorded_cmdlist);
    slot->recorded = true;
    slot->completion =
        device_->get_compute_stream()->submit(slot->cmdlist.get());
    if (state.diagnostics_enabled) {
      ++state.recorded;
    }
    if (statistics != nullptr) {
      statistics->record_graph_capture();
      if (is_recapture) {
        statistics->record_graph_recapture();
      }
    }
    state.last_path = GraphReplayLastPath::record;
    return finish_structured_submission();
  }

  struct BufferAccess {
    DeviceAllocation allocation{kDeviceNullAllocation};
    bool read{false};
    bool write{false};
  };
  using BufferAccessMap =
      std::unordered_map<DeviceAllocationId, BufferAccess>;

  // Disabled until the A/B qualification gate has covered real dependency
  // chains and independent branches. The legacy path preserves eager
  // placement and write-only dependency tracking while this is zero.
  const bool hazard_planner_enabled =
      get_environ_config("TI_VULKAN_GRAPH_HAZARD_PLANNER", 0) != 0;
  BufferAccessMap pending_accesses;
  bool pending_global_read = false;
  bool pending_global_write = false;

  auto count_barrier = [&](bool exit_boundary) {
    if (!state.diagnostics_enabled) {
      return;
    }
    if (exit_boundary) {
      ++state.exit_barriers;
    } else {
      ++state.dependency_barriers;
    }
  };
  auto insert_all_pending = [&](bool exit_boundary) {
    if (pending_global_read || pending_global_write ||
        !pending_accesses.empty()) {
      // A topology-stable memory barrier remains valid when a ready replay
      // slot patches runtime descriptors to same-structure allocations.
      cmdlist->memory_barrier();
      count_barrier(exit_boundary);
    }
    pending_global_read = false;
    pending_global_write = false;
    pending_accesses.clear();
  };
  auto insert_task_dependencies =
      [&](const BufferAccessMap &current_accesses,
          bool current_global_read,
          bool current_global_write) {
        if (!hazard_planner_enabled) {
          insert_all_pending(/*exit_boundary=*/false);
          return;
        }

        const bool current_has_access =
            current_global_read || current_global_write ||
            !current_accesses.empty();
        if ((pending_global_read || pending_global_write) &&
            current_has_access) {
          const bool only_reads =
              pending_global_read && !pending_global_write &&
              !current_global_write &&
              std::all_of(current_accesses.begin(), current_accesses.end(),
                          [](const auto &item) {
                            return item.second.read && !item.second.write;
                          });
          if (only_reads) {
            if (state.diagnostics_enabled) {
              ++state.rar_elisions;
            }
          } else {
            cmdlist->memory_barrier();
            count_barrier(/*exit_boundary=*/false);
            pending_global_read = false;
            pending_global_write = false;
            pending_accesses.clear();
            return;
          }
        }

        if ((current_global_read || current_global_write) &&
            !pending_accesses.empty()) {
          const bool only_reads =
              current_global_read && !current_global_write &&
              std::all_of(pending_accesses.begin(), pending_accesses.end(),
                          [](const auto &item) {
                            return item.second.read && !item.second.write;
                          });
          if (only_reads) {
            if (state.diagnostics_enabled) {
              ++state.rar_elisions;
            }
          } else {
            cmdlist->memory_barrier();
            count_barrier(/*exit_boundary=*/false);
            pending_global_read = false;
            pending_global_write = false;
            pending_accesses.clear();
            return;
          }
        }

        if (state.diagnostics_enabled) {
          for (const auto &[id, access] : pending_accesses) {
            (void)access;
            if (current_accesses.find(id) == current_accesses.end()) {
              ++state.barrier_deferrals;
            }
          }
        }
        bool dependency_hazard = false;
        for (const auto &[id, current] : current_accesses) {
          auto previous = pending_accesses.find(id);
          if (previous == pending_accesses.end()) {
            continue;
          }
          const bool hazard =
              (previous->second.write &&
               (current.read || current.write)) ||
              (previous->second.read && current.write);
          dependency_hazard = dependency_hazard || hazard;
          if (!hazard && state.diagnostics_enabled &&
              previous->second.read && current.read) {
            ++state.rar_elisions;
          }
        }
        if (dependency_hazard) {
          cmdlist->memory_barrier();
          count_barrier(/*exit_boundary=*/false);
          pending_global_read = false;
          pending_global_write = false;
          pending_accesses.clear();
        }
      };
  auto merge_pending_access = [&](const BufferAccess &access) {
    if (access.allocation == kDeviceNullAllocation) {
      return;
    }
    auto [it, inserted] =
        pending_accesses.try_emplace(access.allocation.alloc_id, access);
    if (!inserted) {
      it->second.read = it->second.read || access.read;
      it->second.write = it->second.write || access.write;
    }
  };

  size_t resource_set_index = 0;
  for (auto &pd : prepared) {
    const auto &task_attribs = pd.kernel->ti_kernel_attribs().tasks_attribs;
    for (int task_index = 0; task_index < task_attribs.size(); ++task_index) {
      const auto &attribs = task_attribs[task_index];
      BufferAccessMap current_accesses;
      bool current_global_read = false;
      bool current_global_write = false;
      for (const auto &bind : attribs.buffer_binds) {
        const bool may_read =
            (bind.access & TaskAttributes::BufferBind::kAccessRead) != 0;
        const bool may_write = bind.may_write();
        if (!may_read && !may_write) {
          continue;
        }
        if (bind.buffer.type == BufferType::Args ||
            bind.buffer.type == BufferType::ArgPack) {
          continue;
        }
        if (state.diagnostics_enabled) {
          state.effect_reads += may_read ? 1 : 0;
          state.effect_writes += may_write ? 1 : 0;
        }
        if (bind.buffer.type == BufferType::Rets ||
            bind.buffer.type == BufferType::NodeAllocatorPool) {
          current_global_read = current_global_read || may_read;
          current_global_write = current_global_write || may_write;
          continue;
        }
        DeviceAllocation allocation = kDeviceNullAllocation;
        if (bind.buffer.type == BufferType::ExtArr) {
          allocation = pd.any_arrays->at(bind.buffer.root_id);
        } else {
          DeviceAllocation *resolved =
              pd.kernel->get_buffer_bind(bind.buffer);
          if (resolved != nullptr) {
            allocation = *resolved;
          }
        }
        if (allocation == kDeviceNullAllocation) {
          current_global_read = current_global_read || may_read;
          current_global_write = current_global_write || may_write;
          continue;
        }
        auto [it, inserted] = current_accesses.try_emplace(
            allocation.alloc_id,
            BufferAccess{allocation, may_read, may_write});
        if (!inserted) {
          it->second.read = it->second.read || may_read;
          it->second.write = it->second.write || may_write;
        }
      }
      insert_task_dependencies(current_accesses, current_global_read,
                               current_global_write);

      ShaderResourceSet *bindings =
          slot->resource_sets[resource_set_index++].get();
      TI_ERROR_IF(
          !update_task_bindings(pd, task_index, bindings,
                                /*patch_existing=*/false),
          "Vulkan graph replay descriptor preparation failed");

      cmdlist->bind_pipeline(pd.kernel->get_pipeline(task_index));
      RhiResult status = cmdlist->bind_shader_resources(bindings);
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan graph resource binding error: RhiResult({})",
                  status);

      const int group_x = (attribs.advisory_total_num_threads +
                           attribs.advisory_num_threads_per_group - 1) /
                          attribs.advisory_num_threads_per_group;
      if (pd.indirect_dispatch == kDeviceNullPtr) {
        status = cmdlist->dispatch(group_x);
      } else {
        insert_all_pending(/*exit_boundary=*/false);
        cmdlist->buffer_transition(
            pd.indirect_dispatch, 3 * sizeof(std::uint32_t),
            {BufferBarrierStage::Transfer |
                 BufferBarrierStage::Compute,
             BufferBarrierAccess::TransferWrite |
                 BufferBarrierAccess::ShaderWrite,
             BufferBarrierStage::IndirectCommand,
             BufferBarrierAccess::IndirectCommandRead});
        status = cmdlist->dispatch_indirect(pd.indirect_dispatch);
        cmdlist->buffer_transition(
            pd.indirect_dispatch, 3 * sizeof(std::uint32_t),
            {BufferBarrierStage::IndirectCommand,
             BufferBarrierAccess::IndirectCommandRead,
             BufferBarrierStage::Compute,
             BufferBarrierAccess::ShaderRead |
                 BufferBarrierAccess::ShaderWrite});
      }
      TI_ERROR_IF(status != RhiResult::success,
                  "Vulkan graph dispatch error: RhiResult({})", status);

      if (hazard_planner_enabled) {
        for (const auto &[id, access] : current_accesses) {
          (void)id;
          merge_pending_access(access);
        }
        pending_global_read =
            pending_global_read || current_global_read;
        pending_global_write =
            pending_global_write || current_global_write;
      } else {
        for (const auto &[id, access] : current_accesses) {
          (void)id;
          if (access.write) {
            merge_pending_access(
                BufferAccess{access.allocation, false, true});
          }
        }
        pending_global_write =
            pending_global_write || current_global_write;
      }
    }
  }
  insert_all_pending(/*exit_boundary=*/true);

  slot->key = std::move(key);
  slot->structure_key = std::move(structure_key);
  slot->cmdlist = std::move(recorded_cmdlist);
  slot->recorded = true;
  slot->completion = device_->get_compute_stream()->submit(slot->cmdlist.get());
  if (state.diagnostics_enabled) {
    ++state.recorded;
  }
  if (statistics != nullptr) {
    statistics->record_graph_capture();
    if (is_recapture) {
      statistics->record_graph_recapture();
    }
  }
  state.last_path = GraphReplayLastPath::record;
  return true;
}

void GfxRuntime::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->buffer_barrier(src);
  current_cmdlist_->buffer_copy(dst, src, size);
  current_cmdlist_->buffer_barrier(dst);
}

void GfxRuntime::copy_image(DeviceAllocation dst,
                            DeviceAllocation src,
                            const ImageCopyParams &params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  transition_image(dst, ImageLayout::transfer_dst);
  transition_image(src, ImageLayout::transfer_src);
  current_cmdlist_->copy_image(dst, src, ImageLayout::transfer_dst,
                               ImageLayout::transfer_src, params);
  transition_image(dst, ImageLayout::transfer_src);
}

DeviceAllocation GfxRuntime::create_image(const ImageParams &params) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  GraphicsDevice *gfx_device = dynamic_cast<GraphicsDevice *>(device_);
  TI_ERROR_IF(gfx_device == nullptr,
              "Image can only be created on a graphics device");
  DeviceAllocation image = gfx_device->create_image(params);
  track_image(image, ImageLayout::undefined);
  last_image_layouts_.at(image.alloc_id) = params.initial_layout;
  image_sampler_configs_.at(image.alloc_id) = params.sampler_config;
  return image;
}

void GfxRuntime::track_image(DeviceAllocation image, ImageLayout layout) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  last_image_layouts_[image.alloc_id] = layout;
  image_sampler_configs_.try_emplace(image.alloc_id, ImageSamplerConfig{});
}
void GfxRuntime::untrack_image(DeviceAllocation image) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  invalidate_graphics_command_replay_locked(image);
  last_image_layouts_.erase(image.alloc_id);
  image_sampler_configs_.erase(image.alloc_id);
}
void GfxRuntime::transition_image(DeviceAllocation image, ImageLayout layout) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  ImageLayout &last_layout = last_image_layouts_.at(image.alloc_id);
  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  current_cmdlist_->image_transition(image, last_layout, layout);
  last_layout = layout;
}

void GfxRuntime::synchronize() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  synchronize_impl(/*check_hash_overflow=*/true);
}

void GfxRuntime::synchronize_impl(bool check_hash_overflow) {
  flush_if_pending();
  device_->get_compute_stream()->command_sync();
  if (graphics_submission_used_) {
    // Every runtime graphics submission publishes a compute-stream bridge, so
    // graphics is already complete here. Synchronizing its stream retires the
    // command buffers and their pipeline/image references without extending
    // ordinary execution with a host wait.
    auto *graphics_device = dynamic_cast<GraphicsDevice *>(device_);
    TI_ASSERT(graphics_device != nullptr);
    graphics_device->get_graphics_stream()->command_sync();
  }
  // The stream is idle, so every retirement-requested state can now release
  // its graph-owned command/resource objects.
  collect_ready_graph_replays(graph_replay_retirement_count_);
  // Profiler support
  if (profiler_) {
    device_->profiler_sync();
    auto sampled_records = device_->profiler_flush_sampled_time();
    for (auto &record : sampled_records) {
      profiler_->insert_record(record.first, record.second);
    }
  }
  ctx_buffers_.clear();
  // R2.a: after wait_idle, all submitted/pending pool buffers are GPU-safe.
  if (ctx_buffer_pool_enabled()) {
    recycle_pools_to_free();
  }
  ndarrays_in_use_.clear();
  argpacks_in_use_.clear();
  // Hash SNodes use a static table. Since there is no device-side grow path,
  // overflow must become a host-visible error at the sync boundary instead of
  // silently returning ambient-zero cells.
  if (check_hash_overflow) {
    check_hash_overflow_counters();
  }
  fflush(stdout);
}

StreamSemaphore GfxRuntime::flush() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  StreamSemaphore sema;
  if (current_cmdlist_) {
    insert_pending_dispatch_barriers();
    sema = device_->get_compute_stream()->submit(current_cmdlist_.get());
    current_cmdlist_ = nullptr;
    current_cmdlist_dispatch_count_ = 0;
    ctx_buffers_.clear();
    if (ctx_buffer_pool_enabled()) {
      flush_pending_pool_to_submitted(sema);
    }
  } else {
    auto [cmdlist, res] =
        device_->get_compute_stream()->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);
    cmdlist->memory_barrier();
    sema = device_->get_compute_stream()->submit(cmdlist.get());
  }
  current_cmdlist_dispatch_count_ = 0;
  latest_compute_completion_ = sema;
  return sema;
}

StreamSemaphore GfxRuntime::flush_if_pending() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (!current_cmdlist_) {
    return nullptr;
  }
  return flush();
}

void GfxRuntime::begin_submission_batch() {
  // RuntimeSubmissionTransaction is thread-affine. Holding the recursive host
  // API lock across its calls prevents an unrelated producer from being
  // accidentally absorbed into this stream batch.
  host_api_mutex_.lock();
  try {
    flush_if_pending();
    device_->get_compute_stream()->begin_submission_batch();
  } catch (...) {
    host_api_mutex_.unlock();
    throw;
  }
}

StreamSemaphore GfxRuntime::end_submission_batch() {
  try {
    if (current_cmdlist_) {
      flush();
    }
    StreamSemaphore completion =
        device_->get_compute_stream()->end_submission_batch();
    if (completion) {
      latest_compute_completion_ = completion;
    }
    host_api_mutex_.unlock();
    return completion;
  } catch (...) {
    host_api_mutex_.unlock();
    throw;
  }
}

StreamGpuTiming GfxRuntime::begin_gpu_timing() {
  // RuntimeSubmissionTransaction owns host_api_mutex_ across the batch. Put
  // the marker into the current command list so timing does not split a short
  // structured Graph into one command buffer per region boundary.
  ensure_current_cmdlist();
  return device_->get_compute_stream()->begin_gpu_timing_inline(
      current_cmdlist_.get());
}

void GfxRuntime::end_gpu_timing(const StreamGpuTiming &timing) {
  if (!timing) {
    return;
  }
  // A structured/native payload may have flushed the command list after the
  // begin marker. Reuse the next runtime-owned list instead of allocating a
  // marker-only command list through Stream::end_gpu_timing().
  ensure_current_cmdlist();
  device_->get_compute_stream()->end_gpu_timing_inline(timing,
                                                       current_cmdlist_.get());
}

Device *GfxRuntime::get_ti_device() const {
  return device_;
}

PipelineCache *GfxRuntime::get_backend_cache() const {
  return backend_cache_.get();
}

// G-1: Try to take a buffer from free_pool_ matching (size, usage).
// Returns nullptr if pool disabled or no match. Performs O(N) scan; pool
// capacity is bounded (default 64) so this is acceptable.
std::unique_ptr<DeviceAllocationGuard> GfxRuntime::try_take_pooled_buffer(
    size_t size,
    AllocUsage usage) {
  if (!ctx_buffer_pool_enabled()) {
    return nullptr;
  }
  for (auto it = free_pool_.begin(); it != free_pool_.end(); ++it) {
    if (it->size == size && it->usage == usage) {
      auto guard = std::move(it->guard);
      free_pool_.erase(it);
      return guard;
    }
  }
  return nullptr;
}

bool GfxRuntime::ctx_buffer_pool_enabled() const {
  return ctx_buffer_ring_enabled_;
}

size_t GfxRuntime::count_pooled_buffers(size_t size, AllocUsage usage) const {
  auto count_in = [&](const std::vector<PooledBuffer> &pool) {
    size_t count = 0;
    for (const auto &entry : pool) {
      if (entry.size == size && entry.usage == usage) {
        ++count;
      }
    }
    return count;
  };
  return count_in(pending_pool_) + count_in(submitted_pool_) +
         count_in(free_pool_);
}

std::unique_ptr<DeviceAllocationGuard> GfxRuntime::acquire_ctx_buffer(
    size_t size,
    AllocUsage usage) {
  if (!ctx_buffer_pool_enabled()) {
    return nullptr;
  }
  recycle_completed_pools_to_free();
  if (auto pooled = try_take_pooled_buffer(size, usage)) {
    return pooled;
  }
  if (ctx_buffer_ring_enabled_ &&
      count_pooled_buffers(size, usage) >= ctx_buffer_ring_size_) {
    if (current_cmdlist_) {
      flush();
    }
    recycle_completed_pools_to_free();
    if (auto pooled = try_take_pooled_buffer(size, usage)) {
      return pooled;
    }
    if (wait_for_oldest_submitted_buffer(size, usage)) {
      recycle_completed_pools_to_free();
    } else {
      // Backend does not expose a submission completion token. Fall back to
      // syncing only the compute stream; this is still narrower than the old
      // device_->wait_idle(), which also synced graphics streams.
      device_->get_compute_stream()->command_sync();
      recycle_pools_to_free();
    }
    if (auto pooled = try_take_pooled_buffer(size, usage)) {
      return pooled;
    }
  }
  return nullptr;
}

void GfxRuntime::flush_pending_pool_to_submitted(StreamSemaphore completion) {
  for (auto &entry : pending_pool_) {
    entry.completion = completion;
    submitted_pool_.push_back(std::move(entry));
  }
  pending_pool_.clear();
}

size_t GfxRuntime::recycle_completed_pools_to_free() {
  if (submitted_pool_.empty()) {
    return 0;
  }
  size_t recycled = 0;
  std::vector<PooledBuffer> still_submitted;
  still_submitted.reserve(submitted_pool_.size());
  auto move_to_free = [this, &recycled](PooledBuffer &&entry) {
    entry.completion = nullptr;
    if (free_pool_.size() >= buffer_pool_capacity_) {
      // Drop oldest free entry to bound memory; releases its DeviceAlloc.
      free_pool_.erase(free_pool_.begin());
    }
    free_pool_.push_back(std::move(entry));
    ++recycled;
  };
  for (auto &entry : submitted_pool_) {
    if (entry.completion && entry.completion->is_ready()) {
      move_to_free(std::move(entry));
    } else {
      still_submitted.push_back(std::move(entry));
    }
  }
  submitted_pool_ = std::move(still_submitted);
  return recycled;
}

bool GfxRuntime::wait_for_oldest_submitted_buffer(size_t size,
                                                  AllocUsage usage) {
  for (const auto &entry : submitted_pool_) {
    if (entry.size == size && entry.usage == usage && entry.completion) {
      return entry.completion->wait();
    }
  }
  return false;
}

void GfxRuntime::recycle_pools_to_free() {
  // Caller must guarantee GPU has finished (wait_idle / submit_synced).
  auto move_to_free = [this](std::vector<PooledBuffer> &src) {
    for (auto &entry : src) {
      entry.completion = nullptr;
      if (free_pool_.size() >= buffer_pool_capacity_) {
        // Drop oldest free entry to bound memory; releases its DeviceAlloc.
        free_pool_.erase(free_pool_.begin());
      }
      free_pool_.push_back(std::move(entry));
    }
    src.clear();
  };
  move_to_free(submitted_pool_);
  move_to_free(pending_pool_);
}

void GfxRuntime::ensure_current_cmdlist() {
  // Create new command list if current one is nullptr
  if (!current_cmdlist_) {
    current_cmdlist_pending_since_ = high_res_clock::now();
    current_cmdlist_dispatch_count_ = 0;
    auto [cmdlist, res] =
        device_->get_compute_stream()->new_command_list_unique();
    TI_ASSERT(res == RhiResult::success);
    current_cmdlist_ = std::move(cmdlist);
  }
}

void GfxRuntime::clear_pending_dispatch_barriers() {
  pending_dispatch_global_barrier_ = false;
  pending_dispatch_barrier_buffers_.clear();
  pending_dispatch_barrier_buffer_ids_.clear();
}

void GfxRuntime::add_pending_dispatch_barrier(DeviceAllocation alloc) {
  if (!dispatch_cache_ || alloc == kDeviceNullAllocation) {
    return;
  }
  if (pending_dispatch_global_barrier_) {
    return;
  }
  if (pending_dispatch_barrier_buffer_ids_.insert(alloc.alloc_id).second) {
    pending_dispatch_barrier_buffers_.push_back(alloc);
  }
}

void GfxRuntime::insert_pending_dispatch_barriers() {
  if (!dispatch_cache_ || !current_cmdlist_) {
    return;
  }
  if (!pending_dispatch_global_barrier_ &&
      pending_dispatch_barrier_buffers_.empty()) {
    return;
  }
  if (pending_dispatch_global_barrier_) {
    current_cmdlist_->memory_barrier();
    clear_pending_dispatch_barriers();
    return;
  }
  for (DeviceAllocation alloc : pending_dispatch_barrier_buffers_) {
    current_cmdlist_->buffer_barrier(alloc);
  }
  clear_pending_dispatch_barriers();
}

void GfxRuntime::submit_current_cmdlist_if_timeout() {
  // If we have accumulated some work but does not require sync
  // and if the accumulated cmdlist has been pending for some time
  // launch the cmdlist to start processing.
  if (current_cmdlist_) {
    if (cmdlist_lazy_submit_enabled_ && debug_mode_) {
      flush();
      return;
    }
    if (cmdlist_lazy_submit_enabled_ &&
        cmdlist_lazy_submit_min_dispatches_ > 0 &&
        current_cmdlist_dispatch_count_ <
            cmdlist_lazy_submit_min_dispatches_) {
      return;
    }
    constexpr uint64_t max_pending_time = 2000;  // 2000us = 2ms
    auto duration = high_res_clock::now() - current_cmdlist_pending_since_;
    if (std::chrono::duration_cast<std::chrono::microseconds>(duration)
            .count() > max_pending_time) {
      flush();
    }
  }
}

void GfxRuntime::ensure_listgen_buffer_bytes(size_t requested_bytes,
                                             const char *reason) {
  requested_bytes = std::max(requested_bytes, kListGenMinBufferSize);
  requested_bytes = align_up_to(requested_bytes, sizeof(uint32_t));
  if (requested_bytes > kListGenMinBufferSize) {
    requested_bytes = align_up_to(requested_bytes, kListGenBufferAlignment);
  }

  if (listgen_buffer_ && requested_bytes <= listgen_buffer_size_) {
    return;
  }
  if (listgen_buffer_ && listgen_explicit_size_) {
    TI_ERROR(
        "Vulkan listgen buffer capacity is fixed at {:.3f} MiB by "
        "vulkan_listgen_buffer_MB, but {} requires {:.3f} MiB. Increase "
        "vulkan_listgen_buffer_MB or unset it and enable "
        "vulkan_listgen_dynamic_size.",
        listgen_buffer_size_ / 1048576.0, reason,
        requested_bytes / 1048576.0);
  }

  const bool replacing_used_buffer = listgen_buffer_ && listgen_buffer_used_;
  if (replacing_used_buffer) {
    synchronize();
  }

  auto [buf, res] = device_->allocate_memory_unique(
      {requested_bytes,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});
  TI_ASSERT_INFO(res == RhiResult::success, "listgen allocation failed");
  listgen_buffer_ = std::move(buf);
  listgen_buffer_size_ = requested_bytes;
  listgen_capacity_entries_ =
      requested_bytes / sizeof(uint32_t) > 0
          ? requested_bytes / sizeof(uint32_t) - 1
          : 0;

  for (auto &[launch_id, kernel] : ti_kernels_) {
    (void)launch_id;
    kernel->set_listgen_buffer(listgen_buffer_.get());
  }
  if (replacing_used_buffer) {
    clear_sparse_list_cache_resident();
  }
  listgen_buffer_used_ = false;

  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, cmd_res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(cmd_res == RhiResult::success);
  cmdlist->buffer_fill(listgen_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());

  if (listgen_dynamic_size_ && !listgen_explicit_size_ &&
      requested_bytes > kListGenMinBufferSize) {
    TI_WARN(
        "Vulkan listgen buffer auto-sized to {:.3f} MiB for {} "
        "(capacity {} entries). Set ti.init(vulkan_listgen_buffer_MB=32) "
        "to force the legacy 32 MiB capacity.",
        requested_bytes / 1048576.0, reason, listgen_capacity_entries_);
  }
}

void GfxRuntime::ensure_listgen_capacity_entries(size_t requested_entries,
                                                 const char *reason) {
  if (requested_entries <= listgen_capacity_entries_) {
    return;
  }
  if (!listgen_dynamic_size_ || listgen_explicit_size_) {
    TI_ERROR(
        "Vulkan listgen buffer capacity {} entries ({:.3f} MiB) is smaller "
        "than {} required entries for {}. Set ti.init("
        "vulkan_listgen_buffer_MB=...) high enough or enable "
        "vulkan_listgen_dynamic_size=True for auto sizing.",
        listgen_capacity_entries_, listgen_buffer_size_ / 1048576.0,
        requested_entries, reason);
  }
  const size_t required_words =
      requested_entries + 1 + kListGenAutoSlackEntries;
  ensure_listgen_buffer_bytes(required_words * sizeof(uint32_t), reason);
}

void GfxRuntime::ensure_listgen_capacity_for_kernel(
    const CompiledTaichiKernel &kernel) {
  const size_t requested_entries =
      kernel.required_listgen_capacity_entries();
  if (requested_entries > 0) {
    ensure_listgen_capacity_entries(requested_entries, "kernel launch");
  }
}

void GfxRuntime::update_listgen_buffer_for_snode_tree(
    const CompiledSNodeStructs &compiled_structs) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  const size_t requested_entries = estimate_listgen_entries(compiled_structs);
  if (requested_entries > 0) {
    ensure_listgen_capacity_entries(requested_entries,
                                    "SNodeTree materialization");
  }
}

void GfxRuntime::init_nonroot_buffers() {
  {
    auto [buf, res] = device_->allocate_memory_unique(
        {kGtmpBufferSize,
         /*host_write=*/false, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT_INFO(res == RhiResult::success, "gtmp allocation failed");
    global_tmps_buffer_ = std::move(buf);
  }
  {
    auto [buf, res] = device_->allocate_memory_unique(
        {kHashOverflowBufferSize,
         /*host_write=*/false, /*host_read=*/true,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT_INFO(res == RhiResult::success,
                   "hash overflow diagnostics allocation failed");
    hash_overflow_buffer_ = std::move(buf);
  }

  if (listgen_initial_buffer_size_ > 0) {
    ensure_listgen_buffer_bytes(listgen_initial_buffer_size_, "initialization");
  }

  // Need to zero fill the buffers, otherwise there could be NaN.
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);

  cmdlist->buffer_fill(global_tmps_buffer_->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  cmdlist->buffer_fill(hash_overflow_buffer_->get_ptr(0),
                       kBufferSizeEntireSize, /*data=*/0);
  stream->submit_synced(cmdlist.get());
}

void GfxRuntime::add_root_buffer(size_t root_buffer_size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  add_root_buffer(static_cast<int>(root_buffers_.size()), root_buffer_size);
}

void GfxRuntime::add_root_buffer(int root_id, size_t root_buffer_size) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(root_id < 0 ||
                  static_cast<std::size_t>(root_id) > root_buffers_.size(),
              "Cannot install root buffer at invalid id {}.", root_id);
  TI_ERROR_IF(static_cast<std::size_t>(root_id) < root_buffers_.size() &&
                  root_buffers_[root_id],
              "Cannot replace live root buffer id {}.", root_id);
  if (root_buffer_size == 0) {
    root_buffer_size = 4;  // there might be empty roots
  }
  auto [new_buffer, res_buffer] = device_->allocate_memory_unique(
      {root_buffer_size,
       /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Storage});
  TI_ASSERT_INFO(res_buffer == RhiResult::success,
                 "Failed to allocate root buffer");
  Stream *stream = device_->get_compute_stream();
  auto [cmdlist, res_cmdlist] =
      device_->get_compute_stream()->new_command_list_unique();
  TI_ASSERT(res_cmdlist == RhiResult::success);
  cmdlist->buffer_fill(new_buffer->get_ptr(0), kBufferSizeEntireSize,
                       /*data=*/0);
  stream->submit_synced(cmdlist.get());
  if (static_cast<std::size_t>(root_id) == root_buffers_.size()) {
    root_buffers_.push_back(std::move(new_buffer));
  } else {
    root_buffers_[root_id] = std::move(new_buffer);
  }
  // cache the root buffer size
  root_buffers_size_map_[root_buffers_[root_id].get()] = root_buffer_size;
}

void GfxRuntime::remove_root_buffer(int root_id) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(root_id < 0 ||
                  static_cast<std::size_t>(root_id) >= root_buffers_.size() ||
                  !root_buffers_[root_id],
              "Cannot remove missing root buffer id {}.", root_id);
  root_buffers_size_map_.erase(root_buffers_[root_id].get());
  root_buffers_[root_id].reset();
  hash_overflow_watches_.erase(
      std::remove_if(hash_overflow_watches_.begin(),
                     hash_overflow_watches_.end(),
                     [root_id](const HashOverflowWatch &watch) {
                       return watch.root_id == root_id;
                     }),
      hash_overflow_watches_.end());
}

DeviceAllocation *GfxRuntime::get_root_buffer(int id) const {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (id < 0 || static_cast<size_t>(id) >= root_buffers_.size() ||
      !root_buffers_[id]) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return root_buffers_[id].get();
}

size_t GfxRuntime::get_root_buffer_size(int id) const {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  if (id < 0 || static_cast<size_t>(id) >= root_buffers_.size() ||
      !root_buffers_[id]) {
    TI_ERROR("root buffer id {} not found", id);
  }
  auto it = root_buffers_size_map_.find(root_buffers_[id].get());
  if (it == root_buffers_size_map_.end()) {
    TI_ERROR("root buffer id {} not found", id);
  }
  return it->second;
}

void GfxRuntime::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  for (const auto &ref : image_refs) {
    TI_ASSERT(last_image_layouts_.find(ref.image.alloc_id) !=
              last_image_layouts_.end());
    transition_image(ref.image, ref.initial_layout);
  }

  ensure_current_cmdlist();
  insert_pending_dispatch_barriers();
  op(device_, current_cmdlist_.get());

  for (const auto &ref : image_refs) {
    transition_image(ref.image, ref.final_layout);
  }
}

void GfxRuntime::enqueue_graphics_op_lambda(
    std::function<void(GraphicsDevice *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs,
    const std::vector<std::uint64_t> &replay_key) {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  TI_ERROR_IF(!op, "Runtime graphics operation must not be empty");
  auto *graphics_device = dynamic_cast<GraphicsDevice *>(device_);
  TI_ERROR_IF(graphics_device == nullptr,
              "The active runtime device does not support graphics commands");

  for (const auto &ref : image_refs) {
    TI_ERROR_IF(last_image_layouts_.find(ref.image.alloc_id) ==
                    last_image_layouts_.end(),
                "Runtime graphics operation references an untracked image");
  }

  // Publish all preceding runtime work before crossing to the graphics queue.
  // If there is no current list, latest_compute_completion_ still represents
  // the most recent explicit flush or the bridge from an earlier graphics op.
  if (StreamSemaphore flushed = flush_if_pending()) {
    latest_compute_completion_ = std::move(flushed);
  }

  Stream *graphics_stream = graphics_device->get_graphics_stream();
  const bool replay_requested = !replay_key.empty();
  if (!replay_requested && !retained_graphics_replay_.key.empty()) {
    const bool touches_retained_image = std::any_of(
        image_refs.begin(), image_refs.end(), [&](const auto &ref) {
          return std::find(retained_graphics_replay_.images.begin(),
                           retained_graphics_replay_.images.end(), ref.image) !=
                 retained_graphics_replay_.images.end();
        });
    if (touches_retained_image) {
      invalidate_graphics_command_replay_locked();
    }
  }

  bool replay = false;
  bool record_retained = false;
  std::size_t retained_slot_index =
      RetainedGraphicsCommandReplay::kSlotCapacity;
  if (replay_requested) {
    ++retained_graphics_replay_.attempts;
    const bool exact_key = retained_graphics_replay_.key == replay_key;
    const bool layouts_stable = std::all_of(
        image_refs.begin(), image_refs.end(), [&](const auto &ref) {
          return last_image_layouts_.at(ref.image.alloc_id) == ref.final_layout;
        });
    if (!exact_key || !layouts_stable) {
      if (!retained_graphics_replay_.key.empty()) {
        if (!exact_key) {
          ++retained_graphics_replay_.binding_misses;
        }
        if (exact_key && !layouts_stable) {
          ++retained_graphics_replay_.layout_misses;
        }
        invalidate_graphics_command_replay_locked();
      }
      retained_graphics_replay_.key = replay_key;
      retained_graphics_replay_.images.clear();
      retained_graphics_replay_.images.reserve(image_refs.size());
      for (const auto &ref : image_refs) {
        retained_graphics_replay_.images.push_back(ref.image);
      }
      retained_graphics_replay_.prewarmed = true;
      ++retained_graphics_replay_.prewarms;
      if (exact_key && !layouts_stable) {
        ++retained_graphics_replay_.fallbacks;
      }
      retained_graphics_replay_.last_path = 1;
    } else if (retained_graphics_replay_.prewarmed) {
      for (std::size_t offset = 0;
           offset < RetainedGraphicsCommandReplay::kSlotCapacity; ++offset) {
        const std::size_t index =
            (retained_graphics_replay_.next_slot + offset) %
            RetainedGraphicsCommandReplay::kSlotCapacity;
        const auto &slot = retained_graphics_replay_.slots[index];
        if (slot.command_list &&
            (!slot.completion || slot.completion->is_ready())) {
          retained_slot_index = index;
          replay = true;
          break;
        }
      }
      if (!replay) {
        for (std::size_t offset = 0;
             offset < RetainedGraphicsCommandReplay::kSlotCapacity; ++offset) {
          const std::size_t index =
              (retained_graphics_replay_.next_slot + offset) %
              RetainedGraphicsCommandReplay::kSlotCapacity;
          if (!retained_graphics_replay_.slots[index].command_list) {
            retained_slot_index = index;
            record_retained = true;
            break;
          }
        }
      }
      if (replay || record_retained) {
        retained_graphics_replay_.next_slot =
            (retained_slot_index + 1) %
            RetainedGraphicsCommandReplay::kSlotCapacity;
        retained_graphics_replay_.last_path = replay ? 3 : 2;
      } else {
        ++retained_graphics_replay_.fallbacks;
        ++retained_graphics_replay_.busy_fallbacks;
        retained_graphics_replay_.last_path = 4;
      }
    }
  } else {
    retained_graphics_replay_.last_path = 0;
  }

  std::unique_ptr<CommandList> graphics_commands;
  CommandList *submitted_commands = nullptr;
  if (replay) {
    submitted_commands = retained_graphics_replay_
                             .slots[retained_slot_index]
                             .command_list.get();
    for (const auto &ref : image_refs) {
      last_image_layouts_[ref.image.alloc_id] = ref.final_layout;
    }
  } else {
    auto [new_commands, graphics_result] =
        graphics_stream->new_command_list_unique();
    TI_ERROR_IF(
        graphics_result != RhiResult::success,
        "Runtime graphics command-list allocation failed: RhiResult({})",
        graphics_result);
    graphics_commands = std::move(new_commands);

    for (const auto &ref : image_refs) {
      const ImageLayout previous = last_image_layouts_.at(ref.image.alloc_id);
      if (previous != ref.initial_layout) {
        graphics_commands->image_transition(ref.image, previous,
                                            ref.initial_layout);
      }
    }

    op(graphics_device, graphics_commands.get());

    for (const auto &ref : image_refs) {
      if (ref.initial_layout != ref.final_layout) {
        graphics_commands->image_transition(ref.image, ref.initial_layout,
                                            ref.final_layout);
      }
      last_image_layouts_[ref.image.alloc_id] = ref.final_layout;
    }

    if (record_retained) {
      auto &slot = retained_graphics_replay_.slots[retained_slot_index];
      slot.command_list = std::move(graphics_commands);
      submitted_commands = slot.command_list.get();
      retained_graphics_replay_.peak_slots = std::max<std::uint64_t>(
          retained_graphics_replay_.peak_slots,
          std::count_if(retained_graphics_replay_.slots.begin(),
                        retained_graphics_replay_.slots.end(),
                        [](const auto &candidate) {
                          return bool(candidate.command_list);
                        }));
    } else {
      submitted_commands = graphics_commands.get();
    }
  }

  std::vector<StreamSemaphore> graphics_waits;
  if (latest_compute_completion_) {
    graphics_waits.push_back(latest_compute_completion_);
  }
  StreamSemaphore graphics_completion;
  try {
    graphics_completion =
        graphics_stream->submit(submitted_commands, graphics_waits);
    TI_ERROR_IF(!graphics_completion,
                "Runtime graphics submission returned no completion token");
  } catch (...) {
    ++retained_graphics_replay_.submit_failures;
    if (record_retained || replay) {
      invalidate_graphics_command_replay_locked();
    }
    throw;
  }
  ++retained_graphics_replay_.graphics_submissions;
  if (record_retained) {
    ++retained_graphics_replay_.records;
  } else if (replay) {
    ++retained_graphics_replay_.replays;
  }
  if (record_retained || replay) {
    retained_graphics_replay_.slots[retained_slot_index].completion =
        graphics_completion;
  }

  // Bring the dependency back to the compute stream immediately. All later
  // kernel submissions and RuntimeCompletion tickets already use this stream,
  // so no second lifetime/completion domain is introduced.
  try {
    Stream *compute_stream = device_->get_compute_stream();
    auto [bridge_commands, bridge_result] =
        compute_stream->new_command_list_unique();
    TI_ERROR_IF(bridge_result != RhiResult::success,
                "Runtime graphics completion bridge allocation failed: "
                "RhiResult({})",
                bridge_result);
    bridge_commands->memory_barrier();
    latest_compute_completion_ =
        compute_stream->submit(bridge_commands.get(), {graphics_completion});
    TI_ERROR_IF(!latest_compute_completion_,
                "Runtime graphics completion bridge returned no token");
  } catch (...) {
    ++retained_graphics_replay_.bridge_failures;
    ++retained_graphics_replay_.submit_failures;
    // Preserve the graphics completion as the next ordering dependency even
    // when bridge recording fails. A non-fatal caller may then fall back
    // without submitting compute work ahead of the graphics operation.
    latest_compute_completion_ = graphics_completion;
    if (record_retained || replay) {
      invalidate_graphics_command_replay_locked();
    }
    throw;
  }
  ++retained_graphics_replay_.bridge_submissions;
  graphics_submission_used_ = true;
}

void GfxRuntime::invalidate_graphics_command_replay_locked(
    DeviceAllocation image) {
  if (image != kDeviceNullAllocation) {
    const bool references_image = std::find(
        retained_graphics_replay_.images.begin(),
        retained_graphics_replay_.images.end(), image) !=
                                  retained_graphics_replay_.images.end();
    if (!references_image) {
      return;
    }
  }
  if (retained_graphics_replay_.key.empty() &&
      std::none_of(retained_graphics_replay_.slots.begin(),
                   retained_graphics_replay_.slots.end(),
                   [](const auto &slot) { return bool(slot.command_list); }) &&
      !retained_graphics_replay_.prewarmed) {
    return;
  }
  for (auto &slot : retained_graphics_replay_.slots) {
    slot.command_list.reset();
    slot.completion.reset();
  }
  retained_graphics_replay_.next_slot = 0;
  retained_graphics_replay_.key.clear();
  retained_graphics_replay_.images.clear();
  retained_graphics_replay_.prewarmed = false;
  ++retained_graphics_replay_.invalidations;
}

void GfxRuntime::invalidate_graphics_command_replay() {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  invalidate_graphics_command_replay_locked();
}

std::unordered_map<std::string, std::uint64_t>
GfxRuntime::debug_graphics_command_replay_stats() const {
  std::lock_guard<std::recursive_mutex> lock(host_api_mutex_);
  const std::uint64_t slots = std::count_if(
      retained_graphics_replay_.slots.begin(),
      retained_graphics_replay_.slots.end(),
      [](const auto &slot) { return bool(slot.command_list); });
  const std::uint64_t inflight = std::count_if(
      retained_graphics_replay_.slots.begin(),
      retained_graphics_replay_.slots.end(), [](const auto &slot) {
        return slot.command_list && slot.completion &&
               !slot.completion->is_ready();
      });
  return {
      {"retained_replay_attempts", retained_graphics_replay_.attempts},
      {"retained_replay_prewarms", retained_graphics_replay_.prewarms},
      {"retained_replay_records", retained_graphics_replay_.records},
      {"retained_replay_replays", retained_graphics_replay_.replays},
      {"retained_replay_fallbacks", retained_graphics_replay_.fallbacks},
      {"retained_replay_busy_fallbacks",
       retained_graphics_replay_.busy_fallbacks},
      {"retained_replay_binding_misses",
       retained_graphics_replay_.binding_misses},
      {"retained_replay_layout_misses",
       retained_graphics_replay_.layout_misses},
      {"retained_replay_graphics_submissions",
       retained_graphics_replay_.graphics_submissions},
      {"retained_replay_bridge_submissions",
       retained_graphics_replay_.bridge_submissions},
      {"retained_replay_bridge_failures",
       retained_graphics_replay_.bridge_failures},
      {"retained_replay_submit_failures",
       retained_graphics_replay_.submit_failures},
      {"retained_replay_invalidations",
       retained_graphics_replay_.invalidations},
      {"retained_replay_slots", slots},
      {"retained_replay_slot_capacity",
       RetainedGraphicsCommandReplay::kSlotCapacity},
      {"retained_replay_peak_slots", retained_graphics_replay_.peak_slots},
      {"retained_replay_inflight_slots", inflight},
      {"retained_replay_last_path", retained_graphics_replay_.last_path},
  };
}

GfxRuntime::RegisterParams run_codegen(
    Kernel *kernel,
    Arch arch,
    const DeviceCapabilityConfig &caps,
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    const CompileConfig &compile_config) {
  const auto id = Program::get_kernel_id();
  const auto taichi_kernel_name(fmt::format("{}_k{:04d}_vk", kernel->name, id));
  TI_TRACE("VK codegen for Taichi kernel={}", taichi_kernel_name);
  spirv::KernelCodegen::Params params;
  params.ti_kernel_name = taichi_kernel_name;
  params.kernel = kernel;
  params.ir_root = kernel->ir.get();
  params.compiled_structs = compiled_structs;
  params.arch = arch;
  params.caps = caps;
  params.enable_spv_opt = compile_config.external_optimization_level > 0;
  params.vulkan_listgen_reuse = compile_config.vulkan_listgen_reuse;
  spirv::KernelCodegen codegen(params);
  GfxRuntime::RegisterParams res;
  codegen.run(res.kernel_attribs, res.task_spirv_source_codes);
  res.num_snode_trees = compiled_structs.size();
  return res;
}

std::pair<const lang::StructType *, size_t>
GfxRuntime::get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                             const std::string &layout) {
  auto [new_ty, size, align] =
      get_struct_type_with_data_layout_impl(old_ty, layout);
  return {new_ty, size};
}

std::tuple<const lang::StructType *, size_t, size_t>
GfxRuntime::get_struct_type_with_data_layout_impl(
    const lang::StructType *old_ty,
    const std::string &layout) {
  TI_TRACE("get_struct_type_with_data_layout: {}", layout);
  TI_ASSERT(layout.size() == 2);
  auto is_430 = layout[0] == '4';
  auto has_buffer_ptr = layout[1] == 'b';
  auto members = old_ty->elements();
  size_t bytes = 0;
  size_t align = 0;
  for (int i = 0; i < members.size(); i++) {
    auto &member = members[i];
    size_t member_align;
    size_t member_size;
    if (auto struct_type = member.type->cast<lang::StructType>()) {
      auto [new_ty, size, member_align_] =
          get_struct_type_with_data_layout_impl(struct_type, layout);
      members[i].type = new_ty;
      member_align = member_align_;
      member_size = size;
    } else if (auto tensor_type = member.type->cast<lang::TensorType>()) {
      size_t element_size = data_type_size_gfx(tensor_type->get_element_type());
      size_t num_elements = tensor_type->get_num_elements();
      if (!is_430) {
        if (num_elements == 2) {
          member_align = element_size * 2;
        } else {
          member_align = element_size * 4;
        }
        member_size = member_align;
      } else {
        member_align = element_size;
        member_size = tensor_type->get_num_elements() * element_size;
      }
    } else if (member.type->is<PointerType>()) {
      if (has_buffer_ptr) {
        member_size = sizeof(uint64_t);
        member_align = member_size;
      } else {
        // Use u32 as placeholder
        member_size = sizeof(uint32_t);
        member_align = member_size;
      }
    } else {
      TI_ASSERT(member.type->is<PrimitiveType>());
      member_size = data_type_size_gfx(member.type);
      member_align = member_size;
    }
    bytes = align_up(bytes, member_align);
    members[i].offset = bytes;
    bytes += member_size;
    align = std::max(align, member_align);
  }

  if (!is_430) {
    align = align_up(align, sizeof(float) * 4);
    bytes = align_up(bytes, 4 * sizeof(float));
  }
  TI_TRACE("  total_bytes={}", bytes);
  return {TypeFactory::get_instance()
              .get_struct_type(members, layout)
              ->as<lang::StructType>(),
          bytes, align};
}

}  // namespace gfx
}  // namespace taichi::lang
