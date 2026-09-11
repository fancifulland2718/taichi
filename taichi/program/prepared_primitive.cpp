#include "taichi/program/prepared_primitive.h"

#include <algorithm>
#include <limits>

#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {

int sort_key_type(DataType type) {
  if (type == PrimitiveType::u32) {
    return 0;
  }
  if (type == PrimitiveType::i32) {
    return 1;
  }
  if (type == PrimitiveType::f32) {
    return 2;
  }
  if (type == PrimitiveType::u64) {
    return 3;
  }
  if (type == PrimitiveType::i64) {
    return 4;
  }
  if (type == PrimitiveType::f64) {
    return 5;
  }
  TI_ERROR("Prepared sort keys require i32/u32/f32/i64/u64/f64");
}

int sort_value_type(DataType type) {
  if (type == PrimitiveType::i32) {
    return 0;
  }
  if (type == PrimitiveType::f32) {
    return 1;
  }
  if (type == PrimitiveType::u32) {
    return 2;
  }
  if (type == PrimitiveType::u64) {
    return 3;
  }
  if (type == PrimitiveType::i64) {
    return 4;
  }
  if (type == PrimitiveType::f64) {
    return 5;
  }
  TI_ERROR("Prepared sort payloads require i32/u32/f32/i64/u64/f64");
}

}  // namespace

std::shared_ptr<PreparedPrimitiveSort> Program::prepare_primitive_sort(
    const storage::DenseStorageDescriptor &keys,
    const storage::DenseStorageDescriptor *values,
    int nan_policy) {
  const auto arch = compile_config().arch;
  TI_ERROR_IF(arch != Arch::cuda && arch != Arch::vulkan,
              "Prepared sort requires CUDA or Vulkan");
  TI_ERROR_IF(nan_policy < 0 || nan_policy > 1 ||
                  (arch == Arch::vulkan && nan_policy != 0),
              "Prepared Vulkan sort supports only nan_policy='last'");
  TI_ERROR_IF(keys.index_rank() != 1 || keys.element_rank() != 0,
              "Prepared sort requires one-dimensional scalar keys");
  const auto n = keys.index_extent(0);
  TI_ERROR_IF(n < 0 || n > std::numeric_limits<int>::max(),
              "Prepared sort supports at most INT_MAX items");
  auto plan = std::make_shared<PreparedPrimitiveSort>();
  plan->count_ = static_cast<int>(n);
  plan->key_type_ = sort_key_type(keys.scalar_type());
  plan->key_bytes_ = data_type_size(keys.scalar_type());
  plan->nan_policy_ = nan_policy;
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&keys};
  if (values) {
    TI_ERROR_IF(values->index_rank() != 1 || values->element_rank() != 0 ||
                    values->index_extent(0) != n,
                "Prepared sort requires a matching scalar payload range");
    plan->value_type_ = sort_value_type(values->scalar_type());
    plan->value_bytes_ = data_type_size(values->scalar_type());
    descriptors.push_back(values);
  }
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!cuda_device_radix_sort_available(),
                "CUDA native stable sort is unavailable");
    auto packet = prepare_external_cuda_storage(
        descriptors, std::vector<bool>(descriptors.size(), true));
    plan->storage_ = std::move(packet.storage);
    std::copy(packet.pointers.begin(), packet.pointers.end(),
              plan->cuda_pointers_.begin());
  } else {
    TI_ERROR_IF(!vulkan_radix_sort_available(),
                "Vulkan native stable sort is unavailable");
    plan->storage_ = prepare_native_storage(
        descriptors, std::vector<bool>(descriptors.size(), true));
    if (values) {
      const auto &a = plan->storage_->binding(0);
      const auto &b = plan->storage_->binding(1);
      TI_ERROR_IF(a.pointer.device == b.pointer.device &&
                      a.pointer.alloc_id == b.pointer.alloc_id &&
                      a.pointer.offset < b.pointer.offset + b.bytes &&
                      b.pointer.offset < a.pointer.offset + a.bytes,
                  "Prepared sort key and payload ranges must not overlap");
    }
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
      const auto &binding = plan->storage_->binding(i);
      DeviceAllocation allocation{binding.pointer.device,
                                  binding.pointer.alloc_id};
      plan->views_[i] = std::make_unique<Ndarray>(
          allocation, descriptors[i]->scalar_type(),
          std::vector<int>{plan->count_});
    }
  }
  // No mathematical operation, command submission, scratch allocation or
  // synchronization. First execution may populate the existing backend arena.
  return plan;
}

std::size_t Program::execute_primitive_sort(const PreparedPrimitiveSort &plan) {
  std::size_t workspace_bytes = 0;
  with_prepared_native_storage(*plan.storage_, [&] {
    if (plan.count_ <= 1) {
      return;
    }
    if (compile_config().arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
      workspace_bytes = cuda::driver_stable_radix_sort_strided(
          reinterpret_cast<void *>(plan.cuda_pointers_[0]),
          reinterpret_cast<void *>(plan.cuda_pointers_[1]), plan.count_,
          static_cast<cuda::CudaDriverSortKeyType>(plan.key_type_),
          static_cast<int>(plan.value_bytes_ / sizeof(std::uint32_t)),
          0, plan.key_bytes_, 0, plan.value_bytes_, plan.value_bytes_ != 0,
          plan.nan_policy_, nullptr, &primitive_workspace_arena_);
#else
      TI_NOT_IMPLEMENTED;
#endif
    } else {
      workspace_bytes = vulkan_radix_sort_u32_ndarray(
          plan.views_[0].get(), plan.views_[1].get(), plan.key_type_,
          plan.value_type_, plan.storage_->binding(0).pointer.offset,
          plan.value_bytes_ ? plan.storage_->binding(1).pointer.offset : 0);
    }
  });
  return workspace_bytes;
}

std::shared_ptr<PreparedPrimitiveCompact> Program::prepare_primitive_compact(
    const std::vector<const storage::DenseStorageDescriptor *> &inputs,
    const storage::DenseStorageDescriptor &flags,
    const std::vector<const storage::DenseStorageDescriptor *> &outputs,
    const storage::DenseStorageDescriptor &count) {
  const auto arch = compile_config().arch;
  TI_ERROR_IF(arch != Arch::cuda && arch != Arch::vulkan,
              "Prepared compact requires CUDA or Vulkan");
  TI_ERROR_IF(inputs.empty() || inputs.size() != outputs.size(),
              "Prepared compact requires matching input/output columns");
  TI_ERROR_IF(flags.index_rank() != 1 || flags.element_rank() != 0 ||
                  flags.scalar_type() != PrimitiveType::i32,
              "Prepared compact flags require a 1D scalar i32 range");
  const auto n = flags.index_extent(0);
  TI_ERROR_IF(n <= 0 || n > std::numeric_limits<int>::max(),
              "Prepared compact capacity must be positive and at most INT_MAX");
  TI_ERROR_IF(count.element_rank() != 0 || count.index_rank() > 1 ||
                  count.scalar_type() != PrimitiveType::i32 ||
                  (count.index_rank() == 1 && count.index_extent(0) < 1),
              "Prepared compact count requires scalar i32 or a nonempty i32 vector");
  auto plan = std::make_shared<PreparedPrimitiveCompact>();
  plan->count_ = static_cast<int>(n);
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&flags, &count};
  std::vector<bool> writable{false, true};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto *input = inputs[i];
    const auto *output = outputs[i];
    TI_ERROR_IF(!input || !output || input->index_rank() != 1 ||
                    output->index_rank() != 1 || input->index_extent(0) != n ||
                    output->index_extent(0) < n,
                "Prepared compact needs matching input/flags and sufficient output capacity");
    // The existing raw-word scatter supports scalar, vector and matrix records.
    sort_value_type(input->scalar_type());
    TI_ERROR_IF(input->scalar_type() != output->scalar_type() ||
                    input->element_shape() != output->element_shape() ||
                    input->element_rank() > 2,
                "Prepared compact input/output record types must match");
    std::size_t bytes = data_type_size(input->scalar_type());
    for (auto extent : input->element_shape()) {
      TI_ERROR_IF(extent <= 0 || bytes > std::numeric_limits<int>::max() / extent,
                  "Prepared compact record size is unsupported");
      bytes *= extent;
    }
    TI_ERROR_IF(arch == Arch::vulkan &&
                    static_cast<std::size_t>(n) >
                        std::numeric_limits<std::uint32_t>::max() / (bytes / 4),
                "Prepared Vulkan compact word count exceeds UINT32_MAX");
    plan->column_bytes_.push_back(bytes);
    descriptors.push_back(input);
    descriptors.push_back(output);
    writable.push_back(false);
    writable.push_back(true);
  }
  if (arch == Arch::cuda) {
    TI_ERROR_IF(!cuda_device_compact_available(), "CUDA native compact is unavailable");
    auto packet = prepare_external_cuda_storage(descriptors, writable);
    plan->storage_ = std::move(packet.storage);
    plan->cuda_pointers_ = std::move(packet.pointers);
  } else {
    TI_ERROR_IF(!vulkan_compact_available(), "Vulkan native compact is unavailable");
    plan->storage_ = prepare_native_storage(descriptors, writable);
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
      const auto &a = plan->storage_->binding(i);
      for (std::size_t j = 0; j < i; ++j) {
        const auto &b = plan->storage_->binding(j);
        TI_ERROR_IF((writable[i] || writable[j]) &&
                        a.pointer.device == b.pointer.device &&
                        a.pointer.alloc_id == b.pointer.alloc_id &&
                        a.pointer.offset < b.pointer.offset + b.bytes &&
                        b.pointer.offset < a.pointer.offset + a.bytes,
                    "Prepared compact writable storage ranges must not overlap");
      }
    }
  }
  return plan;
}

std::size_t Program::execute_primitive_compact(const PreparedPrimitiveCompact &plan) {
  std::size_t workspace_bytes = 0;
  // This one existing submission boundary encloses the entire column sequence.
  // Prefix reuse cannot escape the call or cross another producer/arena user.
  with_prepared_native_storage(*plan.storage_, [&] {
    for (std::size_t i = 0; i < plan.column_bytes_.size(); ++i) {
      const auto bytes = plan.column_bytes_[i];
      std::size_t used = 0;
      if (compile_config().arch == Arch::cuda) {
#ifdef TI_WITH_CUDA
        const auto &pointers = plan.cuda_pointers_;
        used = cuda::driver_compact_strided(
            reinterpret_cast<void *>(pointers[2 + i * 2]),
            reinterpret_cast<void *>(pointers[0]),
            reinterpret_cast<void *>(pointers[3 + i * 2]),
            reinterpret_cast<void *>(pointers[1]), plan.count_,
            static_cast<int>(bytes / 4), 0, bytes, 0, sizeof(std::int32_t),
            0, bytes, 0, nullptr, &primitive_workspace_arena_, i != 0);
#else
        TI_NOT_IMPLEMENTED;
#endif
      } else {
        used = vulkan_compact_ranges(
            plan.storage_->binding(2 + i * 2).pointer,
            plan.storage_->binding(0).pointer,
            plan.storage_->binding(3 + i * 2).pointer,
            plan.storage_->binding(1).pointer, bytes, plan.count_, i != 0);
      }
      workspace_bytes = std::max(workspace_bytes, used);
    }
  });
  return workspace_bytes;
}

}  // namespace taichi::lang
