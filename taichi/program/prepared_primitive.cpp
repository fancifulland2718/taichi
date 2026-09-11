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

}  // namespace taichi::lang
