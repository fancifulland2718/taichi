#include "taichi/runtime/gfx/snode_tree_manager.h"

#include "taichi/runtime/gfx/runtime.h"
#if defined(TI_WITH_VULKAN_POINTER)
#include "taichi/runtime/gfx/snode_allocator.h"
#endif

namespace taichi::lang {
namespace gfx {

SNodeTreeManager::SNodeTreeManager(GfxRuntime *rtm) : runtime_(rtm) {
}

void SNodeTreeManager::materialize_snode_tree(
    SNodeTree *tree,
    const taichi::lang::spirv::PointerLayoutPolicy &policy) {
  auto *const root = tree->root();
  CompiledSNodeStructs compiled_structs = compile_snode_structs(*root, policy);
  runtime_->add_root_buffer(compiled_structs.root_size);
#if defined(TI_WITH_VULKAN_POINTER)
  // 路线 B B-1（2026-04-30）：用 contracts 在该 root_buffer 上构造 BumpOnly
  // allocator，与 codegen 端 contract 字节等价。当前 root_buffer 已由
  // add_root_buffer() memset(0)，allocator::clear_all() 暂未被调用。
  const int root_id = static_cast<int>(runtime_->root_buffers_.size()) - 1;
  std::unordered_map<int, std::unique_ptr<DeviceNodeAllocator>>
      allocators_for_tree;
  if (!compiled_structs.pointer_contracts.empty()) {
    DeviceAllocation root_alloc = *runtime_->root_buffers_[root_id];
    for (const auto &[sid, contract] : compiled_structs.pointer_contracts) {
      BumpOnlyDeviceNodeAllocator::Params p;
      p.device = runtime_->device_;
      p.snode_id = sid;
      p.pool_capacity = contract.pool_capacity;
      p.cell_payload_bytes = contract.cell_stride_bytes;
      p.watermark_offset_in_root = contract.watermark_offset_in_root;
      p.pool_data_offset_in_root = contract.pool_data_offset_in_root;
      p.has_freelist = contract.has_freelist;
      p.freelist_head_offset_in_root = contract.freelist_head_offset_in_root;
      p.freelist_links_offset_in_root = contract.freelist_links_offset_in_root;
      p.has_ambient_zone = contract.has_ambient_zone;
      p.ambient_offset_in_root = contract.ambient_offset_in_root;
      p.alloc_protocol = contract.alloc_protocol;
      p.pool_fraction = contract.pool_fraction;
      p.root_buffer_alloc = root_alloc;
      allocators_for_tree.emplace(sid, create_device_node_allocator(p));
    }
  }
  runtime_->node_allocators_[root_id] = std::move(allocators_for_tree);
#endif
  compiled_snode_structs_.push_back(compiled_structs);
}

void SNodeTreeManager::destroy_snode_tree(SNodeTree *snode_tree) {
  int root_id = -1;
  for (int i = 0; i < compiled_snode_structs_.size(); ++i) {
    if (compiled_snode_structs_[i].root == snode_tree->root()) {
      root_id = i;
    }
  }
  if (root_id == -1) {
    TI_ERROR("the tree to be destroyed cannot be found");
  }
  runtime_->root_buffers_[root_id].reset();
#if defined(TI_WITH_VULKAN_POINTER)
  runtime_->node_allocators_.erase(root_id);
#endif
}

size_t SNodeTreeManager::get_field_in_tree_offset(int tree_id,
                                                  const SNode *child) {
  auto &snode_struct = compiled_snode_structs_[tree_id];
  TI_ASSERT_INFO(
      snode_struct.snode_descriptors.find(child->id) !=
              snode_struct.snode_descriptors.end() &&
          snode_struct.snode_descriptors.at(child->id).snode == child,
      "Requested SNode not found in compiled SNodeTree");

  size_t offset = 0;
  for (const SNode *sn = child; sn; sn = sn->parent) {
    offset +=
        snode_struct.snode_descriptors.at(sn->id).mem_offset_in_parent_cell;
  }

  return offset;
}

DevicePtr SNodeTreeManager::get_snode_tree_device_ptr(int tree_id) {
  return runtime_->root_buffers_[tree_id]->get_ptr();
}

}  // namespace gfx
}  // namespace taichi::lang
