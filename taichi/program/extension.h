#pragma once

#include "taichi/rhi/arch.h"

#include <string>

namespace taichi::lang {

// The Taichi core feature set (dense SNode) should probably be supported by all
// the backends. In addition, each backend can optionally support features in
// the extension set.
//
// The notion of core vs. extension feature set is defined mainly because the
// Metal backend can only support the dense SNodes. We may need to define this
// notion more precisely in the future.

enum class Extension {
#define PER_EXTENSION(x) x,
#include "taichi/inc/extensions.inc.h"

#undef PER_EXTENSION
};

bool is_extension_supported(Arch arch, Extension ext);

// Phase 1c-D (taichi-forge 0.3.x): toggles the experimental sparse-on-Vulkan
// path. Sourced from ti.init(vulkan_sparse_experimental=True) /
// CompileConfig::vulkan_sparse_experimental, with the legacy env var
// TI_VULKAN_SPARSE=1 retained as a compatible fallback. Once enabled the
// flag is sticky for the remainder of the process (matches the previous
// env-var semantics where one ti.init call decided the whole run).
void set_vulkan_sparse_experimental(bool enabled);

}  // namespace taichi::lang
