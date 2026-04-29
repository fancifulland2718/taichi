#include "extension.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace taichi::lang {

namespace {
// Phase 1c-D (taichi-forge 0.3.x): experimental opt-in for sparse SNode on
// Vulkan. Primary source of truth is
// CompileConfig::vulkan_sparse_experimental, propagated here by Program ctor
// via set_vulkan_sparse_experimental() before any extension query runs. The
// legacy env var TI_VULKAN_SPARSE=1 is kept as a compatibility fallback so
// existing scripts and CI invocations continue to work unchanged.
//
// The flag is sticky once turned on for two reasons:
//   1. is_extension_supported() is called during static init from snode.py
//      module loading paths and from compile_to_offloads passes that don't
//      have direct config access; a global "latched" flag matches the
//      original env-var semantics exactly.
//   2. There is at most one Program instance at a time (program.cpp:142
//      TI_ASSERT_INFO num_instances_ == 0), so cross-Program leakage is a
//      non-issue: each ti.init() resets the program before constructing the
//      next one, which re-applies the flag from the new CompileConfig.
bool &vulkan_sparse_flag() {
  static bool flag = []() {
    const char *v = std::getenv("TI_VULKAN_SPARSE");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return flag;
}

bool vulkan_sparse_experimental_enabled() {
  return vulkan_sparse_flag();
}
}  // namespace

void set_vulkan_sparse_experimental(bool enabled) {
  // OR-set: turning the flag on via either the env var or the CompileConfig
  // is sticky for the rest of the process. We never silently turn it off
  // here, because doing so would break cached SNode struct layouts that
  // were already produced with sparse=true.
  if (enabled) {
    vulkan_sparse_flag() = true;
  }
}

bool is_extension_supported(Arch arch, Extension ext) {
  static std::unordered_map<Arch, std::unordered_set<Extension>> arch2ext = {
      {Arch::x64,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::assertion,
        Extension::extfunc, Extension::mesh}},
      {Arch::arm64,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::assertion,
        Extension::mesh}},
      {Arch::cuda,
       {Extension::sparse, Extension::quant, Extension::quant_basic,
        Extension::data64, Extension::adstack, Extension::bls,
        Extension::assertion, Extension::mesh}},
      {Arch::amdgpu, {Extension::assertion}},
      {Arch::metal, {}},
      {Arch::opengl, {Extension::extfunc}},
      {Arch::gles, {}},
      {Arch::vulkan, {}},
      {Arch::dx11, {}},
  };
  // if (with_opengl_extension_data64())
  // arch2ext[Arch::opengl].insert(Extension::data64); // TODO: singleton
  if (arch == Arch::vulkan && ext == Extension::sparse &&
      vulkan_sparse_experimental_enabled()) {
    return true;
  }
  const auto &exts = arch2ext[arch];
  return exts.find(ext) != exts.end();
}

}  // namespace taichi::lang
