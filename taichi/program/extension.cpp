#include "extension.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace taichi::lang {

namespace {
// Phase 1 (taichi-forge 0.3.x): experimental opt-in for sparse SNode on Vulkan.
// Enabled when env var TI_VULKAN_SPARSE is set to "1". Currently only the
// bitmasked storage layout + activate/deactivate/is_active SPIR-V ops are
// validated; struct_for over bitmasked SNodes still requires SPIR-V listgen
// codegen (Phase 1b) and will fail with a clear error otherwise.
bool vulkan_sparse_experimental_enabled() {
  static const bool enabled = []() {
    const char *v = std::getenv("TI_VULKAN_SPARSE");
    return v != nullptr && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}
}  // namespace

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
