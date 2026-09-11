#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "taichi/program/ndarray.h"

namespace taichi::lang {

class PreparedNativeStorage;
class Program;

// Fixed storage metadata, not a second workspace owner or device recording.
// The Program arena still owns radix scratch and pipeline caches. Borrowed
// ndarray wrappers never free the underlying allocation.
class PreparedPrimitiveSort {
 private:
  friend class Program;
  std::shared_ptr<PreparedNativeStorage> storage_;
  std::array<std::unique_ptr<Ndarray>, 2> views_;
  std::array<std::uintptr_t, 2> cuda_pointers_{};
  int count_{0};
  int key_type_{0};
  int value_type_{0};
  int nan_policy_{0};
  std::size_t key_bytes_{0};
  std::size_t value_bytes_{0};
};

}  // namespace taichi::lang
