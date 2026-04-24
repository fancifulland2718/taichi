#include "taichi/compilation_manager/inproc_disk_mirror.h"

#include <cstdlib>

namespace taichi::lang {

namespace {

// All mutable state lives behind function-local statics so there is no
// ordering dependency on global ctor/dtor across translation units.

std::mutex &mirror_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::string> &mirror_store() {
  static std::unordered_map<std::string, std::string> s;
  return s;
}

std::list<std::string> &mirror_fifo() {
  static std::list<std::string> l;
  return l;
}

std::size_t &mirror_total_bytes() {
  static std::size_t t = 0;
  return t;
}

std::size_t &mirror_hits() {
  static std::size_t h = 0;
  return h;
}

std::size_t &mirror_misses() {
  static std::size_t m = 0;
  return m;
}

// Read the cap once per process. 0 disables the mirror entirely.
std::size_t mirror_cap_bytes() {
  // Explicit return type is required: on Windows x64 std::size_t is
  // unsigned long long, but on Linux x86_64 it is unsigned long, while
  // the literal 1024ull is unsigned long long. Without the trailing
  // -> std::size_t, gcc deduces inconsistent types across the early
  // return (`std::size_t{0}`) and the multiplication results and fails
  // with "inconsistent types deduced for lambda return type".
  static const std::size_t cached = []() -> std::size_t {
    constexpr std::size_t kMB = static_cast<std::size_t>(1024) * 1024;
    const char *v = std::getenv("TI_INPROC_DISK_MIRROR_MB");
    if (v != nullptr) {
      try {
        long n = std::stol(v);
        if (n <= 0) {
          return 0;
        }
        return static_cast<std::size_t>(n) * kMB;
      } catch (...) {
        // Fall through to default on malformed input.
      }
    }
    return static_cast<std::size_t>(256) * kMB;
  }();
  return cached;
}

}  // namespace

std::optional<std::string> InprocDiskMirror::get(
    const std::string &kernel_key) {
  std::lock_guard<std::mutex> lk(mirror_mutex());
  auto it = mirror_store().find(kernel_key);
  if (it == mirror_store().end()) {
    ++mirror_misses();
    return std::nullopt;
  }
  ++mirror_hits();
  return it->second;  // copy out so we can drop the lock at return
}

void InprocDiskMirror::put(const std::string &kernel_key,
                           std::string bytes) {
  const std::size_t cap = mirror_cap_bytes();
  if (cap == 0) {
    return;
  }
  if (bytes.size() > cap) {
    return;
  }

  std::lock_guard<std::mutex> lk(mirror_mutex());

  auto existing = mirror_store().find(kernel_key);
  if (existing != mirror_store().end()) {
    // Overwrite in place. Do NOT touch the FIFO position — keeping a key
    // pinned at the same age simplifies reasoning and avoids O(n) list
    // searches on the hot path.
    mirror_total_bytes() -= existing->second.size();
    existing->second = std::move(bytes);
    mirror_total_bytes() += existing->second.size();
    // If the replacement pushed us over the cap, evict oldest entries
    // other than the one we just updated.
    while (mirror_total_bytes() > cap && !mirror_fifo().empty()) {
      const std::string &oldest = mirror_fifo().front();
      if (oldest == kernel_key) {
        // Don't evict the entry we just wrote; stop here. The mirror may
        // remain slightly over cap until the next put/clear, which is
        // acceptable.
        break;
      }
      auto old_it = mirror_store().find(oldest);
      if (old_it != mirror_store().end()) {
        mirror_total_bytes() -= old_it->second.size();
        mirror_store().erase(old_it);
      }
      mirror_fifo().pop_front();
    }
    return;
  }

  // Evict oldest entries until the new value fits.
  while (mirror_total_bytes() + bytes.size() > cap &&
         !mirror_fifo().empty()) {
    const std::string oldest = mirror_fifo().front();
    mirror_fifo().pop_front();
    auto old_it = mirror_store().find(oldest);
    if (old_it != mirror_store().end()) {
      mirror_total_bytes() -= old_it->second.size();
      mirror_store().erase(old_it);
    }
  }
  mirror_total_bytes() += bytes.size();
  mirror_fifo().push_back(kernel_key);
  mirror_store().emplace(kernel_key, std::move(bytes));
}

void InprocDiskMirror::clear() {
  std::lock_guard<std::mutex> lk(mirror_mutex());
  mirror_store().clear();
  mirror_fifo().clear();
  mirror_total_bytes() = 0;
  // Leave hit/miss counters intact — they are cumulative across clears.
}

std::size_t InprocDiskMirror::total_bytes() {
  std::lock_guard<std::mutex> lk(mirror_mutex());
  return mirror_total_bytes();
}

std::size_t InprocDiskMirror::hits() {
  std::lock_guard<std::mutex> lk(mirror_mutex());
  return mirror_hits();
}

std::size_t InprocDiskMirror::misses() {
  std::lock_guard<std::mutex> lk(mirror_mutex());
  return mirror_misses();
}

}  // namespace taichi::lang
