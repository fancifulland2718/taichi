#include "taichi/compilation_manager/kernel_compilation_manager.h"

#include <sstream>

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/compilation_manager/inproc_disk_mirror.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

namespace offline_cache {

template <>
struct CacheCleanerUtils<CacheData> {
  using MetadataType = CacheData;
  using KernelMetaData = typename MetadataType::KernelMetadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    write_to_binary_file(
        data, taichi::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const KernelMetaData &kernel_meta) {
    auto fn = fmt::format(KernelCompilationManager::kCacheFilenameFormat,
                          kernel_meta.kernel_key);
    return {fn};
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    // Do nothing
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    std::string ext = filename_extension(name);
    return ext == kTiCacheFilenameExt;
  }
};

}  // namespace offline_cache

KernelCompilationManager::KernelCompilationManager(Config config)
    : config_(std::move(config)) {
  TI_DEBUG("Create KernelCompilationManager with offline_cache_file_path = {}",
           config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
  auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);
  if (path_exists(filepath)) {
    if (lock_with_file(lock_path)) {
      auto _ = make_unlocker(lock_path);
      offline_cache::load_metadata_with_checking(cached_data_, filepath);
    } else {
      TI_WARN(
          "Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
          lock_path, config_.offline_cache_path);
    }
  }
}

const CompiledKernelData &KernelCompilationManager::load_or_compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) {
  auto cache_mode = get_cache_mode(compile_config, kernel_def);
  const auto kernel_key = make_kernel_key(compile_config, caps, kernel_def);

  // P5.a — serialize all cache-map mutation with cache_mutex_. Heavy
  // compile work happens OUTSIDE the lock inside
  // compile_and_cache_kernel().
  std::unique_lock<std::mutex> lock(cache_mutex_);

  // Wait-loop: another worker may already be compiling this exact key. If
  // so, we block on cache_cv_ and re-probe the cache on wake-up.
  while (true) {
    if (const auto *cached = try_load_cached_kernel_locked(
            kernel_def, kernel_key, compile_config.arch, cache_mode)) {
      return *cached;
    }
    if (in_progress_keys_.count(kernel_key) == 0) {
      break;  // We are responsible for compiling this key.
    }
    cache_cv_.wait(lock);
  }

  // P-Compile-2-B4: snapshot whether the offline cache metadata advertises
  // this key, while we still hold the lock. We use this hint to attempt a
  // disk load (outside the lock) BEFORE falling back to a full compile —
  // multiple workers can therefore do their respective disk reads in
  // parallel rather than serializing through `cache_mutex_`.
  bool disk_metadata_hit = false;
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto it = cached_data_.kernels.find(kernel_key);
    // peek above guarantees `compiled_kernel_data` is null here, so a
    // metadata entry implies a disk-only candidate.
    disk_metadata_hit = (it != cached_data_.kernels.end());
  }

  in_progress_keys_.insert(kernel_key);
  // Drop the lock across compile() — the compile step is pure work on
  // kernel_def / compile_config (both const here) plus per-thread LLVM
  // context, so it is safe to run concurrently with other workers.
  lock.unlock();

  std::unique_ptr<CompiledKernelData> compiled;
  bool from_disk = false;
  try {
    if (disk_metadata_hit) {
      compiled = load_ckd(kernel_key, compile_config.arch);
      if (compiled) {
        from_disk = true;
        TI_DEBUG("Create kernel '{}' from disk cache (key='{}', unlocked)",
                 kernel_def.get_name(), kernel_key);
      }
    }
    if (!compiled) {
      compiled = compile_kernel(compile_config, caps, kernel_def);
    }
  } catch (...) {
    lock.lock();
    in_progress_keys_.erase(kernel_key);
    cache_cv_.notify_all();
    throw;
  }

  lock.lock();
  if (from_disk) {
    // Refresh the metadata entry's `last_used_at` so cache pruning sees a
    // recent access. The previous (locked) `try_load_cached_kernel_locked`
    // path used to do this; preserve that behavior under the unlocked
    // disk-load path.
    auto it = cached_data_.kernels.find(kernel_key);
    if (it != cached_data_.kernels.end()) {
      it->second.last_used_at = std::time(nullptr);
      updated_data_.push_back(&it->second);
    }
  }
  const auto &result = install_compiled_kernel_locked(
      kernel_key, cache_mode, std::move(compiled));
  in_progress_keys_.erase(kernel_key);
  cache_cv_.notify_all();
  return result;
}

void KernelCompilationManager::dump() {
  // P5.a — take a consistent snapshot of the in-memory caches before
  // touching disk. `dump()` is typically called at Program shutdown from
  // the main thread, but lock defensively so it stays correct if a worker
  // is still finishing a compile during shutdown.
  std::lock_guard<std::mutex> guard(cache_mutex_);

  if (caching_kernels_.empty()) {
    return;
  }

  taichi::create_directories(config_.offline_cache_path);
  auto filepath = join_path(config_.offline_cache_path, kMetadataFilename);
  auto lock_path = join_path(config_.offline_cache_path, kMetadataLockName);

  if (!lock_with_file(lock_path)) {
    TI_WARN("Lock {} failed. Please run 'ti cache clean -p {}' and try again.",
            lock_path, config_.offline_cache_path);
    caching_kernels_.clear();  // Ignore the caching kernels
    return;
  }

  auto _ = make_unlocker(lock_path);
  CacheData data;
  data.version[0] = TI_VERSION_MAJOR;
  data.version[1] = TI_VERSION_MINOR;
  data.version[2] = TI_VERSION_PATCH;
  auto &kernels = data.kernels;
  // Load old cached data
  offline_cache::load_metadata_with_checking(data, filepath);
  // Update the cached data
  for (const auto *e : updated_data_) {
    auto iter = kernels.find(e->kernel_key);
    if (iter != kernels.end()) {
      iter->second.last_used_at = e->last_used_at;
    }
  }
  // Add new data
  for (auto &[kernel_key, kernel] : caching_kernels_) {
    if (kernel.cache_mode == CacheData::MemAndDiskCache) {
      auto [iter, ok] = kernels.insert({kernel_key, std::move(kernel)});
      TI_ASSERT(!ok || iter->second.size == 0);
    }
  }
  // Clear caching_kernels_
  caching_kernels_.clear();
  // Dump cached CompiledKernelData to disk
  for (auto &[_, k] : kernels) {
    if (k.compiled_kernel_data) {
      auto cache_filename = make_filename(k.kernel_key);
      // Serialize once into an in-memory buffer so we can both write the
      // file and populate the in-process mirror without re-serializing.
      std::ostringstream oss(std::ios::out | std::ios::binary);
      auto err = k.compiled_kernel_data->dump(oss);
      if (err == CompiledKernelData::Err::kNoError) {
        std::string bytes = oss.str();
        std::ofstream fs{cache_filename,
                         std::ios::out | std::ios::binary};
        TI_ASSERT(fs.is_open());
        fs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        TI_ASSERT(!!fs);
        k.size = bytes.size();
        data.size += k.size;
        // P1.b — seed the mirror so the *next* Program within the same
        // process skips disk entirely.
        InprocDiskMirror::put(k.kernel_key, std::move(bytes));
      } else {
        TI_DEBUG("Dump cached CompiledKernelData(kernel_key={}) failed: {}",
                 k.kernel_key, CompiledKernelData::get_err_msg(err));
      }
    }
  }
  // Dump offline cache metadata
  if (!kernels.empty()) {
    write_to_binary_file(data, filepath);
  }
}

void KernelCompilationManager::clean_offline_cache(
    offline_cache::CleanCachePolicy policy,
    int max_bytes,
    double cleaning_factor) const {
  using CacheCleaner = offline_cache::CacheCleaner<CacheData>;
  offline_cache::CacheCleanerConfig config;
  config.path = config_.offline_cache_path;
  config.policy = policy;
  config.cleaning_factor = cleaning_factor;
  config.max_size = max_bytes;
  config.metadata_filename = kMetadataFilename;
  config.debugging_metadata_filename = "";
  config.metadata_lock_name = kMetadataLockName;
  CacheCleaner::run(config);
}

std::string KernelCompilationManager::make_filename(
    const std::string &kernel_key) const {
  return join_path(config_.offline_cache_path,
                   fmt::format(kCacheFilenameFormat, kernel_key));
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::compile_kernel(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto &compiler = *config_.kernel_compiler;
  auto ir = compiler.compile(compile_config, kernel_def);
  auto ckd = compiler.compile(compile_config, caps, kernel_def, *ir);
  TI_ASSERT(ckd->check() == CompiledKernelData::Err::kNoError);
  return ckd;
}

std::string KernelCompilationManager::make_kernel_key(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &caps,
    const Kernel &kernel_def) const {
  auto kernel_key = kernel_def.get_cached_kernel_key();
  if (kernel_key.empty()) {
    if (!kernel_def.ir_is_ast()) {
      kernel_key = kernel_def.get_name();
    } else {  // The kernel key is generated from AST
      kernel_key = get_hashed_offline_cache_key(compile_config, caps,
                                                (Kernel *)&kernel_def);
    }

    kernel_def.set_kernel_key_for_cache(kernel_key);
  }
  return kernel_key;
}

const CompiledKernelData *KernelCompilationManager::try_load_cached_kernel_locked(
    const Kernel &kernel_def,
    const std::string &kernel_key,
    Arch arch,
    CacheData::CacheMode cache_mode) {
  // Precondition: cache_mutex_ held by caller.
  {  // Find in memory-cache (caching_kernels_)
    const auto &kernels = caching_kernels_;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      TI_DEBUG("Create kernel '{}' from in-memory cache (key='{}')",
               kernel_def.get_name(), kernel_key);
      return iter->second.compiled_kernel_data.get();
    }
  }
  // Find in disk-cache (cached_data_)
  if (cache_mode == CacheData::MemAndDiskCache) {
    auto &kernels = cached_data_.kernels;
    auto iter = kernels.find(kernel_key);
    if (iter != kernels.end()) {
      auto &k = iter->second;
      if (k.compiled_kernel_data) {
        TI_DEBUG("Create kernel '{}' from cache (key='{}')",
                 kernel_def.get_name(), kernel_key);
        return k.compiled_kernel_data.get();
      }
      // P-Compile-2-B4: do NOT call `load_ckd` while holding
      // `cache_mutex_`. Disk I/O is moved out into `load_or_compile`'s
      // unlocked region; we just signal "not in mem yet" by returning
      // nullptr, and the caller will probe disk after dropping the lock.
    }
  }
  return nullptr;
}

const CompiledKernelData &
KernelCompilationManager::install_compiled_kernel_locked(
    const std::string &kernel_key,
    CacheData::CacheMode cache_mode,
    std::unique_ptr<CompiledKernelData> compiled) {
  // Precondition: cache_mutex_ held by caller; `kernel_key` is present in
  // `in_progress_keys_` and is NOT yet in caching_kernels_.
  TI_DEBUG_IF(cache_mode == CacheData::MemAndDiskCache,
              "Cache kernel (key='{}')", kernel_key);
  // Another thread may have raced us to the cache while we held
  // in_progress_keys_, but by contract only one thread owns any given
  // kernel_key at a time, so this must still be absent.
  TI_ASSERT(caching_kernels_.find(kernel_key) == caching_kernels_.end());
  KernelCacheData k;
  k.kernel_key = kernel_key;
  k.created_at = k.last_used_at = std::time(nullptr);
  k.compiled_kernel_data = std::move(compiled);
  k.size = 0;  // Populate `size` within the KernelCompilationManager::dump()
  k.cache_mode = cache_mode;
  const auto &kernel_data = (caching_kernels_[kernel_key] = std::move(k));
  return *kernel_data.compiled_kernel_data;
}

std::unique_ptr<CompiledKernelData> KernelCompilationManager::load_ckd(
    const std::string &kernel_key,
    Arch arch) {
  // P1.b — try the in-process bytes mirror first. On a hit we skip the
  // file open + read entirely; on a miss we fall through to disk and
  // populate the mirror with the bytes we just read so that the *next*
  // Program hits it.
  auto deserialize_from = [&](std::istream &is) -> std::unique_ptr<CompiledKernelData> {
    CompiledKernelData::Err err;
    auto ckd = CompiledKernelData::load(is, &err);
    if (err != CompiledKernelData::Err::kNoError) {
      return nullptr;
    }
    if (auto cerr = ckd->check(); cerr != CompiledKernelData::Err::kNoError) {
      return nullptr;
    }
    return ckd;
  };

  if (auto cached_bytes = InprocDiskMirror::get(kernel_key)) {
    std::istringstream iss(*cached_bytes, std::ios::in | std::ios::binary);
    if (auto ckd = deserialize_from(iss)) {
      return ckd;
    }
    // If the mirrored bytes are somehow corrupt, fall through to disk —
    // don't propagate the error as a hard miss. The disk copy (if any)
    // will also re-populate the mirror below.
  }

  const auto filename = make_filename(kernel_key);
  std::ifstream ifs(filename, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    return nullptr;
  }
  // Read the whole file into a string so we can (a) deserialize from it
  // and (b) stash it into the mirror without re-reading from disk.
  std::string bytes{std::istreambuf_iterator<char>(ifs),
                    std::istreambuf_iterator<char>()};
  ifs.close();

  std::istringstream iss(bytes, std::ios::in | std::ios::binary);
  auto ckd = deserialize_from(iss);
  if (ckd == nullptr) {
    TI_DEBUG("Load cache file {} failed or is corrupt", filename);
    return nullptr;
  }
  // Only cache well-formed bytes so that mirror hits are always valid.
  InprocDiskMirror::put(kernel_key, std::move(bytes));
  return ckd;
}

CacheData::CacheMode KernelCompilationManager::get_cache_mode(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) {
  return compile_config.offline_cache && kernel_def.ir_is_ast()
             ? CacheData::MemAndDiskCache
             : CacheData::MemCache;
}

}  // namespace taichi::lang
