#pragma once

#include <cstdint>
#include <string>

#include "taichi/rhi/arch.h"

namespace taichi::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

// [P-Compile-2-A] Cache schema version, decoupled from TI_VERSION_*.
//
// Bump this constant whenever the offline cache key algorithm changes in a
// way that should invalidate previously written .tic files written by the
// same Taichi binary version (e.g. adding/removing a CompileConfig field
// from the key, changing the IR hash, adding device-cap fields, ...).
//
// Bumping forces every cache lookup to miss and silently fall back to
// recompile + rewrite. We do NOT bump TI_VERSION_PATCH for this because the
// metadata version check there serves a different purpose (binary-format
// compatibility of the metadata file itself).
//
// Current fork builds always mix this schema tag into kernel cache hashes.
// We intentionally do not preserve key compatibility with old .tic files:
// schema bumps miss and recompile under the new key.
//
// History:
//   1 - initial schema (2026-04, baseline before P-Compile-2-A landed).
//       Hash-equivalent to no schema versioning at all.
//   2 - P-Compile-1 phase 1 (2026-04). First schema bump after the P-Compile-1
//       driver experiments.
//   3 - CS-3.B/C (2026-05). LLVMRuntime listgen-reuse state changed from a
//       scalar dirty epoch to per-SNode arrays and list-version dependencies.
//       Old CUDA LLVM kernels embed LLVMRuntime field offsets, so reusing old
//       .tic files after the runtime layout change can launch kernels with
//       stale offsets. Force a cache miss and silent recompile.
//   4 - CS-3.D (2026-05). LLVM OffloadedTask serialization now carries sparse
//       listgen host-elision metadata (clear/listgen child/parent ids and
//       topology-mutation flag). Old cached kernels do not have the metadata,
//       so force a miss before host-side current-list launch skipping is used.
//   5 - CS-3.F (2026-05). OffloadedTask topology-mutation metadata is refined
//       from a boolean to an optional SNode id so CUDA host-side listgen reuse
//       can invalidate only the mutated list when safe. Old cached kernels lack
//       this id and would fall back to over-conservative/global behavior.
//   6 - VS-3 (2026-05). SPIR-V TaskAttributes serialization now carries
//       sparse listgen host-elision metadata (list child/parent ids and
//       topology-mutation id). Old cached Vulkan kernels lack the metadata,
//       so force a miss before GFX runtime listgen skipping is used.
//   7 - VS-2.3/G-3.1 (2026-05). SPIR-V BufferBind serialization now carries
//       conservative read/write access metadata for deferred dispatch barrier
//       coalescing. Old cached Vulkan kernels lack access bits, so force a
//       miss and recompile before runtime consumes this metadata.
//   8 - G-6 (2026-05). SPIR-V adaptive optimizer config was added to the
//       offline-cache key because it changes the per-task optimizer pass chain
//       and therefore emitted SPIR-V bytes.
//   9 - Flag cleanup (2026-05). Transient schema where use_fused_passes was
//       removed from the cache key after the driver skip path became no-op.
//  10 - Fused-pass removal (2026-05). Physically removes use_fused_passes /
//       fused_pass_verify and their driver skip counters. Users should clear
//       old taichi-forge caches when switching to this schema; schema tagging
//       also prevents old .tic artifacts from being reused silently.
//  11 - D2 compile-tier cleanup (2026-05). tiered_full_simplify and
//       full_simplify_global_iter_cap are public IR-shaping config fields and
//       now participate in the cache key; "full" also normalizes the default
//       global-pass cap to unlimited before cache lookup.
//  12 - Native AD / dense bulk API refresh (2026-05). Invalidate stale Vulkan
//       kernels observed on field-loss + ndarray-grad AD paths after native
//       dense clear/fill became the default boundary path.
//  13 - P-Compile-10 C2 (2026-05). Kernel cache keys use dirty-bit
//       invalidation for supported per-kernel effective-config changes and
//       cache the AST body string after first key generation. Old key
//       compatibility is intentionally dropped.
//  14 - R7.1 (2026-07). CUDA cache keys include the resolved LLVM NVPTX
//       compute target because it changes emitted PTX. This prevents a newer
//       device from reusing an artifact compiled for an older fallback target.
//  15 - DF3 (2026-07). Compiled-kernel metadata records the generation-safe
//       SNodeTree dependencies consumed by Graph lifecycle validation. Older
//       artifacts do not contain that metadata and must not be reused.
//  16 - S2 sparse-memory modernization (2026-07). LLVM ListManager replaces
//       its embedded 128K chunk-pointer table with inline pointers plus
//       grow-on-demand directory pages. Cached LLVM kernels embed the old
//       ListManager field offsets, so they must recompile against the new ABI.
//  17 - S2 adaptive element-list chunks (2026-07). Field cache metadata now
//       records each SNode parent id, and LLVMRuntime expands its recycled-list
//       table into bounded chunk-size classes. Old metadata and embedded
//       LLVMRuntime offsets are incompatible with this layout.
//  18 - S2 direct ambient allocation (2026-07). NodeManager records
//       deterministic-slot capacity/current/peak counters so sparse allocator
//       telemetry no longer relies on an ambient element in data_list. Cached
//       CUDA kernels embed NodeManager offsets and must recompile.
//  19 - S2 traversal-list capacity budgeting (2026-07). CUDA deterministic
//       pointer lowering now requires an auto-sized per-SNode dedicated pool;
//       monolithic and explicit fixed-pool configurations use NodeManager.
//       Old cached kernels may contain the unsafe unconditional fast path.
//  20 - S3 CPU listgen work attribution (2026-07). CPU LLVMRuntime appends
//       debug task-local scanned/emitted counters, and listgen helpers branch
//       to counter-enabled implementations. Recompile cached CPU listgen tasks
//       so private telemetry observes the same runtime ABI and behavior.
//  21 - S3 CPU parallel listgen (2026-07). Generic nonroot listgen can call the
//       host ThreadPool for deterministic count/prefix/fill and LLVMRuntime
//       appends a Program-shared offsets workspace and execution strategy.
//       Cached CPU listgen tasks must include the new gated runtime path.
//  22 - S3 CPU stable-topology list reuse (2026-07). CPU StructMeta now enables
//       the existing exact dirty-epoch/parent-version contract, and LLVMRuntime
//       appends a task-local reuse signal for debug attribution. Cached CPU
//       kernels must recompile with listgen reuse enabled.
//  23 - E4 fixed sparse linear systems (2026-07). Native sparse assembly and
//       solver lowering gained device-resident execution paths and metadata.
//  24 - Program-global SNode capacity (2026-07). LLVMRuntime expands every
//       SNode-id-indexed table from 1024 to 4096 entries. Cached LLVM kernels
//       embed runtime field offsets and are ABI-incompatible with this layout.
//  25 - Tree-local LLVM SNode runtime state (2026-08). Field cache metadata
//       records deterministic tree-local node ids and cached kernels use the
//       tree directory instead of Program-global fixed-capacity arrays.
//  26 - LLVM AOT kernel dependency metadata (2026-08). Kernel cache payloads
//       retain used SNodeTree ids and Graph metadata so CUDA AOT launchers bind
//       the compact roots expected by newly compiled kernels.
//  27 - Vulkan kernel acceleration-structure bindings (2026-08). SPIR-V
//       TaskAttributes now carry descriptor bindings for kernel-inline ray
//       queries. Older cache payloads cannot retain or bind the TLAS safely.
//  28 - Forge optimization-input audit (2026-08). Cache keys now cover every
//       active IR/backend-shaping CompileConfig switch that was previously
//       omitted, including global-value caching, CPU range lowering, quant
//       lowering, SPIR-V loop unrolling, resolved GPU block limits, and the
//       complete CUDA deterministic sparse-pool gate.
//  29 - Forge per-kernel thread-local variants (2026-08). LLVM cached task
//       payloads now retain the effective TLS byte count, and task launch
//       identities can explicitly enable or disable thread-local lowering.
//  30 - Forge CUDA artifact controls (2026-08). LLVM compiled-kernel payloads
//       now retain the exact per-kernel register cap and NVVM entry metadata
//       can carry min-CTA and max-register tuning directives.
//  31 - JIT Graph acceleration-structure resources (2026-09). Retain AS reads
//       in kernel Graph metadata; old payloads lack this explicit dependency.
//       Existing opaque intrinsic blockers are preserved.
constexpr std::uint32_t kOfflineCacheSchemaVersion = 31;

std::string get_hashed_offline_cache_key_of_snode(const SNode *snode);
std::string get_hashed_offline_cache_key_context(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel);
std::string get_hashed_offline_cache_semantic_key_context(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         Kernel *kernel);
std::string get_hashed_offline_cache_semantic_key(
    const CompileConfig &config,
    const DeviceCapabilityConfig &caps,
    Kernel *kernel);
void gen_offline_cache_key(IRNode *ast, std::ostream *os);

}  // namespace taichi::lang
