# Sparse SNode on Vulkan — User Guide

> Applies to **Taichi Forge 0.3.0**. Vanilla Taichi 1.7.4's Vulkan/SPIRV backend supports only `dense` + `root`. Taichi Forge additionally supports `pointer`, `bitmasked`, and `dynamic` SNodes on Vulkan, with three-backend (cpu / cuda / vulkan) numerical equivalence.
>
> 中文版：[sparse_snode_on_vulkan.zh.md](sparse_snode_on_vulkan.zh.md)

---

## 1. Overview

| SNode type | vanilla 1.7.4 Vulkan | Taichi Forge 0.3.0 Vulkan | LLVM (cpu/cuda) |
|---|---|---|---|
| `dense` | ✅ | ✅ | ✅ |
| `bitmasked` | ❌ | ✅ | ✅ |
| `pointer` | ❌ | ✅ | ✅ |
| `dynamic` | ❌ | ✅ | ✅ |
| `hash` | ❌ | ❌ (see §6) | ❌ (default-disabled in upstream) |
| `quant_array` / `bit_struct` | ❌ | ⚠️ experimental (see §7) | ✅ |

Supported ops on Vulkan: `activate`, `deactivate`, `is_active`, `length`, `append`, `ti.deactivate`, struct-for (`for I in field:`), and `ti.ndrange`-based sparse listgen.

---

## 2. Enabling

No extra switch is required — `ti.init(arch=ti.vulkan)` is enough; all SNode types in the table above are immediately available.

```python
import taichi as ti
ti.init(arch=ti.vulkan)

x = ti.field(ti.f32)
ti.root.pointer(ti.ij, 32).dense(ti.ij, 8).place(x)

@ti.kernel
def fill():
    for i, j in ti.ndrange(256, 256):
        if (i + j) % 17 == 0:
            x[i, j] = i * 0.1 + j * 0.01

fill()
```

Optional build-time switches (default ON; touch only when bisecting a regression):

| CMake option | Default | Purpose |
|---|---|---|
| `TI_VULKAN_POINTER` | ON | Master switch for `pointer` / `bitmasked` on Vulkan. Turning OFF restores vanilla `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_DYNAMIC` | ON | Master switch for `dynamic` on Vulkan. OFF reverts to `TI_NOT_IMPLEMENTED`. |
| `TI_VULKAN_POINTER_POOL_FRACTION` | ON | Activates the `TI_VULKAN_POOL_FRACTION` env var (see §3.2). OFF makes the env var a no-op; capacity is reserved for the worst case. |

### 2.1 Run-time env var

| Env var | Range | Default | Purpose |
|---|---|---|---|
| `TI_VULKAN_POOL_FRACTION` | `(0.0, 1.0]` | `1.0` | Shrinks each pointer SNode's physical cell pool: `capacity = max(num_cells_per_container, round(total_cells × fraction))`. Invalid / `≤ 0` / `> 1` falls back to `1.0`. |

---

## ⚠️ Known correctness / stability issues

> **Two historical semantic deviations from the LLVM cpu/cuda backend, both now fixed:**
> 1. ✅ Fixed (0.3.1): inactive sparse-cell reads return the dtype's zero value.
> 2. ✅ Fixed (0.3.2, 2026-05-02): 3D pointer **full activation** device-lost is resolved by the C-9 deterministic-slot codegen path; details under "Bug 2" below.

### ✅ Fixed bug 1 (0.3.1, 2026-04-30): inactive sparse-cell reads return zero

LLVM cpu / cuda guarantees that reading an inactive sparse cell returns the dtype's zero value (routed to `ambient_val_addr`, a separate read-only zero region). This is the contract that lets SDF lookups, ray-marching, and neighborhood sampling code "just work" on sparse fields.

**0.3.0 behavior (replaced)**: inactive reads were routed to the *real* address of pool slot 0. Once any kernel legitimately activated and wrote to cell 0, every inactive read returned cell-0's actual data — typical symptom: an SDF ray-march reading garbage values > 500 in empty voxels.

**0.3.1 fix**: [`spirv_codegen.cpp` `pointer_lookup_or_activate(do_activate=false)`](../../taichi/codegen/spirv/spirv_codegen.cpp) now redirects inactive reads to an ambient zone appended at the end of the root buffer (per pointer SNode, `cell_stride` bytes of zero memory, memset at startup, never written by any kernel). All three backends now byte-equivalently return 0 for inactive reads. Acceptance test: [tests/p4/g10_inactive_read_zero.py](../../tests/p4/g10_inactive_read_zero.py). The CMake option `TI_VULKAN_POINTER_AMBIENT_ZONE` (default ON) gates the new path; turning it OFF restores 0.3.0 slot-0 fallback.

**Remaining limitation**: inactive **writes** (out-of-capacity OOC) keep the documented silent-loss slot-0 path described in §3.1; this fix only changes inactive-read semantics.

### ✅ Fixed bug 2 (0.3.2, 2026-05-02): 3D pointer **full activation** device-lost

**Historical trigger conditions (failing any one was already safe; now all are moot)**:

1. The SNode tree contains a 3D pointer level (`ti.root.pointer(ti.ijk, ...)` or `pointer.pointer.dense(ti.ijk, ...)`);
2. **Within a single frame** every cell of the pointer container is first-written (first write implies activate);
3. Followed by a listgen-dependent kernel: struct-for (`for i,j,k in field:`) / `to_numpy()` / another kernel reading that field / a second full ndrange scan;
4. ≥64 threads concurrently first-write to the same pointer cell.

**Historical symptom**: the write kernel completed (`atomic_add` counter was correct) but the next dispatch hit `RHI Error: (-4) vkQueueSubmit failed` / on Windows the process aborted with code `0xC0000409`.

**0.3.2 fix (C-9 deterministic-slot codegen)**: after observing that the default `pool_capacity == total_num_cells_from_root` (i.e. always equals the number of outer cells), pointer activation no longer needs an atomic allocator — every outer cell gets a statically assigned unique pool slot `new_slot = idx_u32 + 1 ∈ [1, capacity]`. The SPIR-V emission collapses to a single `OpAtomicCompareExchange(slot, 0, new_slot)`, replacing the previous four-instruction chain (`CAS-marker + atomicIAdd(watermark) + atomicStore + structured spin-loop`). All threads racing on the same outer cell compute the same `new_slot`, so the spin-loop disappears entirely along with the warp-lockstep deadlock.

For full analysis and the regression matrix see `compile_doc/SNode_Vulkan_规划.md` §14. Acceptance test: [tests/p4/g10p1_user_repro.py](../../tests/p4/g10p1_user_repro.py) (5 consecutive runs Vulkan rc=0, cpu/cuda/vulkan sums equivalent within ±1e-5 single-precision tolerance).

**Default-on**: CompileConfig `vulkan_pointer_deterministic_slot` (default **`True`**), automatically engaged whenever `ti.init(arch=ti.vulkan, vulkan_sparse_experimental=True)`. The codegen falls back to the legacy CAS-marker path (byte-equivalent to 0.3.1) in three cases:
- `vk_max_active` hint shrinks capacity below `worst_capacity`;
- `vulkan_pointer_max_chunks > 1` engages the chunked allocator;
- Multi-instance SNodes (`num_cells_per_container != total_num_cells_from_root`, e.g. certain deeply nested SNode trees).

**Manual opt-out**: `ti.init(vulkan_pointer_deterministic_slot=False)` restores the 0.3.1 behavior byte-for-byte.

### Current recommendation for Vulkan sparse SNodes

- ✅ Universally suitable: MPM / SPH / brick rendering atomic_add scatter; small-active-set + dense compute; nested pointer trees (e.g. OpenVDB-like B+ trees); 3D pointer **full activation** "fill-then-reduce" workloads (**newly supported in 0.3.2**).
- ✅ Brick-scatter is still the recommended idiom for performance (better locality), but no longer required to avoid Bug 2.

---

## 3. Important limitations (read before use)

### 3.1 Pointer / dynamic capacity is **statically sized at compile time**

This is the biggest semantic difference from the LLVM backends. Vanilla LLVM treats each cell as a chunk on `node_allocators`, growing the pool on demand at run time. Vulkan has **no device-side dynamic allocator**, so every cell is reserved up front in the root buffer.

Consequences:

- **Activates beyond the static capacity are silently dropped (silent inactive).** This is enforced by the `cap_v` guard. The kernel does not crash, does not write out-of-bounds, but the write **does not take effect**.
- **There is no run-time error.** Diagnose by reading back `ti.length()` or `ti.is_active()`.

Example:

```python
# pointer(N=32) followed by dense(M=8) → physical capacity = 32 cells.
# Activating more than 32 distinct values of `i` in one frame drops the excess.
ptr = ti.root.pointer(ti.i, 32)
blk = ptr.dense(ti.j, 8)
blk.place(x)
```

### 3.2 Shrinking the pool with `TI_VULKAN_POOL_FRACTION`

If you **know** the steady-state working set is much smaller than the worst case (e.g. sparsity < 25%), shrinking the pool reduces both root-buffer size and GPU memory:

```bash
# Linux / macOS
export TI_VULKAN_POOL_FRACTION=0.25
python your_app.py

# Windows PowerShell
$env:TI_VULKAN_POOL_FRACTION = '0.25'
python your_app.py
```

**Use it when**:

- The configured `N` is far larger than the actual peak active count (e.g. `pointer(ti.i, 4096)` but only ~200 active per frame);
- Combined with the deactivate-freelist (always on) — `ti.deactivate` returns cells to the freelist;
- Memory- or shipping-size-sensitive deployments.

**Do not use it when**:

- You don't know the peak;
- Debugging or development;
- The workload activates the entire set before deactivating any (peak == total).

**Safe degradation**: activates beyond the shrunk capacity also fall under the silent-inactive guard (same mechanism as §3.1).

### 3.3 Dynamic SNode protocol differs from LLVM

The Vulkan `dynamic` uses a **flat-array + length-suffix** protocol:

- Container layout: `[data: cell_stride × N][length: u32]`;
- `ti.append(field, [i], val)` = `OpAtomicIAdd(length, 1)` + cell write;
- `ti.length(field, [i])` = `OpAtomicLoad(length)`;
- `ti.deactivate(dynamic_node, [i])` = `OpAtomicStore(length, 0)`;
- No chunk list. Total capacity equals the static `N`.
- Appends past `N` are silently dropped under the same `cap_v` guard.

Numerical results match LLVM exactly across the regression suite.

### 3.4 SPIR-V warp-lockstep limitation

Any "spin until winner finishes the slot" protocol (race-to-activate on `pointer` / `dynamic`) is sensitive to the SPIR-V `OpLoopMerge` + GPU warp-lockstep behaviour. The implementation has been verified stable on NVIDIA / AMD / Intel iGPU. Please file an issue with GPU model + driver version if you hit a hang on a new platform.

### 3.5 Not supported

- `hash` SNode — see §6.
- `quant_array` / `bit_struct` — see §7.
- Cross-tree `pointer` (multiple SNode trees referencing each other) — same as LLVM.
- `ti.deactivate` on a `dense` directly under `root` — same as LLVM (`dense` does not support deactivate).

### 3.6 Listgen iteration order is not a contract

The LLVM backend traverses active cells in struct-for (`for I in field:`) following the SNode tree topology. The Vulkan backend only guarantees that **every currently-active cell is visited**; the visit order is unspecified. If your reduce kernel depends on a deterministic iteration order (e.g. `for i in field: result[0] += i.id`), the intermediate accumulation path may differ between backends and produce **non-byte-equal floating-point results** (final values still agree within 1e-5).

Workaround: use `ti.atomic_add`, or sort before reducing.

### 3.7 Declaring an explicit pointer capacity: `vk_max_active`

Useful for nested pointer trees and large-`N` low-sparsity workloads. Given `pointer(ti.ij, N)`, the default behavior reserves `N²` cells worst-case in the root buffer; if you know the steady-state activate count is much lower, the hint shrinks the pool:

```python
# Vulkan only: cap this pointer's physical pool to 1024 cells (instead of the worst case)
blk = ti.root.pointer(ti.ij, 1024, vk_max_active=1024)
blk.dense(ti.ij, 8).place(x)
```

Rules:

- The kwarg is honored only on the Vulkan backend; other backends ignore it (with a warn-once on first use).
- 4-tier fallback priority: `vk_max_active` (kwarg) > `vulkan_pointer_pool_fraction` (`ti.init` kwarg) > `TI_VULKAN_POOL_FRACTION` (env var) > worst-case reservation.
- Floor: the value is clamped up to `num_cells_per_container` so the deactivate freelist can hold at least one cell.
- Out-of-capacity: silent inactive (same as §3.1), no exception.
- Nested pointers: each level can be hinted independently — `outer = ti.root.pointer(ti.ij, OUTER, vk_max_active=N1); inner = outer.pointer(ti.ij, MID, vk_max_active=N2)`.

---

## 4. Verification matrix

The fork is regression-tested for cpu / cuda / vulkan three-backend numerical equivalence:

| Test | Coverage |
|---|---|
| `tests/p4/vulkan_pointer_smoke.py` | basic activate / lookup |
| `tests/p4/vulkan_pointer_race.py` | multi-thread race-to-activate |
| `tests/p4/vulkan_pointer_recycle.py` | freelist recycle, 32 cycles |
| `tests/p4/vulkan_pointer_listgen.py` | struct-for |
| `tests/p4/vulkan_pointer_deactivate_all.py` | bulk deactivate + revive |
| `tests/p4/vulkan_pointer_ported.py` | vanilla pointer test ported to three backends |
| `tests/p4/vulkan_bitmasked_ported.py` | full bitmasked suite |
| `tests/p4/vulkan_dynamic_basic.py` | dynamic basic / cycle / is_active |
| `tests/p4/g2_pool_fraction.py` | `TI_VULKAN_POOL_FRACTION=0.25` three-backend equivalence |
| `tests/p4/g4_probe.py` | dynamic flat-array protocol |
| `tests/p4/g8_cache_compat.py` | offline cache fresh / hit / corrupt-recover |
| `tests/p4/c1_jit_capacity.py` | `vk_max_active` 4-tier fallback capacity resolution |
| `tests/p4/c1_nested_pointer_pointer_dense.py` | nested `pointer.pointer.dense` three-backend equivalence (with hint) |
| `tests/p4/g10p1_user_repro.py` | Bug 2 full-activation device-lost regression guard (PASS since 0.3.2) |

Run (PowerShell):

```powershell
$env:PYTHONPATH = 'D:\taichi\python'
python tests\p4\vulkan_pointer_smoke.py
```

---

## 5. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Writes silently lost; `ti.length()` < expected | Beyond static capacity (§3.1 / §3.3) | Increase `N`, or reduce per-frame concurrent activates. |
| Some writes lost after setting `TI_VULKAN_POOL_FRACTION=0.5` | Shrunk capacity insufficient (§3.2) | Raise the fraction, or unset to restore `1.0`. |
| `RuntimeError: hash not yet supported` | Using `ti.root.hash(...)` | Disabled in vanilla and the fork — see §6. |
| Vulkan first launch slow, second launch fast | Offline cache cold compile | Expected; subsequent runs hit the cache. |
| `~/.cache/taichi/` `ticache.tcb` corrupted on next launch | Built-in fallback recompile | Forge 0.3.0+ handles `kVersionNotMatched` / `kCorrupted` automatically; no exception. |
| Hang / device lost in race tests | Warp-lockstep (§3.4) | Open an issue with GPU model + driver version. |

---

## 6. About `hash` SNode

`hash` is **default-disabled in both vanilla taichi 1.7.4 and this fork** — `ti.root.hash(...)` raises `RuntimeError("hash not yet supported")`, an upstream condition since at least 2021.

### 6.1 Real-time physics & rendering verdict

We surveyed the canonical real-time GPU workloads to decide whether shipping `hash` SNode on Vulkan would unlock anything new. Conclusion: **no real-time physics or rendering pipeline depends on it**, so further investment is not planned.

| Workload | Typical sparse pattern | Idiomatic Forge replacement |
|---|---|---|
| Rigid-body broadphase (PBD / Bullet-style spatial hash) | Bounded world AABB → fixed-size hash bucket array | `dense` bucket + atomic counter, with a user `hash21` over `floor(x / cell_size)` |
| MPM / MLS-MPM / SPH / PBF | Bounded simulation grid, sparse activation | `pointer.bitmasked.dense` (the upstream MPM-128 / MLS-MPM pattern) |
| FEM / volumetric meshes | Adaptive but bounded | `pointer.dense` tree |
| Cloth / hair / Eulerian fluid | Dense or low-sparsity | `dense` / `bitmasked.dense` |
| OpenVDB-style SDF / voxel cone tracing | B+-tree of bricks | `pointer.pointer.dense` (Forge supports it on Vulkan) |
| Instant-NGP / NeRF hash-grid encoding | Fixed-size hash table with explicit collision | `dense` of size `T` (commonly `2^14`–`2^19`) + user-level coordinate hash |
| Volumetric GI radiance probes | Dense grid | `dense` |

Why `hash` SNode is **not** the right tool for these:

- All of them have a **bounded coordinate space** (the simulation domain or screen-space frustum), so the unbounded-coordinate property of `hash` is unused.
- Vulkan has no device-side dynamic allocator; a faithful `hash` SNode would have to choose between (a) statically reserving the worst-case bucket array — functionally identical to a user-level hash on `dense`, or (b) chunked re-hashing — a feature category this fork has explicitly de-scoped.
- Race conditions on GPU hash insertion are non-trivial (linear-probe + CAS + warp lockstep) and overlap with the `pointer` race-to-activate work that already lives in `pointer` SNode.

### 6.2 Recommended substitutes

1. `ti.root.pointer(ti.ij, N).dense(ti.ij, M).place(...)` plus `TI_VULKAN_POOL_FRACTION=0.05` (§3.2). Covers ~95 % of real sparse scenarios.
2. A user-level hash function (`hash21` / `hash22`-style noise hashes) plus `dense` buckets. Idiomatic for instant-NGP-style encoders and rigid-body broadphase.

---

## 7. About `quant_array` / `bit_struct`

`quant_array` (bit-packed integer / fixed-point fields) and `bit_struct` (compound bit-packed structs) are **LLVM-only in vanilla taichi**. Forge 0.3.0 ships **experimental codegen** on Vulkan:

- The frontend extension gate is opt-in via `ti.init(arch=ti.vulkan, vulkan_quant_experimental=True)` or the env var `TI_VULKAN_QUANT=1` (see [forge_options.en.md](forge_options.en.md) §3). Default is OFF and behaves identically to vanilla 1.7.4 (the quant codegen entry points raise `TI_ERROR` immediately).
- Supported with the gate ON:
  - **`quant_array`**: `QuantInt` / `QuantFixed` member **read + write (including multi-threaded concurrent `ti.atomic_add` via a SPIR-V `OpAtomicCompareExchange` spin RMW)**, byte-equivalent to the cpu / cuda backends.
  - **`bit_struct` / `BitpackedFields(max_num_bits=32 or 64)`**: multi-field same-word RMW write (the `optimize_bit_struct_stores` IR pass coalesces per-field stores into a single `BitStructStoreStmt` under the default `quant_opt_atomic_demotion=ON`; the residual `is_atomic == true` path uses the same CAS-loop), byte-equivalent to cpu / cuda. The MPM-style 11/11/10 quant_fixed packed-position baseline [tests/p4/g9_quant_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_baseline.py) reports `max_err = 9.77e-4 ≤ bound 1.95e-3` on all three backends; the concurrent-atomic-add race baseline [tests/p4/g9_quant_atomic_race.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_atomic_race.py) (N=1024, K=64-way same-word contention) reports `max_err = 3.94e-3 ≤ bound 1.57e-2` on all three.
  - **Atomic add** `ti.atomic_add(quant_field, delta)`: only `AtomicOpType::add` is implemented; physical_type must be i32 or i64 (matches the LLVM `quant_type_atomic` constraint). The SSA return value is the input `delta` (not the old field) — almost no user code consumes the return value of an `atomic_add` on a quant member, so the dequant round-trip is skipped.
- **Not yet supported**:
  - **`QuantFloat` shared-exponent** (`ti.types.quant.float(...)` + `BitpackedFields(shared_exponent=True)`): `TI_NOT_IMPLEMENTED` at the visitor entry. **Explicitly deferred**: this fork has no shared-exponent demo driving it (the original requirement is quant_fixed) and the float bit-manipulation path carries non-trivial cross-driver rounding/denormal risk. Production workloads that need it should keep using the LLVM cpu / cuda backend.
  - **Non-add atomic ops** (`atomic_min` / `max` / `bit_and` / `bit_or` / `bit_xor`) on quantized fields: same restriction as the LLVM backend (a vanilla design decision, not a Vulkan-side regression).
- Unsupported sites raise a clear `TI_NOT_IMPLEMENTED` / `TI_ERROR` pointing at the offending statement, never a silent miscompile.

Workarounds when the gate is OFF or when an unsupported codegen site is hit:

- Use `ti.f16` (half precision) as a poor man's quantization.
- Pack manually with bit operations on `ti.u32` fields.

Regression baselines: [tests/p4/g9_quant_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_baseline.py) (`bit_struct` MPM-style 11/11/10 packing), [tests/p4/g9_quant_array_baseline.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_array_baseline.py) (`quant_array` 8-bit single field), and [tests/p4/g9_quant_atomic_race.py](https://github.com/taichi-dev/taichi/blob/master/tests/p4/g9_quant_atomic_race.py) (`atomic_add` multi-thread same-word contention).

---

## 8. Compatibility and versioning

- **API**: every public Python API (`ti.root.pointer/.dense/.bitmasked/.dynamic/.place`, `ti.activate/.deactivate/.is_active/.length/.append`, `ti.root.deactivate_all`, etc.) preserves vanilla 1.7.4 semantics on the LLVM backends. The newly available SNode types on Vulkan only add reach; nothing existing is broken.
- **Offline cache**: cache keys already include the SNode-tree structural hash, so changes such as the pool fraction or the dynamic protocol invalidate the cache automatically.
- **C-API**: sparse SNode root-buffer layouts are exposed through the existing `c_api/include/` headers; AOT artefact format is unchanged.
- **Wheel**: the published binary wheel ships every sparse-SNode capability of this backend (pointer / bitmasked / dynamic / experimental quant) enabled by default. No additional build flags are required.

---

## 9. References

- New compile-time / run-time / architecture / modernization options in this fork: [forge_options.en.md](forge_options.en.md)
- Tests: `tests/p4/vulkan_*.py` and `tests/p4/g*.py` in this repository.
