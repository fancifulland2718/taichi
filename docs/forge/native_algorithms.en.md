# Native Algorithms

This document describes Forge public algorithm entry points that may use native
CPU, CUDA, or Vulkan implementations when the input contract is supported.
Unsupported `method="auto"` cases must fall back to a correct generic path;
unsupported explicit native methods should reject clearly.

For a module-oriented list of all Forge-only API symbols, see
[Forge API reference](forge_api_reference.en.md).

The core native algorithm family first shipped in Forge 0.4.0, with Graph
native replay and device-side checks following in 0.4.1 and 0.4.23. This page
documents the published 0.6.2 portability and safety contract. See the
[release notes](release_notes.en.md) for the introduction version of each
capability.

## Public Entry Points

| Entry point | Purpose |
| --- | --- |
| `ti.algorithms.sort(keys, values=None, ...)` | Stable Forge sort dispatcher. |
| `ti.algorithms.sort_by_key(keys, values, ...)` | Sort keys and permute payload values. |
| `ti.algorithms.parallel_sort(keys, values=None)` | Vanilla-compatible legacy sorter. |
| `ti.algorithms.PrefixSumExecutor(n).run(values)` | Prefix sum / scan. |
| `ti.algorithms.device_prefix(values, extent, ...)` | Compose fixed-capacity primitive inputs through a device-resident valid count. |
| `ti.algorithms.DevicePrefixSequence(capacity)` | Record a fixed-topology valid-prefix pipeline as one logical Graph native node. |
| `ti.algorithms.DevicePrefixWorkspace(max_items)` | Reuse staging and child primitive workspaces across a valid-prefix pipeline. |
| `ti.algorithms.DeviceWorklist(capacity, dtype, telemetry=..., transition_mode=..., unique_key_capacity=...)` | Own reusable front/back storage and staged, direct, or dense-key unique incremental transitions. |
| `ti.algorithms.device_worklist_append(...)` | Atomically append from Taichi scope without a host count readback. |
| `ti.algorithms.device_worklist_append_direct(...)` | Atomically append while publishing the bounded extent directly. |
| `ti.algorithms.device_worklist_append_unique_direct(...)` | Append once per dense integer key in the current transition without clearing the tag table. |
| `ti.algorithms.DeviceWorklistSequence(args)` | Record one worklist transition as a Graph native action. |
| `ti.algorithms.DeterministicScatterReducePlan(indices, num_groups)` | Reuse a fixed scatter topology with stable source-order floating-point reduction. |
| `ti.algorithms.experimental_compact(values, flags, output, count, ...)` | Filter values by flags and write compacted output. |
| `ti.algorithms.experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, ...)` | Encode consecutive integer-key runs entirely on device. |
| `ti.algorithms.experimental_unique(values, output, count, ...)` | Select the first item from every consecutive equal run. |
| `ti.algorithms.experimental_unique_by_key(keys, values, unique_keys, unique_values, count, ...)` | Select the first payload from every consecutive key run. |
| `ti.algorithms.experimental_segmented_reduce(values, layout, output, ...)` | Reduce each reusable dense segment without a host round trip. |
| `ti.algorithms.experimental_segmented_scan(values, layout, output, ...)` | Inclusive/exclusive scan inside each reusable dense segment. |
| `ti.algorithms.experimental_reduce(values, output, op="sum", ...)` | Reduce values into `output[0]`. |
| `ti.algorithms.experimental_histogram(values, bins, ...)` | Histogram integer values into bins. |
| `ti.algorithms.experimental_transform(src, dst, scale=..., bias=..., ...)` | Elementwise affine transform and copy. |
| `ti.algorithms.experimental_gather(src, indices, dst, ...)` | Indexed read. |
| `ti.algorithms.experimental_scatter(src, indices, dst, ...)` | Indexed write. |
| `ti.algorithms.experimental_scatter_add(src, indices, dst, ...)` | Indexed add with backend atomics or staged reduction where supported. |
| `ti.algorithms.experimental_bucket_builder(keys, values, offsets, output, ...)` | Build grouped/bucketed value output. |
| `ti.algorithms.experimental_grouped_reduce(keys, values, output, ...)` | Reduce values by integer group key. |
| `ti.algorithms.count_if(flags, ...)` / `any_if()` / `all_if()` | Launch device-side numeric predicate checks from Python scope. |
| `ti.algorithms.nan_count(values, ...)` / `inf_count()` / `all_finite()` | Count NaN/Inf/non-finite values on the device. |
| `ti.algorithms.index_bounds_check(indices, lower=..., upper=...)` | Count out-of-range indices on the device. |
| `ti.algorithms.max_abs(values, ...)` / `max_abs_delta(values, reference, ...)` | Compute convergence/error metrics as device-side max-abs reductions. |

The `experimental_` prefix means the entry point is Forge public API but may
evolve more conservatively than long-standing vanilla APIs.

## Backend Selection

Most APIs accept `method="auto"`. The auto method chooses a native backend only
when the dtype, shape, layout, and backend capability are known to be supported.
Otherwise it preserves correctness through a generic or host fallback.

Common explicit method families include:

- `cpu_native`
- CUDA driver-only native methods
- deprecated CUDA CUB reference methods in reference-enabled developer builds
- Vulkan native methods where available
- `host_stable` or legacy fallback methods for sort-like operations

Explicit native methods are useful for testing or controlled deployments. They
should not be used as portability promises across all backends.

## CompileIQ boundary

Forge 0.6.3 does not expose algorithm-level CompileIQ search entry points. The
historical reduce-provider and segmented-scan offline searchers remain private
qualification tools for auditing their existing positive and negative
evidence. They are not algorithm API contracts and do not form a new provider
routing layer. Applications should continue to use the ordinary primitives,
explicit `method=`, or their existing `method="auto"` behavior.

The public CompileIQ entry point is Graph-owned:
`ti.graph.compileiq_recipe_search(graph)` searches only complete Graph execution
recipes that Forge has proven legal and can materialize exactly. CompileIQ does
not receive raw provider, block-size, workgroup-shape, PTXAS-flag, or segment-
offset axes. Search and qualification remain offline and do not change
primitive defaults; compile/search build time is diagnostic only and is not an
admission gate.

## Machine-readable capability contract

Since Forge 0.6.0, immutable schema-v1 descriptors are exposed for every
current primitive family:

```python
contract = ti.algorithms.primitive_capability("experimental_reduce")
for operand in contract.operands:
    print(operand.name, operand.dtypes, operand.ranks, operand.layouts)

ti.init(arch=ti.vulkan)
active = ti.algorithms.resolve_primitive_capability("reduce")
for method in active.methods:
    print(method.method, method.program_available)
```

`primitive_capability(name)` and `primitive_capabilities()` are static and may
be called before `ti.init()`. They report role-specific operand dtype, rank,
layout, and storage constraints; backend methods; stability, determinism, and
atomic-order behavior; primal/forward/reverse/explicit-adjoint support; Graph,
AOT, workspace, and fallback contracts. Entry-point names such as
`experimental_reduce` are accepted as aliases for their family.

`resolve_primitive_capability(name)` requires an active Program. It filters
methods to the current CPU/CUDA/Vulkan backend and evaluates the same
side-effect-free provider probes used by dispatch. A true
`program_available` value means that the Program contains the provider. It is
not a promise that an arbitrary dtype/layout request is valid; method entries
remain `input_dependent=True`, and the public operation performs the final
request validation before writing.

The catalog is also the source of truth for public `method=` validation and
native AD policy. This prevents a documented method or adjoint capability from
drifting away from dispatch behavior.

### Automatic differentiation

- Under `ti.ad.Tape()`, `method="auto"` selects a native primal only when the
  concrete input also has a complete registered backward. Otherwise it uses
  the declared kernel fallback; an unsupported explicit native method rejects
  before writing.
- Under `ti.ad.FwdMode()`, transform, reduce-sum, gather, scatter, and
  scatter-add use their differentiable helper-kernel fallback. Their JVPs are
  regression-tested on CPU, CUDA, and Vulkan. Explicit native methods reject
  because native forward launchers are not implemented.
- Scan, grouped-reduce, segmented scan, and segmented reduce currently reject
  `FwdMode` before writing. Segmented-reduce reverse AD is limited to the
  grouped ndarray sum path; serial/dense-field mode rejects.
- Sort, compact, RLE/Unique, histogram, bucket-builder, device checks, and
  device metrics are declared non-differentiable and reject automatic AD
  contexts before writing. Run such preprocessing or diagnostics outside
  Tape/FwdMode.

These rules describe automatic AD only. Native Graph nodes remain primal-only,
and native-node AOT serialization remains unsupported.

### CUDA runtime portability

The standard CUDA primitive providers in current source use the dynamically
loaded CUDA Driver API and Forge-owned kernels. Standard runtime wheels contain
no CUB or CUDART dependency, users do not install a local CUDA Toolkit, and the
project publishes one runtime wheel per operating system rather than one wheel
per CUDA version. `method="auto"` never selects a Toolkit-reference provider.

The explicit `cuda_cub` / `cuda_cub_*` methods are deprecated development
references. They are compiled only by the separate reference workflow, warn
when called, and are absent from standard runtime wheels. Runtime wheels already
published with 0.5.0's bundled-CUDART layout remain loadable for compatibility;
they are not evidence for the new driver-only dependency class.

Driver-only removes the CUDA Runtime library dependency but does not by itself
prove a lower minimum NVIDIA driver. PTX acceptance and every claimed driver
floor still require execution on the target driver. See
[Building wheels](build_wheels.en.md) for the current build boundary and
[Linux revalidation](linux_revalidation.en.md) for outstanding Linux and
older-driver evidence.

### CUDA 0.6.0 historical performance snapshot and current boundary

The table below is the 0.6.0 qualification snapshot, not a measurement of every
later `master` optimization. It is one unified hot-path result from the Windows
development host (RTX 5090, driver 610.62, Python 3.10.11) at 1,048,576 i32
items. Each entry is the per-call median of 30 samples, with 20 submissions per
sample before synchronization. The idle guard found no other Python or GPU
compute process. CUB came only from the non-publishing CUDA 13.2 reference
build; correctness was checked separately against NumPy oracles.

| Primitive | driver-only median | CUB reference median | Relative throughput | Qualification reference | Driver workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| scan | 0.0272 ms | 0.0190 ms | 69.8% | 90% | 4 KiB |
| reduce-sum | 0.0228 ms | 0.0193 ms | 84.6% | 90% | 4 KiB |
| histogram-256 | 0.1243 ms | 0.1215 ms | 97.7% | 90% | 0 |
| stable compact | 0.0279 ms | 0.0228 ms | 81.8% | 80% | 4.00 MiB |
| stable i32 key/value sort | 0.4883 ms | 0.1491 ms | 30.5% | 80% | 28.06 MiB |

Against the qualification references in the table, histogram and compact met
the reference while scan, reduce, and sort did not. Standard wheels still
select the correct, asynchronous, driver-only Forge provider because CUB is not
a release dependency and a host round trip is not a suitable GPU hot-path
default. This is not a claim of CUB parity and is not a cross-device or
cross-driver guarantee.

The paired 0.6.1 release-candidate wheels retain the same 1,024-item tiled
scan, fused tiled compact ranks, and stable hierarchical 4-bit LSD radix
contract, but stop the radix histogram hierarchy as soon as its top level fits
one scan tile. At the table's
1,048,576-item size, that statically reduces a 32-bit sort from 16 to 8
histogram-scan launches and from 8 to 0 histogram uniform-add launches, with no
workspace growth. A separate wheel-to-wheel test used the public 0.6.0 wheels
(`dbc683028`) and paired 0.6.1 release-candidate wheels on the same RTX
5090/610.62 system. Three fresh processes per wheel each ran ten warmups and 100
end-synchronized native sorts. The median of process medians was 0.51245 ms for
0.6.0 and 0.36455 ms for 0.6.1, a 28.9% latency reduction; reported peak
workspace changed from 29,426,176 to 29,425,664 bytes. Thirteen installed-wheel
CUDA dtype/payload and large-hierarchy stability cases passed. This paired
result uses a different synchronization protocol from the historical table and
therefore supplements rather than rewrites that snapshot.

## Data Contracts

- Dense 1D `ti.ndarray` inputs are the primary native algorithm ABI.
- Dense field/SNode paths are supported only where the implementation can prove
  a compatible dense layout or provide a safe staging path.
- `StructNdarray` can be used as opaque payload for order/copy-style
  primitives. Scalar and packed tensor member views are supported by selected
  numeric primitives.
- Sparse, non-contiguous, or complex SNode topologies should not be assumed to
  use native paths.
- Plain `experimental_scatter()` requires unique in-range destination indices.
  CPU native scatter validates this before writing and rejects duplicates;
  use `experimental_scatter_add()` when duplicate targets are intended.
- Floating-point duplicate-target scatter-add can differ by backend atomic
  order. Use it only where that numerical contract is acceptable.

## Device-resident valid-prefix pipelines

`DevicePrefix` lets fixed-capacity primitives share one device-written
`DeviceExtent` without observing the count on the host:

```python
workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
prefix = ti.algorithms.device_prefix(values, extent, workspace=workspace)
active = prefix.compact(flags, compacted, compacted_extent)
active.sort()
active.scan(scanned)
```

The wrapper currently composes stable compact, scan, reduce, stable sort,
consecutive unique/RLE, grouped sum, and bucket building. It supports scalar
1D `i32/u32/i64/u64/f32/f64` storage, subject to each primitive's narrower
contract. Result storage keeps its fixed physical capacity. Only the prefix
below the paired extent is defined; preparing a provider may overwrite the
inactive suffix with a neutral value or sentinel.

This is a composition layer, not a second algorithm provider. It reuses the
normal `method="auto"` selection and a `DevicePrefixWorkspace`, so no count
readback, per-count allocation, or Graph rebuild is needed between operations.
The underlying fixed-capacity provider may still process capacity-sized
scratch. A workspace may be reused serially but not by concurrent submissions.

For Graph execution, `DevicePrefixSequence` records the same prefix operations
over symbolic ndarray arguments and is appended through
`GraphBuilder.append_native()`. This gives one user ticket without host count
observation. Provider selection, workspace topology, and operation routing are
fixed at materialization; replay does not repeat the operation-kind branch.
The current core Prefix providers do not yet expose an enclosing-backend Graph
command recipe. They are therefore a segmented native diagnostic route:
`admission="auto"` rejects them, while explicit admission reports their loose
helper and queue topology instead of presenting the node as backend-recorded.
This does not fuse provider kernels. When a compact result
feeds Vulkan bounded dispatch, create `output_extent.dispatch_state(block_dim)`
and pass it to both compact and `dispatch_bounded()`: the compact scatter then
publishes the indirect packet with its count, removing one preparation
dispatch. CPU/CUDA do not consume this packet; CUDA independently uses its
exact logical range and may select 12.4+ adaptive physical control.

On the current Windows qualification machine, a compact-to-scan chain with a
10% active prefix was 1.05x faster on CPU, 1.32x on CUDA, and 1.90x on Vulkan
than the same chain with an explicit `DeviceExtent.snapshot()` between the two
operations. These are synchronization-elimination measurements, not portable
throughput guarantees. The paired, end-synchronized harness is
`benchmarks/dynamic_workload_bench.py`.

## Device-resident worklists

`DeviceWorklist` adds lifecycle and optional accounting to the valid-prefix
primitives. It owns two fixed-capacity scalar ndarrays, two `DeviceExtent`
objects, reusable primitive workspace, and either the full counter set or a
lean mandatory state. A custom producer appends into the back storage and
publishes it without observing the count on the host:

```python
worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)

@ti.kernel
def produce(values: ti.types.ndarray(dtype=ti.i32, ndim=1),
            extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
            generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
            overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
            limit: ti.i32, requested: ti.i32):
    for i in range(requested):
        ti.algorithms.device_worklist_append(
            values, extent_state, generated, overflow, limit, i
        )

worklist.prepare_next()
produce(*worklist.append_arguments(), requested)
worklist.commit_next()
```

`telemetry=False` omits accepted/rejected/conflict/winner arrays, bindings, and
device writes. Staged transitions retain generated/overflow/generation in 12
bytes. `transition_mode="direct"` further reduces mandatory state to
overflow/generation (8 bytes) and uses `device_worklist_append_direct()` to
publish extent during slot reservation; it is restricted to atomic append,
requires `telemetry=False`, and must not be followed by `finalize_next()`.

For a bounded dense key domain, setting `unique_key_capacity` on a direct
worklist adds one persistent i32 generation tag per key and one scalar epoch.
The producer calls `device_worklist_append_unique_direct()` while visiting the
current active prefix. The tag elects one append for every emitted key without
clearing the tag table, so a producer can generate cells and overlapping halo
neighbors directly into the next frontier. Invalid keys, capacity overflow,
and epoch exhaustion set sticky overflow and make later stages fail closed.
The primitive never scans the dense domain: end-to-end work scales with the
active domain only when the application also removes full-domain copy/select
and initializes retired/default state in its producer or consumer.

When a workload already has a globally ordered boundary/record kernel, that
kernel may call `device_worklist_recycle_direct()` with
`worklist.recycle_arguments()`, then the host calls
`commit_recycled_next()`. This recycles the consumed front and advances the
generation without a separate next-level prepare dispatch. It must run exactly
once after the previous producer and old-front reads have completed; it is not
a substitute for a cross-block barrier inside a producer.
Unique worklists use the separate
`device_worklist_recycle_unique_direct()` / `unique_recycle_arguments()` pair.
That boundary advances both generation and the unique epoch; using the ordinary
recycle ABI is rejected because stale generation tags could otherwise suppress
valid appends. Generation or epoch exhaustion sets sticky overflow instead of
wrapping.

The overflow-free path performs one atomic slot reservation per item. Atomic
append order is unspecified. One producer owns a transition; independent
Graph submissions must be ordered before they write the same worklist. An
overflow clamps the published count to capacity and remains visible in both
the `DeviceExtent` and worklist counters. A forged or mismatched Graph capacity
binding fails closed before writing values. `select(flags)` preserves source
order. `resolve_conflicts(keys, priorities=...,
policy="min_priority", strategy="auto", key_capacity=...,
output_shape="compact_winner_list")` separates the
conflict algorithm from the sort provider. A compact bounded integer domain can
use deterministic `dense_atomic` arbitration; otherwise `radix_grouped` uses
the backend native stable-sort provider. Equal priority is resolved by ordinal
and then source index in both strategies. Dense arbitration rejects
out-of-domain keys with overflow, while radix winner reduction scans each
sorted key run. A distribution dominated by one or a few very long radix runs
has lower parallelism and should be benchmarked separately. Use
`benchmarks/device_worklist_conflict_bench.py` for paired, identical-input
strategy qualification with parity, raw samples/CV, and workspace accounting.
When only per-key ownership is consumed, request
`output_shape="dense_winner_table"` with `dense_atomic` and
`telemetry=False`. The result is a `key_capacity`-sized source-index table with
`0x7fffffff` for empty keys; no compact extent, winner list, scan, or compact
materialization is produced.

Fixed-domain producers can call `resolve_conflicts_from_mask()` with dense key
and active-flag arrays. It preserves original source indices while eliminating
stable candidate compaction and attribute gathering. If no explicit ordinal is
needed, source index is the deterministic tie break and the ordinal pass and
buffer are omitted. This telemetry-free winner-table route is explicit: users
should retain a fused direct-claim kernel when its workload-specific physical
plan is cheaper.

For priority/claim arbitration without custom ordinals, CPU and CUDA pack the
signed priority order and source-index tie break into one u64 atomic-min pass.
Vulkan retains the portable 32-bit multi-pass route. The returned
`arbitration_route` exposes that choice. Packing removes one dispatch but costs
four additional workspace bytes per key, so it does not change the guidance to
benchmark against a workload-specific fused claim.

For Graph replay, create symbolic arguments with `worklist.graph_args(name)`.
Append separate `DeviceWorklistSequence(args).prepare_next()` and
`.finalize_next()` nodes around a staged user producer. A direct transition
normally records only `prepare_next()` plus the direct producer. A host-driven
level loop may instead use the ordered boundary-kernel recycle contract above;
that low-level path is not a `DeviceWorklistSequence` action. A sequence can also
record `select()`, compact-list conflict resolution, or
`resolve_conflict_winner_table()`. Transition helpers are backend-recordable;
provider pipelines without an integrated action remain explicit segmented
routes. Graph-owned staging is allocated before submission;
steady-state replay neither allocates nor reads the count on the host. First
execution may still compile kernels and prepare native provider workspace.
The strategy, provider, and workspace topology are fixed at materialization.
An incremental multi-stage Graph can use `args.transition_arguments(step)` and
record `DeviceWorklistSequence(args).prepare(target="next"|"current")` before
each bounded producer. Consecutive steps alternate the two stable buffers; one
epoch table deduplicates each transition independently, and overflow propagates
to the following stage. After synchronous completion, or after waiting on an
asynchronous ticket, `commit_direct_transitions(steps)` updates only Python-side
front ownership. One worklist remains one completion-ordered workspace lane;
concurrent submissions require independent worklists.
When the worklist is permanently owned by one Graph, call
`worklist.fixed_graph_args(builder, name)` instead. It binds the front/back,
extent, counter, and unique-tag resources privately, so only unrelated user
resources remain in the public replay ABI. The binding is qualified by runtime
generation and the current front parity. Even-transition Graphs retain their
parity; an odd-transition pipeline needs the Graph compiled for the other
parity before its next replay. Wrong-parity replay fails before submission.
Provider-fixed worklists require `workspace_lanes=1`, and asynchronous reuse is
runtime-ordered (with a completion fence when the backend exposes pending work);
use an independent worklist and Graph for real concurrency.
This ownership form does not remove submission-time resource retention and is
not an automatic throughput promise.
On Vulkan, an adjacent recordable prepare and bounded consumer share one
backend Graph region: prepare resets the target and publishes the source
indirect packet in one dispatch. The pair therefore has two physical
dispatches and no loose packet helper.
With full telemetry, `args.observe()` adds all counters to completion-attached
ticket observation, while
`args.decode_observation()` materializes `DeviceWorklistStatistics` after
completion. Lean arguments reject this observation request instead of
materializing hidden telemetry. `execution_report()` is an explicit
synchronized boundary that can
join these counters with a `dispatch_bounded()` snapshot.

On Vulkan, an adjacent recorded `finalize_next()` and
`dispatch_bounded()` now automatically share a Graph-owned indirect packet.
The finalizer publishes count and grid in one dispatch, consecutive matching
consumers reuse the packet, and no public launch-state object or preparation
dispatch is required. An intervening action conservatively disables this
specialization. Passing `worklist.next_extent.dispatch_state(block_dim)` to
both sides remains a compatibility route for explicit packet publication.
CPU and CUDA keep the same source-level composition but do not consume that
packet. CPU uses exact scheduler chunks, while CUDA uses an exact logical
device range on every supported driver and may optionally trim its physical
grid with 12.4+ device updates. Query
`ti.graph.dynamic_work_capabilities()["worklist"]` instead of inferring exact
launch behavior from the common API.

## Deterministic fixed-topology scatter reduction

`DeterministicScatterReducePlan(indices, num_groups)` is an explicit
reproducible alternative to floating-point atomic scatter-add for immutable
connectivity. Construction validates the host-visible integer topology once,
stably groups valid source ordinals by destination, and uploads the permutation
and segment layout. A binding uses one indexed kernel to read changing
contributions in that fixed order and reduce every destination left-to-right on
CPU, CUDA, and Vulkan. Scalar and vector ndarray/root-dense Field bindings are
supported. Negative and out-of-range indices are ignored consistently.

This route is not selected automatically and does not change the atomic API.
It trades serial work within each destination for a stable accumulation order,
but no longer materializes an ordered-value buffer or a separate gather stage.
It is most appropriate for qualification baselines, reproducible fixed-topology
assembly, and modest per-destination valence. `binding.graph_action()` records
one reusable dispatch, and `report()` exposes topology, component shape, zero
ordered/workspace bytes, and the `fused_indexed_serial` route.
Stable order means repeatability under the same backend/build contract; it is
not a cross-backend bit-identity or improved-accuracy promise. Atomic assembly
therefore remains the default performance route, while stable serial is chosen
only when reproducibility is required.

## Consecutive RLE and Unique

Forge provides device-resident consecutive-run primitives:

```python
workspace = ti.algorithms.RunLengthWorkspace(max_items=capacity)
ti.algorithms.experimental_run_length_encode(
    keys,
    unique_keys,
    run_lengths,
    run_count,
    size=active_count,
    workspace=workspace,
)
```

`experimental_unique()` selects the first value from each consecutive equal
run. `experimental_unique_by_key()` also selects the first payload in each
key run. None of these APIs performs an implicit sort or hash-table build.
Sorted input produces global sorted unique output; arbitrary input preserves
consecutive run order. Global first-occurrence unique is not implemented by
the current contract. Unique-by-key accepts StructNdarray raw payloads; dense
MatrixField payloads currently require matching input/output shapes and
`ti.i32` elements.

The current RLE/Unique contract supports `i32/u32/i64/u64` keys. `size=None` consumes the
full fixed capacity; an integer `size` consumes only that active prefix, and
`size=0` is the supported logical-empty representation. Count and lengths
are i32, so capacity is limited to `2^31-1`. Input and output storage must not
alias, and only entries below the device-side count are defined.

This is a fixed-capacity tradeoff: `size` changes semantics, but flags, compact
dispatch, and scratch remain sized to physical capacity so Graph replay does
not rebuild. Capacity-bucket workloads when utilization is persistently low.

The implementation composes one boundary kernel with existing native compact
providers. RLE additionally compacts run starts and launches one length kernel.
It therefore adds no backend ABI or versioned CUDA-library dependency. Minimum
reusable scratch is 4 bytes/item for Unique and 12 bytes/item for RLE, plus the
compact provider's temporary storage. A `RunLengthWorkspace` is reusable but
not concurrently shareable; independent workspaces were stress-tested from two
Python submission threads on CPU, CUDA, and Vulkan.

On the Windows development machine (Ryzen 9 9950X, RTX 5090 driver 610.62),
1,048,576 i32 keys with 262,144 runs measured:

| backend | public RLE | PrimitiveSequence Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 4.85 ms | 4.22 ms | 4.98 ms | 1.03x |
| CUDA | 0.418 ms | 0.456 ms | 12.19 ms | 29.2x |
| Vulkan | 0.643 ms | 0.632 ms | 16.03 ms | 24.9x |

Compilation/warmup was outside timing, workspaces were reused, and no other
Python/GPU compute process was active. These are development measurements, not
cross-driver guarantees. The CUDA Graph delta is about 38 microseconds and is
recorded as general native-node replay overhead; the current implementation
does not add an RLE-specific path for it.

## Reusable Segmented Reduce and Scan

Forge represents fixed-capacity dense topology with a reusable
`SegmentedLayout`:

```python
layout = ti.algorithms.SegmentedLayout.from_offsets(
    np.array([0, 256, 512, 512, 768], np.int32),
    capacity=1024,
)
workspace = ti.algorithms.SegmentedWorkspace(
    max_items=layout.capacity,
    max_segments=layout.num_segments,
)
ti.algorithms.experimental_segmented_reduce(
    values, layout, per_segment_sum, workspace=workspace
)
ti.algorithms.experimental_segmented_scan(
    values, layout, scanned, inclusive=True, workspace=workspace
)
```

Offsets start at zero and are nondecreasing; repeated offsets are empty
segments. Alternatively, `from_segment_ids()` accepts a nondecreasing active
prefix and permits missing IDs. The constructor validates topology on the host
and uploads both offsets and normalized IDs. Passing a Taichi source therefore
synchronizes once at construction. Reusing the layout in direct calls or
`PrimitiveSequence` Graph replay remains device-resident.

The current segmented contract supports scalar 1D plain ndarray and root-dense field
storage with `i32/u32/i64/u64/f32/f64`. Values have exactly layout capacity;
reduce output has exactly one value per segment, while scan output has layout
capacity and only its active prefix is defined. Empty segments reduce to zero.
Scan can be exactly in-place or disjoint. MatrixField, StructNdarray, and sparse
SNode forms remain outside this contract.

Segmented reduce currently implements sum. Ndarray `auto` composes the existing
grouped-reduce provider; dense field and explicit `serial` use left-to-right
segment-local accumulation. Integer sums are exact. Grouped floating sums are
method/order dependent, while serial floating sums preserve the documented
order. Only grouped ndarray sum has reverse AD; FwdMode and serial AD reject
before output changes.

Segmented scan implements inclusive/exclusive sum. Float scan always preserves
segment-local left-to-right order. Integer `auto` is intentionally coarse:
CPU/Vulkan and ordinary short CUDA segments use zero-scratch serial scan.
CUDA switches to `global_scan` only when there are at least 65,536 active
items and the longest segment contains at least 4,096 items. The latter runs a
global provider followed by race-free segment-base correction. Users can
inspect `workspace.last_scan_method` or explicitly choose `serial` /
`global_scan`. This is a stable policy boundary, not a promise that the
threshold is optimal for every device.

Topology memory is reported separately by `layout.topology_bytes`
(four bytes per capacity item plus four bytes per offset). Workspace current
and peak report only reusable execution scratch. Short serial scan needs zero
scratch; global scan can retain provider storage and one base value per
segment. Immutable layouts may be shared across Python submission threads, but
each producer/Graph needs an independent workspace.

On the Windows development machine, the representative workload is 1,048,576
items, 4,096 segments of length 256, five median trials with 20 hot replays
per trial, reused layout/workspaces, and compilation/warmup excluded. GPU
measurements are taken only with no other Python/GPU compute process active.

| backend | reduce public | reduce Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.770 ms | 0.805 ms | 1.003 ms | 1.30x |
| CUDA | 0.0756 ms | 0.0736 ms | 2.881 ms | 38.1x |
| Vulkan | 0.0751 ms | 0.0716 ms | 4.538 ms | 60.4x |

| backend | i32 scan public | i32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.500 ms | 0.495 ms | 3.108 ms | 6.22x |
| CUDA | 0.165 ms | 0.161 ms | 6.304 ms | 38.3x |
| Vulkan | 0.176 ms | 0.187 ms | 8.859 ms | 50.3x |

| backend | f32 scan public | f32 scan Graph | host round-trip | host/public |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.604 ms | 0.516 ms | 3.714 ms | 6.15x |
| CUDA | 0.146 ms | 0.161 ms | 8.008 ms | 54.9x |
| Vulkan | 0.167 ms | 0.197 ms | 10.237 ms | 61.2x |

The immutable topology occupies 4,210,692 bytes. Its one-time build/upload was
10.67 ms on CPU, 17.40 ms on CUDA, and 32.56 ms on Vulkan. Short scan scratch
was zero; CPU grouped reduce retained 262,144 bytes, while the measured
CUDA/Vulkan grouped providers retained no Python-owned scratch.

A single counterexample with 64 segments of length 16,384 was used to prevent
short-workload overfitting:

| backend | explicit global scan | explicit serial | measured preference |
| --- | ---: | ---: | --- |
| CPU | 5.984 ms | 0.586 ms | serial, 10.2x |
| CUDA | 0.871 ms | 1.800 ms | global, 2.07x |
| Vulkan | 3.855 ms | 1.597 ms | serial, 2.41x |

These historical measurements describe the coarse ordinary API dispatch;
they are not a threshold sweep, a cross-driver guarantee, or evidence against
new whole-Graph implementations.

`GraphBuilder.segmented_scan()` has a separate CUDA complete-recipe domain for
fixed, disjoint 1D i32/u32 arrays and immutable segment layouts. The default
recipe providers retain serial, shared-memory chunk carry, global correction,
and (for mixed segment lengths) bucketed strategies, alongside register-shuffle
and hierarchical-warp carry. The new strategies perform exact modulo-2^32
inclusive/exclusive sums without global scratch; hierarchical carry uses 16
bytes of shared memory per block and two block barriers per chunk. Warp-only
carry needs neither shared memory nor block barriers. These are physical
implementations of the complete operation, not public block-size search axes.

The global-correction strategy uses a retained CUDA Graph with private,
Graph-bound scan scratch; ordinary Program arena growth/clear does not affect
it. Only `num_items` are processed, preserving unused capacity. This changes host
submission and scratch lifetime, not the mathematical sum or the ordinary
primitive's `method="auto"`. See [Graph runtime optimization](graph_runtime_optimization.en.md#retained-global-segmented-scan)
for recording, memory and first-use compilation trade-offs.

Search them through `builder.freeze().search_recipes(...)` and compare actual
workload metrics. Ordinary `method="auto"` is unchanged. This does **not** fuse
arbitrary neighboring producer/consumer kernels: value semantics, visible
intermediate stores and cross-language lowering still need a separate contract.

### Complete Graph segmented reduction

`GraphBuilder.segmented_reduce(values, layout, output, op="sum")` adds a
separate CUDA recipe domain for fixed, disjoint plain 1D i32/u32 ndarrays and
host-published immutable segment offsets. Output has one element per segment;
empty segments produce zero and all strategies use exact modulo-2^32 sums.
Floating-point reordering and automatic differentiation are not supported by
this entry point. The existing `experimental_segmented_reduce()` and
`PrimitiveSequence.segmented_reduce()` retain their broader contracts and auto
dispatch behavior; the new Graph baseline is segment-serial, not that auto route.

The default complete-recipe providers discover serial, warp-owned and
block-owned segments, plus chunk-partial/finalize when a segment spans multiple
internal chunks. These are full strategies, not exposed block parameters.
The two-stage route owns frozen tile offsets, segment-to-partial offsets and
partial results. Their exact requested bytes appear in the physical/resource
report; they are not driver VRAM peak. Already-owned input layout storage is
shared, not reallocated per candidate. Kernels and bindings are prepared before
steady replay, which launches the retained nested CUDA Graph without reading
device results back to Python. Disposal uses the existing Graph allocation
lifetime; the operation uses one workspace lane.

Use `builder.freeze().search_recipes(...)` and the existing selection/resolve
workflow. Short segments can favor serial execution; long segments can favor
the extra parallelism of partial/finalize at the cost of scratch and setup.
There is no universal speedup or implicit ordinary-auto change, and adjacent
producer/consumer kernels are not fused by this operation.

## Device-side Numeric Checks

These APIs are Forge additions. They are not vanilla Taichi 1.7.4/1.8.0 APIs.
They must be called from Python scope and cannot be called inside `@ti.kernel`
or `@ti.func`. The API call submits a native backend primitive and writes one
device-side scalar. The scalar is copied back to Python only when callers use
`to_int()`, `to_bool()`, `ok()`, or `to_float()`.

`check_count`-style APIs support scalar 1D `ti.ndarray`, `StructNdarray` scalar
member views, and root-dense-place dense fields with
`i32/u32/i64/u64/f32/f64`. `metric_reduce`-style APIs support the same input
forms for `f32/f64`; Vulkan currently exposes only the `f32` metric fast path.
For `max_abs_delta`, dense fields can be compared directly with same-shaped
plain ndarrays or `StructNdarray` scalar member views; this uses the backend
native mixed-storage path and does not stage through host memory.

```python
workspace = ti.algorithms.CheckWorkspace(max_items=n)
bad = ti.algorithms.index_bounds_check(indices, lower=0, upper=n, workspace=workspace)

# Python branching synchronizes by reading one scalar.
if not bad.ok():
    raise RuntimeError("indices out of bounds")
```

For hot loops, explicitly reuse `CheckWorkspace` / `MetricWorkspace` to keep
the result scalar, scratch buffers, and backend replay plans alive.

## Workspaces

Most native algorithms accept a `workspace=` object or return a reusable
workspace. Reusing workspaces keeps backend scratch buffers and native plans
alive across frames or repeated calls. This is the preferred pattern for hot
loops. Concurrent calls must use independent workspaces unless the individual
workspace explicitly documents synchronization.

Ordinary primitive GPU scratch is owned by the active Program's arena and is reclaimed
by the existing workspace clear/reset APIs; cached resources are not kept in
process-global owner maps. CPU primitive scratch retains at most 8 MiB per
family and worker thread, uses transient allocations from 8 through 64 MiB, and
keeps the established serial fallback above 64 MiB. These are bounded retention
policies, not promises that every operation's peak temporary storage is 8 MiB.

For public calls with `workspace=None`, cacheable algorithms isolate implicit
workspaces by active Program and Python submission thread. Each thread context
holds at most 64 entries by default, and the process holds at most 16 contexts.
Set `TAICHI_FORGE_DEFAULT_WORKSPACE_CACHE_LIMIT` and
`TAICHI_FORGE_DEFAULT_WORKSPACE_CONTEXT_LIMIT` before startup to lower or
disable those bounds. A new thread beyond the context limit uses an uncached
workspace; it never evicts or clears a foreign thread's resource that may still
be referenced by asynchronous work.

Call `clear_default_workspaces()` only after primitive submissions are
quiescent. It atomically detaches cache metadata and clears resources outside
the metadata lock; concurrently clearing an explicitly or implicitly in-use
workspace is outside the supported contract. `ti.reset()` establishes the same
kind of quiescent boundary and clears these caches.

```python
workspace = None
for _ in range(num_steps):
    workspace = ti.algorithms.experimental_transform(
        src, dst, scale=2.0, bias=1.0, method="auto", workspace=workspace
    )
```

Observability is opt-in and metadata reads do not wait for the device:

```python
ti.algorithms.set_primitive_diagnostics_enabled(True, clear=True)
ti.algorithms.experimental_transform(src, dst, method="auto")
snapshot = ti.algorithms.get_primitive_runtime_diagnostics()
print(snapshot["providers"], snapshot["fallbacks"])
print(ti.algorithms.get_primitive_workspace_statistics())
```

Both snapshots use `schema_version=1`. Runtime diagnostics report providers,
their `dependency_class`, fallbacks, and raw counters. The workspace snapshot
keeps Program-owned provider bytes separate from logical current/peak bytes in
the Python default cache. The domains can refer to the same underlying
resource and must not be added. CUDA CUB reference names are reported as
aliases of canonical driver families to avoid double counting. Reading this
metadata does not call `ti.sync()`. To explain one automatic choice, clear the
interval, execute only that call, and then read the snapshot.

## Graph Interaction

Forge can replay DSL-defined native primitive sequences through graph execution
when the sequence is produced by Forge's own algorithm layer.
`PrimitiveSequence.run_length_encode()`, `unique()`, `unique_by_key()`,
`segmented_reduce()`, and `segmented_scan()` retain fixed arrays/layouts and
reusable workspaces; replay does not read their device state back to Python.
`DeviceCheckResult`, `DeviceMetricResult`, and `PrimitiveSequence` can be
appended as native graph nodes with `GraphBuilder.append_native(...)`. Graph
replay updates only the device-side scalar. It does not automatically copy the
result back to Python and does not turn the check result into host-side control
flow inside the graph.

```python
workspace = ti.algorithms.MetricWorkspace(max_items=n)
err = ti.algorithms.max_abs_delta(values, reference, workspace=workspace)

builder = ti.graph.GraphBuilder()
builder.append_native(err)
graph = builder.compile()

graph.run({})
print(err.to_float())  # Explicitly reads one device scalar.
```

This does not expose arbitrary native callbacks to users, and ordinary Python
algorithm calls do not require graph. Native graph nodes are JIT replay nodes;
`ti.aot.Module.add_graph()` currently exports ordinary kernel CGraphs only and
rejects graphs containing Forge native nodes.

## Relationship to Vanilla Taichi

Vanilla-compatible `parallel_sort()` remains available. The broader dispatcher
and `experimental_*` primitive APIs are Forge additions.
