# Graph Runtime and Optimization

This document is the public source of truth for Forge graph runtime architecture,
backend replay, performance policy, diagnostics, and validation. For migration
from vanilla Taichi 1.7.4, see
[Graph compatibility and migration guide](graph_migration_guide.en.md). For exact public
signatures, see [Forge API reference](forge_api_reference.en.md).
The static-Field feature contract is maintained separately in
[Dense Field Graph](dense_field_graph.en.md).

The base Graph modernization and native-node replay model first shipped in
Forge 0.4.1. This page describes the current source contracts, including the
0.6.3 recipe additions; an older installed artifact may lack a named native
capability. See the
[release notes](release_notes.en.md) for the introduction version of each
capability.

## Scope and invariants

Forge keeps Taichi's public graph-builder model. Backend optimization must not
change these contracts:

- `GraphBuilder.compile()` freezes the dispatch and sequential definition.
- `Graph.run(args)` accepts either an exact runtime-argument dictionary or a
  `GraphBindingSet` published by the same Graph.
- One run of one `Graph` is a complete host transaction. Its CGraph and native
  nodes cannot be interleaved by another caller.
- Independent graphs remain independently submitable. Runtime guards end after
  host submission and do not add a default `ti.sync()`.
- `ti.reset()` invalidates graphs owned by the previous runtime.
- Destroying a referenced SNodeTree invalidates the Graph even if a later tree
  reuses the same numeric id.
- `Graph.run()` is primal-only and rejects active or concurrently entering
  Tape/FwdMode contexts rather than silently dropping automatic differentiation.
- An optimized backend path may fall back to ordinary dispatch, but may not
  silently change results, bindings, or execution order.

Runtime synchronization protects Forge-owned launch, replay, and resource
state. It does not make application-owned simulation and rendering data safe.
Asynchronous producers and renderers still need snapshots, slots, double
buffering, or another explicit ownership protocol.

## Runtime argument discovery and template adapters

Public applications should declare graph arguments through
`GraphBuilder.dispatch()` and pass either an exact-key dictionary or a
`GraphBindingSet` from the same Graph to `Graph.run()`. A physics or rendering
engine can use the keyword-only `template_args=` parameter
to bind a data-oriented `self`, a Field, or another `ti.template()` argument at
definition time:

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
```

These objects participate in specialization but do not enter the `Graph.run()`
dictionary. Field contents may change between replays; replacing Field identity
or layout requires rebuilding the graph. An ndarray or texture compile exemplar
still has a symbolic Arg, and each run still receives the real runtime resource.

Forge treats the durable AOT plan as the source of truth for dispatch
definitions and incrementally records its real symbolic argument names. This
recovers the exact runtime key set even when a legacy adapter bypasses the
Python fast-path registration in public `GraphBuilder.dispatch()`. Validation
remains strict: missing keys and extra keys not declared by the AOT plan still
raise instead of being ignored. Direct access to `_aot_graph_plan` and the
native builder remains only for legacy compatibility; new engine code should
use `template_args=`.

For a mixed `CGraph(a) -> native -> CGraph(b)`, the native node ends the
current CGraph segment. Forge recovers symbolic names only from AOT items added
to that segment, so `a` cannot contaminate the later `b` segment.
`Graph.run()` still validates the exact union of every segment's arguments. A
Field-only CGraph receives an empty argument map at the C++ execution layer,
and a native-only node receives no runtime dictionary.

Python flattens one invocation once and reuses its resource signature and
containers under the same Graph's per-Graph lock. The CompiledGraph binding
constructs a segment-local C++ `IValue` map from that segment's own
declarations; Python does not copy another dictionary per segment. This keeps
the backend semantics segment-local while preserving a zero-copy host path.
Legacy adapters that access underscored objects still work, but recovery reads
only AOT items added since the previous segment flush.

## Published immutable bindings and the two cache layers

`Graph.bind(arguments)` publishes an exact argument dictionary as the stable
public `GraphBindingSet` API. Each version snapshots Python scalar/matrix
values and retains device-resource identities; resource contents remain
dynamic. Replay of the same qualified published version reuses a preflattened
frame. It does not reconstruct Python storage descriptors or repeat owner,
layout, structured-control, or alias qualification. Dynamic providers,
replacement, derived/fixed bindings, temporary lanes, and per-submission owners
explicitly block this fast path and retain conservative final-frame validation.

This is complementary to the native CGraph's four-slot runtime binding-plan
cache:

- A Python BindingVersion owns Forge-level layout/alias legality, the immutable
  invocation snapshot, atomic concurrent publication, and version selection
  after a paced wait.
- A native plan compares generation-qualified resource objects/handles, reuses
  backend ABI/launch state, and reacquires asynchronous resource leases for
  every submission. It does not certify a Forge memory recipe.

Prepublish one BindingSet per set for a bounded collection of recurring
resources, then alternate them directly:

```python
bindings_a = graph.bind({"source": source_a, "output": output_a})
bindings_b = graph.bind({"source": source_b, "output": output_b})
for bindings in (bindings_a, bindings_b, bindings_a):
    graph.run(bindings)
```

This A-to-B-to-A sequence reuses both the Python qualification certificates
and native MRU slots. `update()` and `replace()` are explicit publication
boundaries: a candidate is fully qualified before it becomes visible, and
failure leaves the old version current. Their cost is outside the replay
performance contract. `fast_path_qualified`, blocker strings, and statistics
are performance diagnostics rather than correctness admission. `ti.reset()`
invalidates the Graph and all of its BindingSets.

### Complete-recipe immutable CUDA frames

On capable CUDA runtimes, the default recipe providers can also generate an
exclusive whole-Graph immutable-frame candidate. It supports one ndarray-only
segment containing ordinary JIT dispatches and fixed-plan commands produced by
`ti.linalg.record_fft()`, `SparseMatrix.record_spmm()`, or (with native matmul-frame
support) `ti.linalg.record_matmul()`. Compatible regions may be composed in one
segment. Arbitrary vendor recordings, external synchronization domains, SNodes,
device-controlled topology and multi-lane workspaces are not included. Captured
commands must produce kernel nodes only. Discovery alone does not prove that an
installed vendor plan meets this condition; binding preparation reports failure.

For this selected recipe, `Graph.bind()` validates and captures each immutable
frame without executing mathematical work. It uploads at most one packed JIT
argument image, retains the provider/allocation owners, and prepares one reusable
executable. Switching published frames updates that executable without recapture,
argument uploads or host-device synchronization. Completion events protect
in-flight frames; switching is not claimed to have zero driver overhead. Matrix
values may still change through the fixed-pattern `update_values()` contract.

Measure preparation, device execution, host submission and retained memory
separately. Raw mapping calls and binding churn include capture/preparation;
published frame objects retain argument images and opaque driver Graph objects.
The composer can wrap explicitly compatible FFT/SpMM computation fragments with
one whole-Graph executor; the executor does not replace their semantic coverage.
Unrelated families are not implicitly compatible. Completed event handles are
reused up to the observed queue peak and retained until executor close/reset;
this avoids repeated event creation/destruction after backlog retirement, without
preallocation or new replay synchronization. Their opaque driver storage is not
included in ndarray requested bytes or advertised as a measured VRAM peak.
Diagnostic binding-frame snapshots expose created/reused/cached/destroyed counts.
The expanded provider domain
invalidates older provider-bound search evidence, not otherwise compatible wheels.
Ordinary runtime auto selection is unchanged. Validation here is Windows-only;
production qualification remains application-owned.

### Retained global segmented scan

The complete i32/u32 `GraphBuilder.segmented_scan()` global-correction strategy
records input copy, hierarchical Driver scan, base gathering and correction in
one retained CUDA Graph. It retains its own exact scratch ndarray through the
existing Graph allocation leases; clearing or growing the ordinary Program scan
arena cannot invalidate that recording. The active `num_items` are scanned and
unused capacity is left unchanged. Preparation loads kernels, but does not run
the user's computation. Replay has no primitive-arena acquisition or host scalar
readback. Ordinary primitive `method="auto"` is unchanged.

This primarily removes separate host launches, not scan kernels. Each live
materialization retains private scratch, so multiple Graphs can use more memory
than a shared arena. Reports distinguish segment bases and retained scan scratch;
requested bytes are not allocator/driver residency. Additional first-use JIT
compilation can also cost more. This path requires the native retained-scan
capability and remains a fixed-resource Graph action, not arbitrary
producer/consumer fusion or a binding-frame-compatible region.

### Certified pointwise values around segmented reduction

On CUDA, default complete-recipe discovery can fuse a pure pointwise producer,
consumer, or both around `GraphBuilder.segmented_reduce()`. The compiler certifies
the actual value IR, not just adjacent dispatch labels. The first supported domain
is scalar i32/u32 addition/subtraction/multiplication, casts, negation and bitwise
expressions (constant shifts 0--31) over compact 1D ndarrays, integer
constants/scalars and exact zero-based iteration domains. It excludes
floating point, random/atomic operations, conditional bodies, multiple stores,
SNodes, views, arbitrary calls and debug/bounds-instrumented kernels. Unsupported
semantics retain the unfused route.

Producer output must be the reduction's fixed values array; the forwarded consumer
input must be its fixed output. `Graph.bind()` checks these identities, exact
iteration coverage and disjoint writes/unforwarded reads once at publication.
Names alone do not establish dataflow. All observable producer/reduction/consumer
stores and unused capacity are preserved; this is not temporary-storage elimination.
An incorrect candidate binding is a structured search failure, not silent fallback.

The provider combines value forwarding with serial, warp, block or partial/finalize
reduction and the existing immutable-frame submission option. Partial/finalize
maps the producer only into the input phase and the consumer only into finalization.
Unaffected prefix/suffix dispatches retain their order and semantic coverage. One
ordered workspace lane is supported. Prepared replay does not rerun the compiler
query, storage proof or argument uploads; new binding publication includes setup.

Fusion is not universally faster. Moving a large map into a few long-running
reduction blocks can reduce parallelism; partial/finalize adds scratch and restores
parallel work. Keep these legal alternatives in search and report device time,
host submission, synchronization and owned scratch separately. CUDA event spans
may include host starvation and are not kernel-active time. Driver Graph/allocator
VRAM remains unknown unless independently measured. The implementation has local
Windows evidence, not production or Linux qualification, and does not change auto.

### Current complete-recipe hardware domains

These are source support boundaries, not a promise that every installed wheel or
vendor runtime implements every optional capability. Use the ordinary
`freeze -> search_recipes -> resolve_recipe -> materialize` workflow and supply
explicit providers alongside `ti.graph.default_recipe_providers()` where shown.

| Domain | Provider selection | Physical strategies and boundary |
| --- | --- | --- |
| Segmented integer reduction | Default | Serial/cooperative/partial-finalize; fixed host-published layout and modular sum |
| Certified pointwise reduction values | Default | Producer/consumer forwarding and immutable submission within the bounded IR contract above |
| Dense matmul | Explicit `ti.hardware.linalg.MatmulRecipeProvider()` | Frozen cuBLASLt choices, operand packing and equivalent epilogues |
| Contraction | Explicit `ti.hardware.tensor.ContractionRecipeProvider()` | Retained cuTENSOR capture, input permutations and epilogue dataflows |
| Shared-A sparse matmul | Explicit `ti.hardware.tensor.SparseMatmulRecipeProvider()` | FP16 2:4, per-invocation compression shared by products; no automatic pruning |
| Sparse solve | Explicit `ti.hardware.linalg.SparseSolveRecipeProvider()` | Private cuDSS analysis/factor lifetimes and captured numerical phases; distinct from legacy root-ordered recording |
| Vulkan FFT | Explicit `ti.hardware.fft.VulkanFftRecipeProvider()` | VkFFT batch scratch sharing and whole-Graph retained command recording |

See [native algorithms](native_algorithms.en.md) and the
[external-provider contracts](external_hardware_providers.en.md) for inputs,
preparation, restoration, runtime dependencies and numeric policies. Search
uses the maintained **CompileIQ fork** and opaque complete-recipe identities;
library names and individual kernel parameters are not search axes. Missing optional
native/adapter support makes only that candidate unavailable. Reports/checkpoints
preserve provider and environment applicability; wheel compatibility is based on
version/ABI/capabilities rather than equality with a source commit.

## Dense Field lifetime and heterogeneous blocks

Dense scalar, vector, and matrix Fields are supported as definition-time
bindings. Their contents may change; identity, layout, shape, dtype, element
shape, SNodeTree generation, and owning runtime may not be hot-rebound in that
form. Bind a compatible Field to an `ArgKind.NDARRAY` runtime slot when its
identity must change between invocations; every submission revalidates the
storage descriptor and generation. Sparse topologies remain outside this dense
storage contract. Heterogeneous applications should group homogeneous
environments inside stable blocks and use explicit snapshot ownership between
asynchronous simulation and rendering.

The complete support matrix, lifecycle transaction, multi-environment layout,
AD boundary, performance evidence, and Linux status are maintained in
[Dense Field Graph](dense_field_graph.en.md).

## Backend execution model

| Backend | Graph execution | Main safety boundary | Replay resource policy |
| --- | --- | --- | --- |
| CPU | Cached JIT dispatch plan; no device graph capture | One compiled graph is a complete replay transaction; ordinary kernels are protected at the whole-kernel boundary | CPU scheduler and JIT state are runtime-owned |
| CUDA | CUDA Driver API capture and executable replay, with patch or recapture when bindings change | Capture/replay and direct submission are serialized at the native host-submission boundary | Captured allocations are generation-qualified and retained until ordered retirement |
| Vulkan | Runtime-owned command recording and replay | GFX recording and replay registry mutations are protected per host API call | Monotonic graph identity, deferred retirement, fixed eight-slot in-flight ring |

Ordinary CGraph paths use bounded MRU state rather than an unbounded signature
history. Each CGraph retains up to four generation-qualified runtime binding
plans; CUDA retains two executable resource signatures and Vulkan retains four
immutable launch signatures. This covers common ping-pong and short ring
buffers while bounding driver objects and retained allocation leases. CUDA
scalar or matrix value changes patch the current compatible executable instead
of consuming another resource slot; Vulkan retains recurring value signatures
inside its same four-slot bound. Stale generations and `ti.reset()` retire all
related entries.

These ordinary-cache limits do not describe the opt-in immutable-frame recipe:
its published frames and observed completion-event peak have the separate
ownership policy described above.

## Task observability without launch control

Forge exposes the final offloaded-task shape through
`kernel.task_manifest(...)` and, for one-segment JIT CGraphs,
`Graph.task_manifest()`. The immutable report distinguishes requested,
selected, and actual grid/block geometry, reports static/dynamic shared bytes,
and assigns a specialization-local stable `task_id`. This is an observation
surface, not a second launch API: it cannot change grid/block geometry.

Graph dispatches accept an optional `label=` and ordinary kernel calls can use
`ti.profiler.dispatch_label(...)`. Labels are invocation state, never mutable
compiled-kernel state, so concurrent callers cannot overwrite each other's
sweep/color/phase identity. Profiler and optional NVTX event names keep the
original task name and append the task identity and label.

Labels do not change CUDA/Vulkan replay qualification. They remain stable task
metadata in manifests and explicit telemetry, while production execution stays
replay-first and adds no device allocation, transfer, or synchronization. A
capture-time profiler cannot manufacture a fresh host annotation for every
later replay; use ticket telemetry or an explicit profiler capture when
per-invocation evidence is required. Vulkan device-indirect dispatch remains
native and its manifest marks actual geometry as invocation-specific.

The CPU path preserves graph semantics and concurrency safety but does not
pretend to offer CUDA-style device graph launch. CUDA and Vulkan optimizations
are backend implementation details below the same public API.

## Structured control

`GraphBuilder.while_loop()`, `if_then_else()`, and `switch()` add
backend-neutral structured regions without introducing a solver-specific
Graph API. Each region is built from fixed `Sequential` definitions. Runtime
values may change between replays, but the argument schema, resource identity,
shape, dtype, and dispatch topology remain fixed.

A bounded iterative program uses a condition region and a body region:

```python
condition = builder.create_sequential()
condition.dispatch(
    evaluate_stop,
    residual_sq,
    initial_norm_sq,
    user_stop,
    predicate,
    status,
    atol,
    rtol,
)

body = builder.create_sequential()
body.append_native(operator.graph_action(direction, product))
body.dispatch(update_iteration, direction, product, counter, status)

builder.while_loop(
    condition,
    body,
    predicate=predicate,
    status=status,
    control_inputs=(residual_sq, initial_norm_sq, user_stop, atol, rtol),
    carried_state=(direction, product),
    counter=counter,
    max_iterations=128,
    lowering_mode="auto",
    name="iterative_program",
)
```

The condition kernel may combine any number of DSL-computed criteria, such as
absolute and relative tolerance, user cancellation, active work, or numerical
breakdown. It writes a one-element integer `predicate` ndarray; nonzero means
continue. An optional, distinct one-element integer `status` ndarray records
why execution stopped. Graph transports and reports that value but does not
assign solver meanings to status codes. The optional `counter` is the exact
logical iteration count. `max_iterations` is always a host-defined safety
bound and does not need to be encoded in a solver-specific condition.

`if_then_else()` selects one fixed branch from a predicate computed by its
condition region. `switch()` selects a zero-based fixed branch, or an optional
default, from a selector computed by its condition region. Branch schemas are
compiled before execution; Python callbacks cannot run inside a region.
The same `while_loop()`, `if_then_else()`, and `switch()` builders are
available on `Sequential`, so a root structured region may contain one more
structured level. Definitions form a single-owner tree. The maximum structured
depth is two. Deeper definitions, cycles, reuse at multiple call sites, and
unqualified `native_required` nested definitions fail before execution during
region construction or Graph compilation.

Current lowering is explicit:

| Backend | Structured `while` | `if` / `switch` |
| --- | --- | --- |
| CPU | Exact `cpu_host_loop`; condition and body use cached compiled dispatch plans | Exact portable host control |
| CUDA | `auto` uses a native CUDA conditional Graph on qualified Driver API 12.8+ runtimes; older drivers use Forge's bounded masked Graph when ordinary CUDA Graph capture is available; otherwise exact portable replay | Qualified 12.8+ runtimes use native CUDA IF/SWITCH nodes; older drivers use the same internal device latch and task-entry gate contract; otherwise exact portable host control |
| Vulkan | Exact portable replay, or qualified `native_required` bounded masking with positive per-region chunk sizes capped at 64 and an eight-chunk/512-iteration region limit | Exact portable host control |

At depth two, CPU executes the complete tree with exact host control and
returns an already-completed submission ticket. CUDA and Vulkan additionally
qualify one ordered native shape: an outer `while` whose body contains from one
through eight leaf inner `while` regions, with ordinary dispatch or qualified
recordable-action gaps between them. It executes under one backend submission
and one ticket. Other shapes retain exact portable-parent control and reject
asynchronous submission; a qualified `auto` leaf may still use its existing
flat native route.

`ti.graph.structured_control_capabilities()` returns the active backend's
schema-v5 portable lowering and device-control qualification. The report
separates primitive availability, complete runtime qualification, compound
submission, terminal observation, per-region chunk and first-gate policy, tail
strategy, queue-submit coalescing, and exact dynamic termination. Vulkan
qualifies bounded `while` execution without claiming native `if`/`switch` or
exact termination of an already encoded command chunk.
CUDA similarly distinguishes `cuda_conditional_graph` from
`cuda_masked_bounded_graph`. The latter encodes at most 4096 dispatches in one
submission, latches control on device, and returns inactive Taichi tasks before
payload side effects. It has no per-iteration host readback, but all encoded
task nodes still reach command issue, so `stops_command_issue_after_exit` and
`exact_dynamic_termination` are false.
The nested `cuda_conditional_graph` report exposes
`exact_control_unavailable_reason`, `masked_control_unavailable_reason`,
`selected_general_graph_control`, and
`selected_general_graph_control_unavailable_reason`. A driver below 12.8 can
therefore report exact control as unavailable while selecting the qualified
internal masked route; it is not mislabeled as a complete control failure.

`lowering_mode="portable"` forces the portable route.
`lowering_mode="native_required"` requires the qualified backend route and
fails before execution when unavailable. On CUDA this means either exact
conditional Graph control or internal bounded-masked device control. On
Vulkan it means a bounded `while` with at most 512 iterations,
at most eight positive-size chunks capped at 64 iterations, runtime replay mode enabled, and no
unsupported profiler or dispatch-cache configuration. An omitted `chunk_size`
selects 64 for compound submission. Recordable provider actions may enter a
structured body only when their provider declares it safe; opaque or
unsupported providers fail closed.

CUDA and Vulkan share one qualified depth-two shape: with tracing disabled and
the outer mode set to `auto` or `native_required`, an outer `while` whose body
contains from one through eight ordered leaf inner `while` regions can execute
as one bounded backend submission. Ordinary dispatches and qualified
recordable actions may appear before, between, or after the inner regions.
The outer loop and every inner loop require counters. All predicate, counter,
and optional status controls must be mutually distinct one-element i32 device
ndarrays owned by the same Program. Vulkan additionally requires conditional
rendering, nested runtime binding, and ordinary replay, with the kernel
profiler and Vulkan dispatch cache disabled. CUDA requires ordinary Graph
capture. On a qualified Driver API 12.4+ runtime it uses device-updatable
kernel-node groups: each business dispatch is compiled once, while small
device updater nodes enable or disable the statically repeated payload groups.
An explicit cached setup probe qualifies that route. When the probe is
unavailable or fails, Forge uses its version-independent two-gate task-entry
masking route; `TI_GRAPH_CUDA_FORCE_MASKED_CONTROL=1` forces that fallback for
A/B qualification on a current driver. Neither CUDA route depends on 12.8
conditional nodes.

The outer and every inner bound are each from 1 through 64; every inner chunk
is positive and no larger than 64 or its budget; and the additive complete
encoded program contains at most 4096 actions. The outer prefix/suffix, gaps
between inner regions, and all loop condition/body sequences must contain only
ordinary dispatches or qualified recordable actions. Vulkan uses bounded
conditional replay. All GPU routes
avoid host readback between the two levels but retain bounded static topology;
none claims exact dynamic command termination. Any other nested shape takes
exact portable-parent control; an eligible leaf `while` may still use its flat
backend route. Vulkan still does not provide native `if`/`switch`.

The device-control capability report exposes `nested_async_route`, the CUDA
candidate/qualified/forced-off state, the explicit fallback route,
`nested_no_host_readback`, and `nested_exact_dynamic_termination`. A submitted
nested Graph can preserve per-outer stop positions in an outer suffix device
trace, or expose a recordable provider's terminal packet after ticket
completion; this does not add a hidden host observation between the loops.

Native structured routes distinguish a pre-submit qualification miss from a
post-submit observation failure. The former may select the documented exact
portable route. Once the queue has accepted a side-effecting submission,
completion, terminal-observation mapping, or trace decoding failures raise
immediately and never trigger portable fallback, so a Graph body is not
executed twice.

Recordable providers may also declare private symbolic scratch bindings backed
by Graph temporary requirements. The Graph memory plan materializes one bounded
arena slot per in-flight invocation, resolves the private symbols before
submission, and keeps them out of `Graph.run()` / `Graph.submit()` arguments.
Bindings are reused for repeated execution of the same arena slot and rebound
when another asynchronous slot is selected. Providers must declare exact byte
and alignment requirements, return the complete declared symbol mapping, and
reject incompatible storage before backend work is submitted.

Explicit complete-recipe search with `GraphResourceLifetimeRecipeProvider` also
offers queue-ordered scratch for a fully lowered, single CUDA CGraph. The final
physical executor must contain only ordinary kernel dispatches, without parallel
lanes, external capture commands, dynamic provider bindings or an alternate
binding executor. Its zero-offset temporary mappings are resolved against owned
storage at materialization. One eagerly allocated slot per Graph instance is
then reused by runtime stream order, without arena completion polls or waits.
This is not a raw slot-count axis; unsupported Graphs keep the completion ring,
and ordinary runtime defaults do not change. Pool reserved pages may remain
unchanged even when requested scratch shrinks. Measure host, device and actual
reservation separately; scratch reduction is not a device-speedup claim.

At the Graph root, consecutive ordinary CGraph segments and compatible
recordable-provider actions are lowered into one backend region. Fixed and
private temporary bindings are merged before compilation; conflicting bindings
fail explicitly. Structured regions inline the same provider dispatches only
when the provider has qualified the corresponding condition/body/branch role.

`Graph.control_flow_stats()` returns one immutable `GraphWhileReport` or
`GraphBranchReport` per static structured definition for the latest `run()`;
repeated nested calls retain only that definition's latest invocation. Native CUDA
IF/SWITCH execution keeps `Graph.run()` fire-and-continue: selector readback and
report construction are deferred until `control_flow_stats()` is requested, so
that diagnostic call is the explicit synchronization point. While reports
include the selected lowering, logical and executed iterations, encoded and
masked iteration slots, observation boundaries, predicate/counter/status
traces, terminal status, transfer bytes,
and native-upgrade reason. The strict Vulkan outer while report additionally exposes
`region_path`, `structured_depth`, `nested_region_path`,
`nested_logical_iterations`, and `nested_encoded_iterations`.
`Graph.run(args, trace=True)` synchronously returns an ordered
`GraphControlFlowTrace` containing every nested invocation, including its
sequence, definition path, invocation path, parent iteration, and report.
Tracing bypasses strict Vulkan nested replay and uses the exact portable-parent
path so that every invocation remains observable.

Portable structured regions use synchronous
`Graph.run()` and are rejected by `Graph.submit()` rather than being hidden
behind an asynchronous ticket. CUDA `while_loop`, `if_then_else`, and `switch`
regions declared with `lowering_mode='native_required'` may use `Graph.submit()`
when either exact conditional Graph lowering or the bounded masked CUDA Graph
route is available. Qualified Vulkan
`native_required` `while_loop` regions may also use `Graph.submit()`; Vulkan
branches remain portable. Ordered CUDA setters and Vulkan predicate gates
consume control state without a per-region host readback. After asynchronous
structured submission, read terminal state from `ticket.observations()`;
synchronous control-flow reports remain unavailable for that submission.
An eligible depth-two `while -> while` Graph uses one `Graph.submit()` ticket
on CUDA and Vulkan; CPU executes the exact host-controlled hierarchy before
returning a completed ticket. The inner terminal may be consumed by an outer
suffix kernel without host synchronization. A SolvePlan action exposes this
through its device terminal packet; general code can copy the inner counter or
status into a device trace in the outer suffix. This retains every outer
invocation's stop position. Synchronous `Graph.run(trace=True)` remains the
richer diagnostic path and intentionally uses portable execution.
For per-invocation diagnosis, `submit(telemetry="summary")` records entry/exit
control scalars around submitted root while regions and `ticket.telemetry()` reports
logical stop positions, encoded and masked iteration slots, skipped coarse
chunks, the queue-counter window, and host enqueue time. The default path does
not allocate telemetry storage or enqueue these snapshot kernels. Use
`telemetry="timestamps"` when whole-ticket and structured-region GPU timestamps
are also required; `telemetry=True` remains its compatibility alias. Summary
mode inserts no backend timestamp markers and reports GPU duration as
`disabled_by_mode` rather than inferring it from wall time.

The same opt-in telemetry owns a `GraphPipelineReport` selected from the
post-optimization execution root. It preserves coalesced CGraph/native stage
boundaries and immutable `NativeActionManifest` values for provider-declared
symbolic effects, public and provider-derived private bindings, temporaries,
and backend eligibility. The report
never exposes storage objects or addresses. Stage dispatch counts are static
compiled counts; provider temporary bytes are declarations rather than an
allocation peak. Structured-region timestamps are attached only in timestamp
mode and where the backend measured that region, while ordinary stages keep
their duration unavailable and retain the whole-ticket timing separately.
Vulkan writes these opt-in markers into runtime-owned command lists instead of
allocating marker-only command lists. Timestamp mode still reports
`gpu_measurement_path_changed=True`; it is diagnostic timing, not the normal
execution latency. Calling
`ticket.pipeline_report()` returns this same ticket-owned object. With
`telemetry=False`, no pipeline report or telemetry arena is materialized.
Call `graph.prepare_telemetry("summary")` before a sampling window to allocate
the bounded arena and compile packed snapshot kernels without executing the
Graph or reading its runtime arguments. `prepare_telemetry("timestamps")`
also performs one empty instrumented transaction so lazy backend event/query
initialization is paid at that explicit boundary. The optional `slots` count
prepares bounded concurrent records for currently materialized workspace lanes;
default submissions remain unaffected.

### Vulkan compound structured transactions

A single Vulkan submission transaction may contain multiple ordered bounded
`while` regions. The runtime pre-enqueues every qualified replay chunk, keeps
region dependencies in Graph program order, and publishes one final
`SubmissionTicket`. It does not read a terminal predicate between regions.
Submit-only replay omits the per-chunk terminal shader and host-observation
copy used by synchronous control-flow reports; an explicit
`GraphBuilder.observe()` snapshot still executes once at the transaction end.
This is a generic Graph contract: solver, line-search, contact, and other
meanings remain in user kernels and recordable providers.

Ticket telemetry uses two packed i32 device snapshots per while region and one
post-completion host readback. Queue counters are currently a non-exact
device-wide transaction-window delta, because a concurrent external
graphics/interop producer can change them. GPU timestamps are reported as
unavailable instead of being inferred while timestamp instrumentation would
invalidate the qualified compound replay path.

Each region honors its explicit `chunk_size` for compound submission. The
default first chunk uses compact per-iteration indirect masking. A region may
instead select `coarse_conditional` for the first chunk; this is useful when the
region itself may be inactive and fails closed unless conditional rendering is
qualified. Under `auto`, a small gate shader copies the entry predicate into
stable control storage for every later chunk and one conditional command
surrounds that whole chunk. Termination within the active chunk still leaves a
masked tail, while later inactive chunks skip their shader dispatches. This
reduces the submitted shader workload without claiming device-generated
commands or exact command-stream termination.

The runtime batches command buffers recorded inside the transaction into one
Vulkan queue submission while preserving each command buffer's wait and signal
semaphores. A final empty fence submission establishes the public completion
ticket. First-use observation or replay allocation may leave an earlier command
list that is flushed before the batch; steady-state execution therefore has
one transaction batch plus one completion-fence submission. All command
buffers in a batch conservatively share the batch fence for retirement.

Compound replay prepares the Graph argument bindings, kernel handles, SNode
dependencies, resource retention, and submission guards once per region, then
launches all of that region's chunks from the prepared state. It does not
re-enter the complete JIT-launch preparation path for every chunk. For
qualification and rollback, setting
`TI_VULKAN_COMPOUND_SINGLE_PREPARATION=0` restores the legacy per-chunk
preparation path without changing the Graph.

Within each recorded structured command buffer, Forge derives allocation-level
read/write effects from task buffer bindings. It emits memory barriers for
read-after-write, write-after-read, and write-after-write dependencies, and
flushes pending effects at controller and command-buffer boundaries. Independent
tasks and read-after-read pairs no longer receive an eager barrier. Unknown
global effects remain conservative. The reported capability fields are
`compound_single_preparation` and `structured_barrier_policy`; setting
`TI_VULKAN_STRUCTURED_HAZARD_PLANNER=0` restores eager per-task barriers for
qualification.

The GFX host API mutex is held for the complete transaction, preventing an
unrelated producer from being absorbed into the batch. The fixed eight-slot
replay ring provides bounded inter-invocation backpressure. A
non-indirect Graph may still use ordinary task launch when every slot is busy;
an indirect Graph instead waits for the oldest slot because its device-written
dispatch packet has no semantics-preserving ordinary fallback. This wait is
entered only after all eight slots are in flight and never grows the ring. A
`SubmissionPacer` can regulate complete invocations and lanes, but neither
mechanism preempts GPU work or assigns priorities. Large compound submissions
should keep explicit iteration budgets and use application-level pacing when
they share a device with latency-sensitive work.

Conditional-control metadata is uploaded asynchronously on the ordered default
stream and retained until the associated replay completes. The runtime keeps at
most two deferred replay batches; a third rapid submission waits for the oldest
batch instead of growing host staging and event state without bound. This
backpressure does not create a worker thread, an additional CUDA stream, or
device concurrency. `Graph.execution_stats()` exposes
`asynchronous_control_updates`, `deferred_replay_waits`, and
`peak_deferred_replay_batches` for qualification.

## Opt-in completion tickets

`Graph.run(args)` retains its established hot path and return contract.
Applications that need explicit asynchronous ownership can instead call
`ticket = Graph.submit(args)`. Submission validates the same exact runtime
arguments, serializes one Graph invocation at the same host boundary, and
publishes exactly one Program-local completion after all mixed CGraph and
native segments have been enqueued.

`ticket.done()` performs a nonblocking backend query; `ticket.wait()` waits for
that invocation rather than the whole device. Neither method inserts a default
`ti.sync()`. CPU completion is immediate. CUDA uses a Driver API event and
Vulkan uses a stream semaphore; a short GPU invocation is allowed to be ready
before its ticket is returned. Completion errors are sticky and surface from a
later `done()`, `wait()`, or runtime synchronization boundary.

Context-fatal CUDA errors and Vulkan device loss also become the Program's
immutable first fault. Once observed, later `Graph.run()`, `Graph.submit()`,
kernel, ticket-recording, synchronization, and Vulkan display submissions
fail fast instead of issuing more backend work. A failed Graph invocation is
never retried through ordinary dispatch. Stop producers and use `ti.reset()`
to retire the old Program; this does not promise recovery of a lost context or
device, so a real backend loss may require restarting the process. See
[Fatal backend errors and runtime reset](forge_api_reference.en.md#fatal-backend-errors-and-runtime-reset).

Pending runtime arguments are retained by the Program completion domain.
Graphs and Forge native workspaces are retained by the Python runtime owner
registry until the same completion becomes ready, even if the ticket is
dropped. Collection occurs on later submission, polling, synchronization, and
reset; the native completion queue is bounded so abandoned tickets cannot make
backend tracking grow without limit. This is a deliberately small completion
API: callbacks, `asyncio` adaptation, cross-Program ordering, and an explicit
Graph dependency scheduler remain out of scope.

## Bounded cooperative submission pacing

Applications with multiple asynchronous producers should combine completion
tickets with explicit admission pacing. Share a `ti.graph.SubmissionPacer`
across related `Graph.submit()` calls or CUDA/Vulkan batch-solve submissions to
bound backend invocations in flight, per-lane occupancy, and calls waiting for
admission. Admission occurs before backend enqueue, so the complete host launch
sequence of one paced invocation does not interleave with another. Invocations
already admitted to the backend remain asynchronous.

Scheduling is work-conserving round robin across lanes and FIFO within a lane.
Assign stable lanes to independent rhythms such as physics, rendering, and
streaming. Set `max_in_flight_per_lane` when one producer must not occupy every
slot. `on_saturation='wait'` applies backpressure. A real-time loop that cannot
block can use `on_saturation='raise'` and explicitly degrade or skip that frame
before any backend work has been submitted.

While a caller is blocked for capacity, the pacer uses that caller to poll all
in-flight completions with bounded adaptive backoff. This allows a later fast
invocation to free capacity without waiting for the oldest slow invocation and
does not require a persistent worker thread. The per-lane limit remains the
mechanism that reserves capacity against a high-rate producer; completion
polling does not preempt work already queued on the device.

The mechanism coordinates only calls sharing that pacer; ordinary kernels,
`Graph.run()`, and unpaced submissions remain outside its control.
`statistics()` exposes current and peak in-flight/queued counts, per-lane
grants and completions, rejections, backend failures, and admission wait time
for capacity and cadence validation. The pacer does not provide priorities,
deadlines, callbacks, or cross-Program dependencies. Applications needing
those policies should implement them above this admission boundary.

### Concurrency and resource budgeting

Host asynchrony is not a proxy for device parallelism. The public contract does
not guarantee that paced invocations receive independent CUDA streams or Vulkan
queues. Increasing `max_in_flight` first increases the number of invocations
allowed to queue and retain resources, not the number of available GPU cores.
An incomplete Graph may retain runtime-argument allocations, native
workspaces, replay command state, and a completion object. Mixing Graph and
solve work also adds plan-clone workspaces and operator numeric generations.

Pacer admission is based on invocation count and does not weight memory or
estimated GPU time. Its schema-v2 `statistics()["contract"]` identifies the
admission unit, lack of a device-concurrency guarantee, non-preemptive behavior,
and workspace, generation, and unpaced-submission exclusions. Graph
`execution_stats()` exposes `persistent_argument_bytes` and
`replay_slot_saturation_fallbacks`, but backend-driver command buffers,
descriptor pools, and allocator reservation still require backend profiling
and process-memory measurement. The persistent-argument total includes
condition/body caches and qualified control/observation arenas owned by
structured nodes even though those internal caches are not expanded into
public CGraph segments.

Start with one invocation in flight. Increase the limit to two only when an
Nsight or equivalent trace demonstrates useful overlap between host enqueue or
wait and productive GPU work, while peak memory, p95/p99 latency, and replay
saturation remain within budget. Do not treat the runtime completion safety cap
as an application queue depth. Coarsen Graph work or increase batching before
creating one asynchronous ticket per small task.

## CUDA capture and replay

Each CUDA graph executable owns its capture stream, stable argument buffers,
resource signature, and retirement state. Kernel-module launches receive the
capture stream explicitly; capture-owned buffers are retired in that stream's
order.

On the host, a process-wide native submission transaction keeps one graph
capture/replay, or one complete multi-task kernel, contiguous with direct
driver-kernel submission on the shared primary context and default-stream
ordering domain. Python graph arguments remain per invocation and native
execution may release the GIL. The transaction ends after steady-state host
enqueue, so GPU execution remains asynchronous; initial capture or recapture
keeps only its required local synchronization.

The resource signature includes generation-qualified allocation identity, byte
span, dtype, shape, element shape, and layout. The executable holds allocation
leases while captured work can still reference them. Deleting an ndarray,
running Python GC, or reusing an allocator slot therefore cannot redirect an
old executable to a new allocation.

Changing scalar or matrix values, or rebinding an ndarray with the same
structure, patches graph-owned argument buffers and reuses the executable.
Structural ndarray changes recapture. Texture arguments conservatively use
ordinary dispatch until they have equivalent lifetime ownership. Old leases
and host patch payloads retire behind CUDA events with a bounded in-flight
budget.

Capture failures are classified rather than treated as one permanent state:

| Failure class | Policy |
| --- | --- |
| Unsupported argument or preflight condition | Use ordinary dispatch for that invocation |
| Unsupported graph structure or `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` | Disable capture for that fixed graph cache |
| Other non-fatal capture or instantiate failure | Retry after 1, 2, 4, 8, 16, then at most 32 ordinary invocations |
| Illegal address, assert, launch failure, or another context-fatal result | Raise immediately; do not launch the same invocation again |
| Exception while capture is active | End capture through the native guard, then propagate the exception |

This policy follows NVIDIA's
[Driver API result semantics](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html)
and [stream capture contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html).
The implementation dynamically loads the CUDA Driver API and adds no CUDA
Toolkit header, CUDART, or CUDA-versioned wheel dependency.
Windows builds enable standard MSVC exception unwinding (`/EHsc`) when the
embedding build has not selected another `/EH*` mode. Linux compiler flags are
unchanged.

## Vulkan replay and slot capacity

Vulkan replay uses a monotonic nonzero graph identity and explicit
runtime-local registration. It does not use a reusable JIT-cache host address
as identity. Destroying or clearing a cache requests retirement; command
buffers, descriptors, and completion semaphores remain owned until every
in-flight slot is ready. Later launches or synchronization collect completed
state without adding a retirement wait. Runtime shutdown closes registration
before device teardown.

Replay deliberately uses a fixed ring of eight in-flight slots. Ordinary
`Graph.run()` replay may take ordinary dispatch when all slots are busy.
Asynchronous structured submission instead waits at the next complete replay
slot boundary before enqueueing that region; it never submits a partial
invocation and then falls back. Local bounded-growth experiments removed rare
saturation fallbacks but did not produce repeatable median throughput gains.
A 1024-graph churn sample with a 16-slot cap increased driver-reported Vulkan
memory by about 2.55 GiB even though host RSS and exact results remained
stable. Driver-retained command, descriptor, and semaphore pools can outlive
host graph state.

Forge therefore does not expose slot capacity as a DSL option or grow it per
graph. Re-evaluate this policy with both
`tests/python/vulkan_graph_slot_bench.py` and the graph-retirement stress;
eliminating fallback counts alone is not a sufficient optimization result.

## Diagnostics

Use the stable, frozen `Graph.execution_stats()` schema v7 report. It exposes
definition counts, compiled task count, segment-local runtime arguments,
generation-qualified static dependencies, a pointer-free layout fingerprint,
execution/fallback path, replay eligibility, persistent argument bytes, and
immutable per-segment counters. Schema v7 also reports deduplicated
provider-generation memory reports without folding requested provider bytes into
Graph-owned persistent bytes. Application code should not read the internal
`Graph._graph_stats` cache.

Each CGraph segment also exposes a `replay_attribution` shape. Production
execution keeps it disabled: no clock reads or counter updates are inserted
into replay. Per-submission measurements belong to explicit ticket telemetry;
private debug instrumentation is not an application contract. GPU durations
from ticket telemetry exclude unmeasured payload work rather than inferring it
from host wall time.

The report distinguishes capture or record, exact replay, patched replay,
recapture, ordinary fallback, structural rejection, transient failure, retry
backoff, capture exceptions, native dispatch, and Vulkan slot saturation.
Reading a report never enables those counters or changes a later execution
path. Counters that were not collected remain zero with
`counters_complete=False`; request submission telemetry for a measured
execution. Reading an ordinary report does not call `ti.sync()`.

Persistent argument bytes are only Forge-visible host/backend argument
storage. They exclude opaque graph executables, command buffers, descriptor
pools, allocator high-water marks, and other driver-retained memory. Use GPU
memory telemetry, host RSS, and graph/tree churn stress alongside the report.

## Numerical and automatic-differentiation contract

Replay changes host submission only; it does not change kernel arithmetic,
dispatch order, or Field dependencies. The release matrix requires exact
direct-versus-Graph equality for integer copy/gather/update without data
races. Normal f32 arithmetic uses `rtol=1e-5`; supported f64 paths use
`rtol=1e-12`. Floating atomic/reduction checks use an explicitly stated
tolerance because backend execution order may differ.

Graph is currently primal-only. Calling `Graph.run()` while `ti.ad.Tape()` or
`ti.ad.FwdMode()` is active or entering raises before submission. Conversely,
automatic AD cannot enter while a Graph host submission is active, and
overlapping runtime-global AD contexts are rejected. These guards add no device
wait. An explicit `kernel.grad` may be dispatched into its own Graph and run
manually outside automatic-AD contexts. Forge does not yet provide an immutable
primal/adjoint Graph pair, reverse dispatch ordering, or native-node gradient
executable contract.

## Offline modified-CompileIQ Graph recipe search

The preferred whole-Graph workflow starts from the frozen semantic definition:

```python
definition = builder.freeze()
target = ti.graph.GraphOptimizationTarget(
    objectives=(
        ("device_time_ns", "min"),
        ("host_time_ns", "min"),
        ("materialized_memory_bytes", "min"),
    ),
)
budget = ti.graph.GraphSearchBudget(
    evaluation_limit=48,
    repeat_count=3,
)
session = definition.search_recipes(
    engine="compileiq",
    target=target,
    budget=budget,
)
decision = session.run(evaluator)
materialized = definition.materialize(decision.selection)
physical_report = materialized.materialization_report()
```

The evaluator receives `(graph, recipe_handle)` and returns the named target
metrics. When requested and not supplied by the evaluator, Forge fills
`materialized_memory_bytes` with known Graph allocations observed after the
evaluator. This is not continuous device/process peak memory. Objectives remain a Pareto
problem inside the modified CompileIQ fork; their declared order only selects
one deterministic result from the measured frontier. Constraints are optional
and explicit. Search/time/memory budgets bound work but do not become compile-
time performance admission gates.

### Report context and optional lifecycle costs

`decision.report.context` retains caller workload/evaluation/backend facts, the
frozen provider registry and Forge compile provenance. Recipe annotations retain
frozen fragment configurations, physical tasks and any declared numerical/component
contracts. These facts explain applicability; they do not independently qualify a
driver/library combination, numerical tolerance or production workload. Reports
created before this enrichment return `None` for `context`.

`report.recipe_discovery` (also `report.context["recipe_discovery"]`) preserves
provider fragment counts, optional provider explanations, rejected composition
attempts and planned-physical duplicates. `catalog.discovery_report()` reads the
same cold-generation observations without rediscovery or a library probe.
Providers may optionally implement `explain_discovery(definition)` to return
JSON-safe facts during discovery; these are provider declarations, not measured
performance. An empty fragment result without an explanation remains unknown:
it can be an assembler-only provider or unmatched Graph semantics. Rejection
counts describe attempts made by this session, including bounded exact probes,
not all possible combinations. Measured failures, Pareto nonselection and budget
incompleteness remain in their existing report sections. None of these diagnostic
fields changes eligibility, the search budget or replay behavior.

The public `definition.search_recipes()` entry accepts optional reporting-only
cost metrics through `GraphEvaluationContract`. For example:

```python
evaluation_contract = ti.graph.GraphEvaluationContract({
    "metric_definitions": {
        "device_us": {
            "unit": "us", "scope": "device_event_elapsed_including_idle_gaps",
            "source": "CUDA events", "interval": "after warmup; 64 replays / 64",
        },
    },
    "correctness": "application-owned reference and tolerance",
    "synchronization": "application-defined completion boundaries",
    "cost_profiles": {
        "lifecycle": {
            "scope": "end-to-end elapsed time for one Graph generation",
            "unit": "ms",
            "setup": "setup_ms",
            "first": "first_ms",
            "steady": "steady_ms",
            "amortization_model": "setup_plus_first_plus_remaining_steady",
        },
    },
})
session = definition.search_recipes(
    target=target, budget=budget, evaluation_contract=evaluation_contract,
)
# evaluator returns target metrics plus its own measured setup_ms, first_ms
# and steady_ms. No timing is inferred from these names.
decision = session.run(evaluator)
```

Optional `metric_definitions` annotate named objectives/constraints with an explicit
`unit`, `scope`, `source` and `interval`; extra JSON facts, such as synchronization
or aggregation, are preserved. This does not add a metric or instrument execution.
JSON and Markdown keep undeclared semantics explicit and do not infer them from
names. CUDA event elapsed time may include idle gaps; it is not active kernel time.
Declare whether active work means a sum or a union when kernels can overlap.
Changing these caller facts changes the evaluation contract for evidence reuse.

Cost-profile units are `s`, `ms`, `us` or `ns`, shared by all phases in one profile. Specify the
actual scope: preparing a binding is not necessarily the full generation setup.
The setup/first/steady mappings may be omitted individually; missing values or
`None` are unavailable, not zero. Supplied durations must be finite and nonnegative.
A declared cost metric is retained as an opaque trial observation, not silently
added to CompileIQ objectives/constraints. If also explicitly targeted, it remains
a normal target metric. Undeclared extra metrics are still rejected.

Amortization is opt-in. Its model is `T(N) = setup + first + (N - 1) * steady` for
`N >= 1`: first replaces one steady execution, and setup/first must not overlap.
Report estimates compare complete feasible baseline/candidate evidence at the
same stage and fidelity. Missing data, nonpositive steady savings and incomparable
evidence do not produce a break-even count. Median estimates and arithmetic bounds
over observed sample extrema are separate; the latter are **not confidence intervals**.
Overlap and single-sample evidence are labeled, not turned into adoption gates.
Independent host/device/end-to-end profiles are never summed automatically.

Markdown also renders provider-owned `preparation_observation` facts already
retained in recipe annotations. FFT observations cover plan creation; SpMM
observations cover preparation that may reuse cached plans. Shared initialization
is not separated, and selected-only restore is not measured. These observations
are not trial metrics, isolated cold-start costs or whole-Graph setup. A missing
baseline observation is unavailable, not zero. Repeated fragments can share plans:
do not sum their times or workspace bytes, or interpret workspace as process VRAM.
They do not populate `cost_profiles` or drive selection/amortization automatically.

JSON preserves original cost observations, including failed trials, and derived
summaries; Markdown uses the same facts. Search-wrapper materialization, evaluator
and cleanup wall times are separate diagnostics, not substitutes for caller-owned
first/steady measurements. The two resource snapshots bracket materialization and
the evaluator; they cannot detect every intermediate allocation or pool reservation.
Reporting adds no probe, synchronization or validation to steady Graph replay and
does not enable a recipe in runtime `auto`.

### Complete-recipe search boundaries

Forge constructs complete recipes from compatible fragments across independent
regions and families until the user evaluation budget is filled. Public recipe
handles expose only semantic identity, selected families, coverage, aggregate
resources, and submission shape. Provider choices and raw tile, block, padding,
lane, workgroup, or PTXAS axes remain private. A partial budget returns a report
with exact measured and missing recipe IDs; a decision is made only from a
complete feasible recipe. The decision does not change runtime `auto`, install
a decision cache, or alter ordinary `GraphBuilder.compile()` behavior.

`ti.graph.compileiq_recipe_search(graph)` remains the lower-level complete-
recipe surface for callers that manage batches, checkpoints, and
materialization contexts directly. The Graph must be backed by a frozen
`GraphDefinition`; the call searches the same whole-Graph catalog as
`definition.search_recipes(...)`. There is no public fallback to the historical
family executable-space adapter. A Graph without a definition is rejected
instead of exposing raw kernel or backend parameters to CompileIQ.

Exact fusion partitions, structured-control routes, scheduling, recording
topology, and provider decisions remain Forge-owned fragment/materializer
details. Their legality boundaries, source groups, and physical choices are
part of recipe identity, and compatible fragments may be composed into one
complete recipe. CompileIQ receives neither those implementation fields nor a
Cartesian product of raw parameters; it receives only the opaque identity of
each already-complete candidate.

Memory staging follows the same rule. The compiler and binding publication
boundary may prove index geometry, halo, shared bytes, layout, ownership, and
alias requirements for a complete materializer, but these facts are not public
CompileIQ axes. Stable replay does not repeat the heavy proof. Materialization
verifies the semantic definition, recipe identity, physical manifest, and
published binding context transactionally; it does not install a runtime
selector or change ordinary `auto` behavior.

The constructor accepts a modified CompileIQ implementation compatible with the
Forge V2 protocol epoch. It verifies the required schemas and API surface, the
self-consistent capability identity, and the bundled-core manifest lock. Forge
also hashes every installed CompileIQ Python source file and binds that identity
to the search session and checkpoint. A source change therefore invalidates
resume/reuse evidence, but does not make an otherwise compatible wheel
uninstallable. Fork commit, platform wheel filename/SHA-256, package version,
core identity, and Python-source identity are qualification provenance, not an
installation allowlist. The current qualified Windows snapshot is
[`fancifulland2718/CompileIQ@f604a79`](https://github.com/fancifulland2718/CompileIQ/commit/f604a79934792a5b17cade81299603fbdb626130);
compatible later fork builds do not require a Taichi source change merely
because their commit or wheel hash differs. Unmodified upstream or an
incompatible protocol raises `CompileIQGraphUnavailableError`. CompileIQ
remains an optional offline tool: the Forge wheel does not depend on it, and
importing `ti.graph` does not import CompileIQ.

The built-in catalog discovers optimization fragments from the frozen
definition. Provider presence is not a claim that every Graph or backend emits
a candidate:

| Provider family | Physical decision represented | Availability boundary |
| --- | --- | --- |
| `map_fusion` | Exact map partition/fusion plan | Only declared compatible map regions. |
| `graph_memory` | Direct versus staged memory plan | Only sources with a complete memory recipe and binding proof. |
| `offload_phase_fusion` | Provider-owned offload phase plan | Only source-published alternatives; no raw task knob. |
| `sparse_traversal` | Sparse traversal/active-set plan | Only source-published complete lifecycle alternatives. |
| `branch_join_schedule` | Branch/fork/join schedule | Only legal dependency DAG alternatives. |
| `recording_partition` | Binding-frontier recording partition | CUDA/source capability and exact frontier identity required. |
| `workspace_concurrency` | Whole-Graph workspace lane/concurrency plan | Only eligible complete-Graph pairs with fixed resource identity. |
| `bounded_execution` | Device-bounded execution plan | Only a source with a complete bounded route on the active backend. |
| `structured_control` | Complete structured-control route | Only source-published control domains; no internal control knob is exposed. |
| `graph_reduction` | Map/partial/finalize phase plan | Only a source that owns a complete reduction recipe. |
| `native_algorithm` | Complete provider-neutral algorithm plan | Only source-published algorithm alternatives; not a provider router. |

`runtime_assembly` is the common assembler, not a searchable optimization
family. An external provider may join through the same versioned descriptor,
fragment, resolution, and materialization contracts. The same provider set must
be supplied again when resolving a persisted selection.

CompileIQ sees fixed opaque ordinal tokens rather than Forge recipe IDs.
GraphMemory, flat control, and nested control use distinct provider namespaces, domain versions, and
semantic identities. Forge decodes the result, checks complete search coverage
including the baseline, and materializes it through the same Graph execution
identity used by explicit recipes. The selected physical control route is
frozen when that Graph is constructed; later internal-selector changes cannot
mutate the compiled identity. A search winner is not runtime admission.
Correctness, exact route and binding identity, lifecycle, memory stability, and
worst-positive performance must be qualified independently. Map-fusion may
then use an exact-scope qualification cache; missing, stale, or mismatched
evidence fails closed to its baseline. Structured-control searches cannot emit
that runtime cache: a winner is available only through explicit offline
reconstruction and does not mutate the runtime `auto` policy. Compile/search
build time is diagnostic only and is not an admission gate.

The complete `graph_memory` domain has been formally qualified on an RTX 5090
with driver 610.62 at matching source, shim, and native commit
`835eea2cb18c49ef66470ae4a378493fb97a0db2`. Each of three scopes used ten fresh
processes, balanced AB/BA order, and five paired blocks of at least 250 ms per
route in every process. Every worker used the reviewed CompileIQ fork to cover
and exactly reconstruct both opaque recipes. For the 16,777,216-item radius-1
and radius-4 scopes, staged/direct median ratios were 0.947679 and 0.965961 and
worst-process ratios were 0.948109 and 0.967921, so both exact scopes passed
worst-positive. The 16,384-item radius-1 scope had a 1.007171 median and a
1.035936 worst process, so it remains a negative and is not admitted.

All 30 workers passed exact correctness, route and binding identity,
source/native provenance, noise, and VRAM-plateau gates. Across two 10,000-replay
waves, the Forge device pool remained at 168,173,696 bytes in two raw chunks and
process-level GPU-memory delta was zero. Large scopes used 1061.66 MiB and the
small scope used 741.660--741.664 MiB. Per-block static shared memory was 520
bytes for radius 1 and 544 bytes for radius 4. These are whole dual-route worker
plateau observations rather than exclusive-VRAM claims for one recipe. The
artifact status is `partially_qualified` with its strict gate passed at
`.agent/experiments/graph-memory-compileiq-835eea2c/qualification.json`, SHA-256
`7d97c2c2bce00b8d42a4e7e9f47888b088fefc77b0d4a91ef9207e7d4bc2e6d6`.
This evidence permits only explicit offline reconstruction of the exact large
scopes above; it does not admit the negative small scope or change runtime
`auto`.

The R12 exact-partition qualification used the same reviewed fork and a
ten-fresh-process balanced AB/BA protocol, with every final route block at
least 250 ms. Candidate/baseline medians were 0.97800 for the 4,096-item
dispatch-sensitive Graph and 0.97232 for the 1,048,576-item bandwidth Graph;
their worst-process ratios were 0.98863 and 0.99153, so both exact scopes
passed the worst-positive gate. The 65,536-item compute-heavy Graph had a
0.98703 median but a 1.01958 worst process and was therefore retained as a
negative rather than admitted. The complete two-task kernel plan was also
retained as a negative at 1.00021 median and 1.01134 worst. All four scopes
passed result, exact-route, and device-memory-pool stability checks. With two
negative scopes, the run did not trigger the three-scope negative-cluster
review threshold. The artifact is
`.agent/experiments/forge-compileiq-r11-r12/qualification.json`, SHA-256
`4674e0774c3ec6574b0a8f6586fdde6412b408290f32302a74af8d978b5c34e5`.

The 2026-08-31 local Windows RTX 5090 qualification used the exact reviewed
modified CompileIQ capability, ten fresh processes per scope, ten balanced
AB/BA blocks per process, and at least 250 ms per final route block. With 4,096
items, 12 actual iterations, and one body action, masked/conditional median
time was 0.88569 for a maximum of 20 iterations (1.129x for masked, positive in
all ten processes) but 2.72953 for a maximum of 128 iterations (conditional
retained). Both scopes passed exact state/result and memory-stability checks.
This is a workload-profile crossover, not a new default or a general speedup
claim. The full local artifact is
`.agent/experiments/structured-control-compileiq-r5/qualification.json`,
SHA-256
`efd53010a68bb896ca6de3b63a83a4a40b1379c9e7817596123ffb8be1a37db7`;
the negative scope remains recorded rather than rolled back.

The same protocol qualified the R6 depth-2 domain with one and two ordered
inner regions. All scopes preserved exact i32 results, reported the requested
physical route, covered both opaque tokens through the exact modified fork,
and remained memory-stable. For steady replay, masked/device-update median
ratios were 2.12995 and 2.13856, so the device-update baseline was retained and
the masked route remains an explicit recorded negative. Cold first-submit
medians nevertheless favored masked at 65.61 versus 74.79 ms and 71.30 versus
83.06 ms; the two-inner scope favored masked in all ten fresh processes.
Persistent allocation also fell from 30,884 to 532 bytes and from 59,996 to
796 bytes. These cold and memory crossovers establish a real second physical
plan, but do not override the steady-performance gate or change runtime
`auto`. Ratio CV reached 0.0744 and one unrelated idle `GameViewer.exe` process
held 16 MiB of GPU memory, so the evidence is deliberately local rather than a
general speedup claim. The artifact is
`.agent/experiments/nested-control-compileiq-r6/qualification-v1.json`, SHA-256
`9514354378bc14562ec000b8a8ac3d5bc8d07acbd7ce19ff7ed184f0da904fca`.

## Performance and memory trade-offs

Graph is most useful when dispatch topology and resource structure stay stable
across many replays, such as fixed-shape simulation substeps, repeated native
primitive chains, and render or staging chains. It is less useful when Python
changes topology every frame, resource structure changes frequently, or one
large kernel dominates launch overhead.

Do not compare graph paths with compile warm-up included. Warm kernels and
graphs first, synchronize at the same measurement boundaries, report median
and tail latency, and record GPU memory before and after long replay and churn
samples. Check results against ordinary dispatch rather than judging only
throughput.

`benchmarks/dynamic_workload_bench.py` compares a device-count-driven payload
through direct dispatch, fixed masked Graph, and `dispatch_bounded()`. The
earlier cross-backend baseline on the Windows qualification machine used
1,048,576 f32 elements, 16 nontrivial payload operations per active element,
and zero/10%/full counts. It produced the following median ratio ranges. A
ratio above one means bounded Graph was faster than the named baseline:

| Backend | direct / bounded | fixed Graph / bounded | Device-known route |
| --- | ---: | ---: | --- |
| CPU | 1.04x-1.43x | 0.988x-0.990x | masked capacity |
| CUDA | 7.01x-7.23x | 0.901x-0.975x | masked capacity |
| Vulkan | 1.53x-1.66x | 0.822x-0.952x | exact indirect, one-to-one range |

The Graph routes substantially reduced direct submission overhead, but the
fixed Graph was faster than bounded dispatch in every measured single-payload
case. This qualification run used the default fixed masked route on CPU/CUDA,
which stayed near the fixed Graph. Vulkan visits only the packet-sized logical
range, but its preparation dispatch and dependency outweighed that saving for
this workload. Use bounded dispatch for its device-known/exact-work contract
or when complete-chain measurements justify it; do not substitute it for a
fixed Graph solely because the active count is sparse. All three runs retained
correct results and non-growing runtime-owned memory; the Vulkan Graph instance
owned one stable 12-byte packet.

CPU was subsequently requalified after its bounded lowering moved from
per-element callbacks to adaptive contiguous JIT chunks and selected exact
scheduling by default. With 262,144 elements and the same 16-operation payload,
exact/fixed-masked p50 ratios at zero/10%/full counts were
6.55x/2.78x/0.997x; the p95 ratios were 6.50x/2.67x/0.999x. Thus sparse work
now benefits materially while full-capacity work stays within one percent of
the fixed Graph in this qualification. Results remained correct, and 1,000
alternating exact replays retained stable runtime, host-pool, and device-pool
ownership. These figures characterize this CPU and workload rather than
promising a universal ratio.

CUDA was subsequently requalified after its default bounded lowering moved to
an exact logical device range while retaining the saturation-capped physical
grid. `dynamic_workload_bench.py --cuda-route compare` builds masked, default
exact, and 12.4+ adaptive Graphs in one runtime and randomizes the measured
variant order against fixed Graph and direct execution. With 4,194,304 items,
16 payload operations, and 10% useful work, default-exact/masked p50 ratios at
zero/10%/full counts were 1.002x/1.022x/0.997x. The adaptive ratios were
0.960x/1.001x/1.029x, showing that the updater has a workload-dependent
crossover. A second large-capacity case with 16,777,216 items, one payload
operation, and 1% useful work measured default-exact/masked ratios of
1.049x/1.040x/1.012x at zero/1%/full counts. All routes produced equal output;
the default exact route used a 16-byte private argument prefix, the adaptive
route added one 32-byte persistent control per payload, and 2,000 alternating
replays retained non-growing runtime memory. These results support defaulting
to logical exactness without defaulting to the 12.4+ updater.

The adaptive route was then requalified for repeated bounded payloads. Two or
more consecutive payloads with the same extent/capacity/block contract now
share one stateful updater; a singleton retains per-node control. With 64
payloads, 16,777,216 items, one operation, and 1% useful work, the grouped
policy was about 1.3x/1.9x/1.04x faster than per-node control at
zero/1%/full count across repeated qualification; a stable full-count rerun
measured 5056 us grouped versus 5420 us per-node. Persistent bounded-control
storage fell from 2,048 to 560 bytes. The benchmark still forces
`device_update`: this optimization does not change the general CUDA default
from logical exactness.

Vulkan standalone bounded consumers now apply the same amortization principle
to packet preparation. Consecutive matching consumers share one prepared
12-byte packet; any intervening action invalidates it. In a 64-consumer,
4,194,304-item, one-operation run, zero/1%/full medians fell from
3.14/3.14/3.17 ms to 1.68/1.70/1.72 ms, while packet storage fell from 768 to
12 bytes. Bounded/fixed ratios recovered from about 0.53-0.54x to 0.97-0.98x.
The long replay qualification stayed within Vulkan's bounded eight-slot
in-flight ownership and retained stable memory.

Recorded worklist finalization now provides the same optimization without a
public launch-state object. When `DeviceWorklistSequence.finalize_next()` is
adjacent to one or more matching bounded consumers, Vulkan lowering gives the
producer one Graph-owned 16-byte packet, publishes count and grid together,
and removes the preparation dispatch. Consecutive consumers reuse that packet;
an intervening action restores the standalone 12-byte packet and prepare path.
`DeviceDispatchState` remains a compatibility adapter for explicit packet
producers such as an existing `DevicePrefixSequence`. CPU ignores that packet
and uses its exact adaptive scheduler. CUDA also ignores it, uses its exact
logical range, and may select 12.4+ adaptive physical control.

The paired `device_worklist_bench.py` qualification used 262,144 items, 10%
active work, four consumers, and four payload operations on the same Windows
Vulkan device. The automatic Graph-owned route measured 470.3 us median versus
480.7 us for the explicit compatibility state, with a paired
compatibility/automatic median ratio of 0.9975. Both paths remained correct and
memory-stable across 1,000 replays. The fixed Graph remained faster at
367.5 us for this light payload, so producer publication fusion removes fixed
overhead; it does not change the workload-dependent exact-versus-fixed
crossover.

`benchmarks/graph_structured_control_bench.py` measures preparation, first run,
steady wall time, control observations, and (where the backend profiler can see
the launches) device kernel time separately. On a local Windows RTX 5090
regression run with 262,144 f32 values and 16 iterations, CUDA native
conditional control had a 464.8 us steady median versus 1,436.6 us for forced
portable replay, a 67.6% reduction (3.09x). Control observations decreased from
17 batches / 204 bytes to 2 batches / 24 bytes. First conditional capture was
20.4 ms and remains preparation cost. The same uninstrumented probe measured
6,513.6 us on CPU host control and 4,375.4 us on Vulkan portable control; these
backend numbers describe the tested execution boundaries, not cross-device
performance promises.

`benchmarks/graph_compound_structured_bench.py` measures a complete ordered
multi-region transaction and reports host enqueue, completion wait, end-to-end
time, runtime waits, and native Vulkan queue-submission counts. On the same
class of Windows device, a warmed 16-region workload with 4,096 f32 values,
a 512-iteration budget per region, and logical termination at iteration 12
measured 55.15 ms median end-to-end with the automatic coarse tail versus
60.96 ms with compact masking for every chunk, a 9.5% reduction. Completion
wait decreased by 14.2% (38.25 ms versus 44.58 ms), while host enqueue remained
effectively unchanged (16.53 ms versus 16.46 ms). Thirty measured invocations
formed thirty transaction queue batches; the focused 72-command regression
also requires exactly one batch for the structured transaction. These are
whole-transaction wall-time and queue-telemetry results. No shader timestamp
is inferred when a Vulkan profiler trace is unavailable.

Against synchronous structured `run()` on the same workload and build,
compound `submit()` reduced median end-to-end time from 56.39 ms to 55.15 ms
(2.2%) and returned host control at 16.53 ms, 70.7% before the synchronous
boundary. Across thirty invocations, native queue-submit calls decreased from
588 to 62 (89.5%). Compound execution still pre-encodes bounded tail command
buffers, so these results demonstrate lower host-control and queue-submit
overhead rather than CUDA-style exact dynamic termination.

The same local Vulkan device was also tested with an eight-chunk,
four-independent-action controller microbenchmark that stopped at logical
iteration 257 of 512 encoded slots. Across 25 warmed samples, single
preparation plus effect-planned barriers reduced median host submit time from
2,168.1 us to 1,359.7 us (37.3%) and submit-plus-wait time from 8,421.1 us to
5,057.8 us (39.9%). A second independent 25-sample trial measured reductions
of 32.2% and 37.4%, respectively. Recorded dependency barriers decreased from
2,561 to 1,017 (60.3%). Known persistent Graph memory stayed at 88 bytes in
both modes; driver-internal Vulkan memory remained unavailable. Use
`--independent-actions 3`, `--compound-preparation`, and `--barrier-policy` on
the benchmark to reproduce the A/B boundary.

Current Dense Field multi-block throughput, compile scaling, cache, RSS/VRAM,
and the Graph/AD guard microbenchmark are reported in
[Dense Field Graph](dense_field_graph.en.md). They are local regression evidence,
not portable performance promises; a relative trial range above 5% remains
observational.

## Native and AOT boundary

Graphs may contain native nodes produced by Forge's own DSL/native algorithm
layer. Arbitrary user native callbacks are not supported. AOT serialization
through `ti.aot.Module.add_graph()` accepts ordinary kernel CGraphs only, not
graphs containing Forge native nodes. Ordinary CGraphs and recordable providers
may fuse into one backend region on the same active backend; every node must
still match the runtime/backend where it was compiled, so this is not
cross-device execution. Numeric-check result nodes replay device work; reading
a result remains explicit.

See [Native algorithms](native_algorithms.en.md) for primitive ownership and
result APIs.

## Validation and platform status

The focused validation set includes:

- `tests/python/test_graph_native_algorithm_recipe.py` for retained integer scan
  recording, padding, modular arithmetic and independence from Program arena cleanup;
- `tests/python/test_cuda_graph_binding_frames.py` for immutable frames, queued
  event reuse, close/reset and allocation lifetime;

- `tests/python/test_graph.py` for public contracts, lifetime, replay,
  structured control, and diagnostics;
- `tests/python/test_graph_iterative_qualification.py` for f32 PCG and
  nonsymmetric BiCGSTAB over the generic structured/provider contracts;
- `benchmarks/graph_structured_control_bench.py` for structured-control
  preparation, steady wall time, observation traffic, and kernel timing;
- `tests/python/test_graph_dense_field.py` for static Field binding, SNodeTree
  generation/lifetime, zero-argument replay, mixed segments, and concurrency;
- `tests/python/test_graph_dense_field_numerics.py` for integer exactness,
  f32/f64 tolerance, AOS/SOA layouts, multiple trees, primal-only AD rejection,
  and explicit grad-kernel Graphs;
- `benchmarks/graph_dense_field_multiblock_bench.py` for fresh-process
  1/2/4/8-block compilation, cache, throughput, fairness, RSS/VRAM, display,
  and reset reports;
- `tests/python/cuda_graph_runtime_bench.py` and
  `tests/python/cuda_graph_dynamic_patch_bench.py` for CUDA replay;
- `tests/python/vulkan_graph_slot_bench.py` and
  `tests/python/vulkan_graph_retirement_stress.py` for Vulkan capacity and
  lifecycle;
- `tests/python/backend_async_runtime_stress.py` for CPU/CUDA/Vulkan
  cross-thread submission;
- `tests/python/ggui_vulkan_queue_concurrency_stress.py` for asynchronous
  producer and display submission;
- backend feature-split builds and native C++ safety tests.

Windows validation covers CPU, CUDA, and Vulkan runtime paths. Linux compiler
branches were kept platform-neutral, but real Linux build, driver, window
system, and long-stress results must be rechecked before making a Linux release
claim. In particular, dense Field Graph still needs GCC/Clang builds, Linux CPU
multi-block runs, CUDA Driver-only/Toolkit-OFF zero-argument capture, Vulkan
validation/headless/headed replay, sanitizer coverage, and allocator-specific
RSS/VRAM/reset measurements. See
[Linux revalidation status](linux_revalidation.en.md) for the exact remaining
matrix.

The recipe-composition, SpMM ownership/reporting, retained-scan and event-retention
closeout was checked on Windows with Python 3.10 and a matching local native
build. This does not requalify Linux, every Python wheel, vendor-version combination
or production workload. CompileIQ compatibility is capability/protocol based, not
an exact commit pin; checkpoint measurement reuse still requires matching contracts.

## Related documents

- [Dense Field Graph](dense_field_graph.en.md)
- [Graph compatibility and migration guide](graph_migration_guide.en.md)
- [Forge API reference](forge_api_reference.en.md)
- [Native algorithms](native_algorithms.en.md)
- [Compilation and advanced-optimization trade-offs](compilation_tradeoffs.en.md)
- [Linux revalidation status](linux_revalidation.en.md)
