# Graph Compatibility and Migration Guide

For the `0.6.3` complete-recipe workflow, see the
[integration and external provider guide](graph_recipe_integration.en.md).
Explicit offline freeze/search/resolve does not change ordinary compile/run or runtime auto.

> The base Graph modernization and native replay model shipped in `0.4.1`.
> Dense Field Graph, replay lifetime hardening, diagnostics, and stricter
> concurrency/AD contracts are `0.5.0` work. See
> [release notes](release_notes.en.md).
> Version `0.6.0` adds structured control, asynchronous tickets and pacing,
> runtime dense-storage binding, recordable operator actions, and Vulkan
> device-written indirect dispatch.

This document describes the public behavior and compatibility boundary of Forge
graph support relative to vanilla Taichi 1.7.4. Backend architecture,
performance, memory policy, and diagnostics are maintained separately in
[Graph runtime and optimization](graph_runtime_optimization.en.md).

For exact signatures of Forge-only graph and native replay APIs, see
[Forge API reference](forge_api_reference.en.md).

## Public compatibility

Forge keeps the familiar graph-builder surface:

- `ti.graph.GraphBuilder`
- `GraphBuilder.dispatch(kernel, *args, template_args=None, label=None)`
- `GraphBuilder.create_sequential()`
- `GraphBuilder.append(sequential)`
- `GraphBuilder.compile()`
- `Graph.run(args)`
- `ti.graph.Arg(...)`

Ordinary kernel graphs continue to use the public CGraph model. AOT graph
serialization for ordinary kernel-dispatch graphs remains compatible with the
existing public model.

## What Forge adds

Forge keeps the graph-builder model and adds explicit Forge extensions plus
backend-owned execution planning for structured control, asynchronous
submission, device-driven dispatch, and DSL-defined native replay.

The user-visible additions are:

- `GraphBuilder.dispatch()` and `Sequential.dispatch()` can use the
  keyword-only `template_args=` parameter to bind a data-oriented `self`, a
  Field, or another definition-time argument;
- optional per-dispatch `label=` values and the thread-local
  `ti.profiler.dispatch_label()` scope correlate stable task identities with
  profiler/NVTX events;
- read-only `kernel.task_manifest()` and one-segment JIT
  `Graph.task_manifest()` reports expose requested/selected/actual geometry
  without adding launch control;
- runtime argument handling for scalar, matrix, ndarray, texture, and RW
  texture paths;
- runtime-storage binding for compatible dense Fields, `DenseNdarrayView`, and
  managed external storage;
- `GraphBuilder.while_loop()`, `if_then_else()`, `switch()`, and structured
  `Sequential` composition to depth two;
- Vulkan device-written single-task execution through
  `GraphBuilder.dispatch_indirect()` / `Sequential.dispatch_indirect()`; CPU
  and CUDA currently fail explicitly;
- `Graph.submit()`, `SubmissionTicket`, opt-in region telemetry, and shared
  `SubmissionPacer` admission control;
- stable ordinary-dispatch fallback when an optimized replay path is not
  supported;
- public `Graph.execution_stats()` execution/fallback/resource reports;
- stop-position and branch reports through `Graph.control_flow_stats()` and
  `Graph.run(trace=True)`;
- explicit `kernel.grad` dispatch for manually managed gradient Graphs outside
  automatic-AD contexts;
- internal native replay for Forge-defined primitive sequences produced by the
  algorithm layer;
- `GraphBuilder.append_native(node, prewarm=False)` for Forge DSL-defined
  nodes such as `PrimitiveSequence`, `DeviceCheckResult`, and
  `DeviceMetricResult`;
- qualified recordable provider actions such as `LinearOperator.graph_action()`
  in a Graph root or structured body, with adjacent ordinary CGraphs eligible
  for same-backend region fusion.

## Definition and lifetime contracts

- `GraphBuilder.compile()` freezes the current dispatch and sequential
  definition. Later builder changes do not change an already compiled graph or
  its lazy AOT result.
- `Graph.run(args)` requires argument keys to exactly match the declared
  runtime arguments. Missing and unexpected keys raise `TaichiRuntimeError`
  before submission.
- One call on one `Graph` is a complete host invocation. Independent graphs
  remain independently submitable; this protection does not wait for GPU
  completion or add a default `ti.sync()`.
- `ti.reset()` invalidates graphs from the old runtime. Rebuild the builder
  and graph after reset.
- Closed-over or `template_args` dense Fields are definition-time bindings.
  Their contents may change, but replacing identity/layout or destroying their
  generation-qualified SNodeTree requires rebuilding the Graph. Use an
  `ArgKind.NDARRAY` runtime slot to replace compatible dense Fields between
  runs. Numeric tree-id reuse does not revive a static binding.
- Same-structure runtime resources may use backend replay; structural changes
  may recapture or fall back. Both paths preserve binding and execution
  semantics.
- Runtime graph safety does not replace an application-level snapshot, slot,
  or producer-consumer protocol for shared simulation and rendering data.
- New engine code should bind a `self`, Field, or another `ti.template()`
  argument through `template_args=`; the Field still does not enter each run's
  dictionary. Legacy adapters that write directly to the durable AOT plan keep
  exact runtime-argument discovery and strict missing/unexpected-key checks.
- `Graph.run()` is primal-only. Active `ti.ad.Tape()` and
  `ti.ad.FwdMode()` are rejected before submission instead of silently losing
  gradient/dual recording. An explicit `kernel.grad` Graph may be run manually
  outside those contexts.

The implementation details behind CUDA resource leases and dynamic patching,
Vulkan identity and deferred retirement, fixed replay capacity, failure
recovery, and `Graph.execution_stats()` are documented in
[Graph runtime and optimization](graph_runtime_optimization.en.md).
Dense Field users should also read
[Dense Field Graph](dense_field_graph.en.md) for the static-binding contract.

## Strict runtime-key migration

Definition-time template bindings must be removed from the runtime dictionary.
Code that kept passing them may have appeared to work with a permissive legacy
adapter, but Forge now rejects the extra key:

```python
builder.dispatch(
    solver.step,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()

graph.run({"state": solver.state})  # wrong: unexpected runtime argument
graph.run({})                       # correct for this zero-runtime-arg graph
```

Keep only declared `ti.graph.Arg` names in `Graph.run()`. This catches stale
engine adapters and misspelled parameters before backend submission.

## Heterogeneous multi-environment layout

Do not create one Graph per homogeneous environment. Use one block/Graph per
stable solver-layout-shape-feature signature and store homogeneous environments
on a leading Field axis. Separate heterogeneous blocks keep independent
writable Fields and submit independently. Simulation/render overlap still
requires slot/epoch-owned snapshot Fields; Graph does not add data-hazard
tracking.

## Native graph boundary

Native graph support is intentionally narrow:

- Supported: native nodes produced by Forge's own DSL/native algorithm layer.
- Not supported: arbitrary user native callbacks inside graph.
- Not supported: AOT serialization for graphs containing Forge native nodes;
  `ti.aot.Module.add_graph()` accepts ordinary kernel CGraphs only.
- Same-backend ordinary CGraphs and recordable providers may fuse. Every node
  must match the active runtime/backend; this does not provide cross-device
  execution.
- Numeric-check result nodes replay only device-side native work. Result reads
  remain explicit through `to_int()`, `to_bool()`, `ok()`, or
  `to_float()`.

This keeps resource ownership and backend lifetime rules explicit.

## Suitable workloads

Graph is most useful when dispatch topology is stable and replayed many times:

- fixed-shape simulation substeps;
- repeated native primitive chains;
- rendering or staging chains with stable resource structure;
- AOT deployments that need a named graph entry point.

Graph is less useful when Python changes dispatch topology every frame, shapes
or resource structure change frequently, or one large kernel dominates launch
overhead.

## Difference from vanilla

Vanilla Taichi 1.7.4 primarily records ordinary kernel dispatches. Forge keeps
that model, then adds backend planning and native primitive replay under the
same public entry point. Unsupported optimized paths fall back instead of
changing graph semantics.
