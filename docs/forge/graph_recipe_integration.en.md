# Complete Graph recipes: downstream integration

[中文](graph_recipe_integration.zh.md)

Use a compatible Windows `taichi-forge` CPython shim and `taichi-forge-runtime` pair.
Package/private ABI compatibility, not equal Git commits, determines pairing. A headless
development artifact is suitable for simulation but does not provide GGUI windows.
See [wheel builds](build_wheels.en.md) and [external dependencies](external_hardware_providers.en.md).

Install the compatible Windows wheel supplied by the
[maintained CompileIQ fork](https://github.com/fancifulland2718/CompileIQ), not the base package
obtained by an unqualified `pip install compileiq`. Fork V2 protocol/capability compatibility
is required; a commit/hash is provenance, not an installation allowlist. Optional vendor
libraries or compilers are needed only for the providers being used.

Keep the shim, runtime and fork wheels together in the application's test
environment; install the supplied files explicitly with
`python -m pip install <runtime.whl> <shim.whl> <fork.whl>`.
Check actual import locations before running the example. Do not carry development
`PYTHONPATH`, `TAICHI_NATIVE_RUNTIME_DIR` or `TAICHI_RUNTIME_DIR` overrides into an
installed-wheel test. A local headless integration set is not a public release or
a rendering-window qualification; windowed tests require a GGUI-enabled runtime.

## Executable example

[complete_recipe_provider.py](../../python/taichi_forge/examples/graph/complete_recipe_provider.py)
implements a complete external provider using public APIs. It replaces two integer passes
with one without changing Forge's family registry or CompileIQ.

```powershell
python -m taichi_forge.examples.graph.complete_recipe_provider --output result --environment-id my-device-driver-runtime
python -m taichi_forge.examples.graph.complete_recipe_provider --output restored --restore result/selection.json --environment-id my-device-driver-runtime
```

Use `--evaluation-limit 2` to produce a partial search, then a larger budget with
`--resume result/checkpoint.json`. The example measures synchronized wall time, not device
time or application acceleration. Supply a truthful, stable environment description.

- `selected`: save the selection artifact; use `with definition.materialize(selection) as handle`,
  then `handle.executor.bind/run`. The handle owns materialization lifetime.
- `resumable`: save the report/checkpoint; recreate the same provider, workload, evaluation,
  environment and target contracts before resuming.
- `no_feasible_candidate` / `failed`: inspect structured failures. Passing a missing selection
  to `materialize(None)` requests the baseline; it is not optimization success.
- `definition.compile()` explicitly chooses baseline. Search does not change runtime auto.

Omitting any of GraphWorkloadContext, GraphEvaluationContract and GraphBackendEnvironment
limits measurement reuse to the current session. Save vendor operation preparation artifacts
when required as well as the selection. Recreate equivalent definitions/providers in a new
process, check applicability, then resolve. Structural reuse may succeed while measurements
need renewal. Neither Python executable deserialization nor AOT binary reuse is implied.

## Provider ownership

The example's descriptor owns stable namespace, versions, semantic fingerprint and assembly
protocol. `discover` recognizes only its known operation; `resolve` reconstructs by stable key;
`expand` returns real survivor neighbors or an empty sequence. `materialize` enrolls owned
resources in `scope.own(..., release=...)` for rollback. `assemble` returns an executor and
actual physical observation. `describe` supplies JSON-safe claims, not measured conclusions.

`PROVIDER_OWNED_WHOLE_GRAPH_V1` requires complete semantic coverage. It does not allow arbitrary
Python callbacks inside ordinary Graphs. Existing Forge region providers contribute through
`RUNTIME_GRAPH_ASSEMBLY_V1` and their owner-specific cold materializers; new applications need
not edit private source/environment routing tables to use the whole-Graph protocol.

`CompiledGraphPhysicalManifest.from_graph(definition, recipe, graph)` observes an actual
compiled Forge Graph at materialization, not at replay. It does not prove mathematical
equivalence. Providers must declare real coverage, binding, numerical and resource contracts;
change domain/implementation identity when physical work changes. The example owns no scratch;
a provider with scratch must report and retire it instead of declaring zero storage.

## Execution identity and memory observations

Physical manifest schema v2 separates the execution/allocation plan from memory observations.
`materialized_physical_id` hashes compiled work, bindings and `resource_plan` (requested sizes,
grouping and lifetime), not cold/warm cache allocations or backing-page sizes. `resources` and
`memory` retain the observation. Earlier v1 physical IDs are not interchangeable with v2;
resolve the structural selection again and renew measurement evidence when required.

`handle.resource_instance_id` identifies a live ownership instance, not a portable selection.
Equal physical IDs permit candidate comparison, not sharing mutable executors. Cross-recipe
instance sharing is disabled unless the provider explicitly returns
`GraphMaterializationProduct(..., shareable_executor=True)` and guarantees safe shared state.
Repeated requests for the same recipe within one context still reuse that context's instance.

## Evaluation boundaries

Restore equivalent input state for every evaluation, bind once, warm up, then measure.
Keep correctness readback, input restoration and library discovery out of steady submission
timings unless they genuinely belong to the measured application workflow on both sides.
Feedback matmul, in-place sort and destructive C2R need explicit state management; Forge does
not copy every input automatically.

Use existing metric_definitions to distinguish device event intervals, active kernels, host
submission and completion waits. Events may include idle gaps; summed overlapping kernels
are not wall time. Existing cost_profiles separate setup/first/steady. Caller/workspace
requests, pool reservation and unknown driver/vendor residency are different quantities.
Unknown is not zero. Nsight/NVML are opt-in diagnostics, not fixed replay checks or gates.
Operation-specific Graph and search boundaries remain in the external hardware guide.
