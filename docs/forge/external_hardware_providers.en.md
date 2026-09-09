# Optional External Hardware Providers

[中文版本](external_hardware_providers.zh.md)

Taichi Forge keeps its official runtime wheels driver-only. Optional CUDA or
vendor libraries are installed and versioned by the application, are loaded
only when explicitly probed or used, and never add a `cu12`/`cu13` Forge wheel
variant. This guide explains that user-managed boundary and gives recommended
configuration for the optional libraries most relevant to simulation and
rendering.

This page is an installation and deployment guide. Installing a library alone
does not select an execution route. Forge exposes explicit retained-provider
APIs for the bounded operations below; discovery probes remain non-executing.

## Support status and call boundary

| Library | Forge status | Installation owner | Forge discovery | Call position |
| --- | --- | --- | --- | --- |
| cuBLAS | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cublas")` | Direct Python or root Graph; not kernel-callable |
| cuSPARSE | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cusparse")` | Domain auto/explicit or root Graph; not kernel-callable |
| cuFFT | Registered D1 provider | User CUDA environment | `ti.hardware.probe("cufft")` | Explicit plan or root Graph; not kernel-callable |
| VkFFT 1.3.4 | Optional ABI1 Vulkan JIT adapter | Current runtime build configuration; older artifacts may omit it | `ti.hardware.probe("vkfft")` or explicit library path | Fixed-storage plan/root Graph; explicit batch and whole-Graph secondary recipes with matching extensions |
| cuDSS 0.8.x | Registered bundled-adapter ABI | Forge adapter; user vendor runtime | `ti.hardware.probe("cudss", library_path=...)` | Domain auto/explicit or root Graph; not kernel-callable |
| OptiX ABI 93/105/118 | Registered bundled-adapter ABI | Forge adapter; user/driver vendor runtime | `ti.hardware.probe("optix", library_path=...)` | Explicit scene/launch or root Graph; not kernel-callable |
| Vulkan driver/ICD | D0 backend dependency, not a D1 provider | OS/GPU driver installation | `ti.init(arch=ti.vulkan)` plus capability queries | Kernel and documented native Vulkan APIs |
| cuSPARSELt 0.8.x-0.9.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.tensor.CusparseLtProvider` / `ti.linalg.record_sparse_matmul` | Retained FP16 2:4 capture and complete shared-A matmul recipes; no kernel intrinsic or automatic rewrite |
| cuTENSOR 2.0.x-2.7.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.tensor.CutensorProvider` / `ti.linalg.record_contraction` | Retained root Graph capture and complete contraction dataflows; no kernel intrinsic or implicit auto rewrite |
| AmgX stable C API | Registered bundled-adapter ABI | Forge adapter; user source build | `ti.hardware.probe(...)` or `ti.hardware.linalg.AmgxProvider` | Host CSR topology, host/device values and vectors; no Graph/kernel/auto route |
| NCCL | Outside Forge's current single-GPU scope | User system package | No public Forge probe or execution API | External multi-GPU communication only |

Registered external providers appear in `ti.hardware.providers()`. Their probe audits a
bounded version family and execution-symbol surface, but still creates no plan
and qualifies no workload. Execution starts only through the documented domain
or explicit provider/plan API. NCCL remains unregistered.

None of these host libraries can be called from inside `@ti.kernel`. Automatic
use is currently limited to documented domain APIs such as qualified
cuSPARSE SpMV and cuDSS solver selection. Installing cuSPARSELt, cuTENSOR,
AmgX, or NCCL never causes compiler rewriting.

### Recording and complete-recipe search are separate capabilities

The following table describes the current source API, not qualification of every
vendor release, driver, GPU, or workload. A library being executable or recordable
does not imply that its algorithms are exposed as CompileIQ search axes.

`ti.hardware.capability(operation_id).to_dict()["recipe_search"]` reports static
semantic/provider entry points and their narrower scope, independently of expert
execution and `graph_integration`. It does not load the optional library or certify
the current workload. `no_builtin_entry_declared` means no built-in complete-recipe
entry is declared for that operation; it does not prohibit application providers.

| Operation | Semantic entry and preparation | Graph and search boundary |
| --- | --- | --- |
| Fixed-pattern sparse-dense product | `SparseMatrix.record_spmm(...)`, then `operation.prepare(input_array, output_array)` | CUDA f32 CSR / compact row-major dense arrays; append the operation with `GraphBuilder.append_native()`. Explicit `ti.hardware.linalg.SparseSpmmRecipeProvider()` adds frozen direct/preprocessed strategies to complete recipes. |
| Batched 2D complex FFT | `ti.linalg.record_fft(...)`, then `operation.prepare()` | CUDA complex-f32, compact arrays `(H, W, 2)` or `(batch, H, W, 2)`, distinct input/output. Explicit `ti.hardware.fft.FftRecipeProvider()` adds separable per-image and, on capable runtimes, cross-batch column plans alongside the whole-transform baseline. |
| Driver-native segmented scan | `GraphBuilder.segmented_scan()` and default recipe providers | Fixed disjoint i32/u32 arrays and immutable segments. Global correction uses retained CUDA recording and Graph-bound scratch; no external Toolkit library is required. It remains a fixed-resource action, not a binding-frame region. |
| Toolkit reset-monoid segmented scan | Existing `GraphBuilder.segmented_scan()` plus `CubSegmentedScanRecipeProvider(manifest_path)` from `taichi_forge.hardware.source_providers` | Optional source-provider addon; bounded i32/u32 sum and immutable segmented layout. Prepared capture, workspace and head-bitset lifetime form the physical recipe; the addon is not part of the portable runtime wheel. |
| Other cuSPARSE / cuFFT / cuDSS expert operations | Existing explicit plans and documented root Graph recording | Recording alone does not provide a recipe generator. cuDSS root ordering must not be described as CUDA Graph capture. |
| Shared-pattern sparse-solve region | `ti.linalg.record_sparse_solve(...)`, then `operation.prepare()` | Explicit `ti.hardware.linalg.SparseSolveRecipeProvider()` searches complete ordering/factor lifecycles with Graph-owned capture; separate from legacy root-ordered cuDSS recording. |
| Vulkan VkFFT | Fixed-storage plan/root Graph; explicit `VulkanFftRecipeProvider` | Batch scratch reuse plus Vulkan immutable secondary Graph recording; not CUDA binding frames or a vendor route axis. |
| cuBLASLt matmul region | `ti.linalg.record_matmul(...)`, then `operation.prepare()` | CUDA compact scalar-f32, fixed shape and optional strided batch. Explicit `ti.hardware.linalg.MatmulRecipeProvider()` composes frozen algorithm/workspace choices, real operand packing, and separate/fused ReLU. The expert retained-plan API remains private. |
| cuTENSOR contraction region | `ti.linalg.record_contraction(...)`, then `operation.prepare()` | Explicit `ti.hardware.tensor.ContractionRecipeProvider()` composes real input permutations and vendor/separate epilogues; includes retained workspace and immutable binding frames. |
| cuSPARSELt shared-A region | `ti.linalg.record_sparse_matmul(...)`, then `operation.prepare()` | Explicit `ti.hardware.tensor.SparseMatmulRecipeProvider()` searches frozen algorithm/resource/epilogue dataflows; current A is compressed once per invocation, not cached across replays. |
| AmgX | Explicit provider plans described below | No complete-recipe provider or general Graph recording route is currently exposed. |

Prepare mathematical operations before freezing the Graph. SpMM, FFT, matmul and contraction require
explicit finite-input / f32 tolerance contracts; Forge does not scan values on
each replay. By default FFT forward and inverse are both unnormalized, so applying both
multiplies the input by `H * W`. Layout, precision and normalization are semantic
requirements, not optimizer choices. Vendor internals not exposed by the library
are reported as unknown, not fabricated kernel counts.

### Matmul preparation and reuse

Matmul semantics are `D = activation(alpha * op(A) @ op(B) + beta * D)`.
`activation` is `"identity"` or `"relu"`; transpose flags, coefficients, dtype,
shape and caller-qualified tolerances are semantic facts, not search axes.
Inputs may change every replay. Output must be distinct from both inputs;
read/read alias is allowed. The tolerance declaration is not an automatic
accuracy guarantee: the evaluator/downstream application validates its values.

```python
operation = ti.linalg.record_matmul(
    512, 512, 512, transpose_a=True, activation="relu",
    absolute_tolerance=2e-5, relative_tolerance=2e-5,
)
operation.prepare(workspace_limit_bytes=32 << 20, heuristic_limit=4)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (*ti.graph.default_recipe_providers(),
             ti.hardware.linalg.MatmulRecipeProvider())
```

Pass this provider set to `definition.search_recipes()` with the normal workload,
evaluation and backend contracts; CompileIQ schedules only complete recipe IDs.
`definition.compile()` materializes the baseline. This semantic description must
be **frozen before execution**; it is not a directly executable expert plan.
Preparation queries metadata and freezes documented algorithm configurations,
not opaque native bytes. It allocates no candidate GPU workspace and executes no
matmul. Materialization reconstructs only the requested plan and its exact
workspace, without rerunning the heuristic. Actual transpose packing, when
selected, refreshes private input buffers on every replay; no constant-input
assumption or hidden value cache is introduced. Equivalent descriptor spelling
alone is not a layout candidate. Fused activation has no externally visible
intermediate output; use separate semantic operations when that output is needed.

Save `operation.preparation_artifact()` with `decision.selection_artifact`.
In another process, construct the same `record_matmul(..., preparation=saved)`
and equivalent Graph, then `resolve_recipe(saved_selection, providers=providers)`
and `materialize(selection)`. Imported setup timings remain historical facts.
Device/component/configuration drift rejects reuse at cold boundaries rather than
silently selecting another algorithm. Closing the operation does not invalidate
plans retained by live Graphs. Fixed bindings are checked at `Graph.bind()`;
immutable binding replay adds no matmul-specific scan or provider validation.

Capable native runtimes also compose these regions with immutable argument-frame
recipes. Older runtimes with typed matmul capture but without frame support omit
that composition. cuBLASLt and its transitive libraries remain user-managed
(`TI_CUBLASLT_LIBRARY_PATH`); ordinary kernels, runtime auto and wheel dependency
profiles do not change. Exact workspace/private-buffer bytes and argument images
are reported separately from unknown vendor/driver storage. Neither fusion nor
packing is a universal winner; host, device and memory evidence remain separate.

The shared CUDA operand-packing helper uses a padded warp-contiguous tile for
wide axes and retains its compact tile for narrow axes, selected from frozen
shape facts. Matmul and contraction reuse this lowering without a new vendor
dependency or public block/tile axis. Its implementation identity is part of
preparation provenance: old packing observations are not silently reused after
a lowering change. Larger per-block shared storage is not additional persistent
Graph workspace, and packing gains are not a claim of end-to-end GEMM speedup.

### FFT and SpMM preparation and reuse

Both separable FFT strategies transform all rows first and use the output array
for in-place columns, without a dense transpose buffer. The per-image plan then
executes columns once per image; the cross-batch plan executes once per column,
batching independent images within each call. This changes physical launch and
memory-access organization, not mathematical normalization or layout. Neither is
universally faster, and whole-transform can outperform both. `prepare()` records
actual workspace and setup facts; the expanded FFT provider domain invalidates
older provider-bound search evidence, not compatible wheels. Older native
runtimes omit the additional strategy; an imported selection needing it fails
explicitly at materialization instead of substituting another plan.

FFT Graph recordings retain only their own physical plan, not the search
operation's entire plan collection. `operation.close()` releases the operation's
preparation ownership and prevents further calls to its `prepare()` or `compile()`;
already-built Graphs keep their plan leases. An unused plan can retire once its
last execution owner is gone. Frozen recipe metadata remains available, and a
later materialization recreates only the requested retired plan, checking its
component and workspace against the prepared facts at that cold boundary.
Frozen definitions retain FFT descriptions rather than baseline plan leases.
Releasing the search operation and builder therefore permits selected-only plan
residency; any other live Graph still legitimately retains its own plan. Baseline
`definition.compile()` reacquires a plan at compile time, never during replay.
For plan-free cross-process restoration, save `operation.preparation_artifact()`
alongside the search decision's selection artifact. Recreate the same
`record_fft(...)` operation with `preparation=saved_preparation`, append it to the
equivalent builder, and freeze. Then use `definition.resolve_recipe(...)` and
`definition.materialize(...)` normally. Do **not** call `prepare()` on this path:
it explicitly prepares all imported candidates. Freeze, catalog discovery and
selection resolution create no FFT plans; materialization creates only the
requested plan. The new native capture-description capability is required; older
compatible runtimes keep ordinary FFT support but reject this restoration path.

FFT output scaling is an explicit mathematical contract, not a library route:

```python
operation = ti.linalg.record_fft(
    (1024, 1024), batch_count=2, direction="inverse",
    output_scale=1 / (1024 * 1024),
    absolute_tolerance=2e-6, relative_tolerance=3e-5,
)
operation.prepare(
    lto_callbacks=True,
    nvrtc_library=nvrtc_path,       # caller-provided absolute library paths
    nvjitlink_library=nvjitlink_path,
)
```

The default `output_scale=1` preserves unnormalized FFT behavior. Other finite
f32 scales describe the same output for every candidate: the existing plans
append a Forge scaling kernel, while the optional whole-transform LTO candidate
fuses scaling into cuFFT stores. Add the existing `FftRecipeProvider` to search
these complete recipes; preparation alone does not select or enable runtime auto.
This remains compact, out-of-place, batched 2D complex-f32. General load/store
callbacks, arbitrary callback code and mutable callerInfo state are not exposed.

Real transforms use the same semantic entry:

```python
forward = ti.linalg.record_fft(
    (height, width), transform="r2c", input="signal", output="spectrum",
    absolute_tolerance=1e-4, relative_tolerance=1e-4,
)
inverse = ti.linalg.record_fft(
    (height, width), transform="c2r", input="spectrum", output="reconstructed",
    output_scale=1 / (height * width),
    absolute_tolerance=1e-4, relative_tolerance=1e-4,
)
```

R2C maps scalar f32 `(H,W)` to an interleaved half-spectrum `(H,W//2+1,2)`;
C2R reverses those shapes. Prepend the batch axis when `batch_count > 1`. The
last width can be odd or even. C2R requires a valid Hermitian spectrum, including
real self-conjugate bins. **C2R overwrites its input even out of place**; its
recording declares that write at freeze time. Replenish the spectrum before
each replay, normally through an upstream FFT/producer. There is no implicit
preservation copy or per-replay symmetry check. The default is unnormalized;
the scale above normalizes the inverse.

Real FFTs support plan-free preparation artifacts, complete Graph search and
immutable binding frames. They currently use only the whole-transform FFT plan:
the enclosing Graph execution strategies are searchable, but C2C decomposition
and LTO callback candidates do not apply. `prepare()` records only applicable
plans, and `prepare(lto_callbacks=True)` rejects real transforms before compiler
loading. This does not change the distinct expert `CufftPlanND` scope or enable
runtime auto. Input mutation follows the [cuFFT data-layout contract](https://docs.nvidia.com/cuda/cufft/index.html#data-layout).

Ordinary scaling needs no NVRTC/nvJitLink. The optional LTO candidate requires
compatible external cuFFT/NVRTC/nvJitLink runtimes and native callback-plan support;
none is added as a portable-wheel shared dependency. Windows dynamic cuFFT
supports LTO callbacks, unlike the legacy static-library callback route. Follow
[NVIDIA's callback compatibility rules](https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines)
and keep cuFFT's transitive dependencies discoverable in the process library path.
The explicit paths identify the supplied compiler/linker, not a guarantee that
all CUDA combinations have been qualified. Missing dependencies or preparation
failure do not silently remove scaling or replace the requested callback with
plain FFT. Existing non-callback candidates remain usable.

Preparation artifacts include callback-source/LTO identity and compiler/linker
facts. Freeze and resolve deserialize neither executable code nor a vendor plan;
selected callback materialization recompiles the controlled source and checks
those facts at that cold boundary. Replay performs no compilation, library probe,
callerInfo update or added synchronization. There is no extra full-size buffer;
vendor workspace is reported per plan, while opaque driver/module residency is
not implied to be zero. JIT preparation time is a cost, not a performance gate.

SpMM uses the same JSON preparation/selection workflow via
`matrix.record_spmm(..., preparation=saved_preparation)`. With native plan-lease
support, `operation.close()` releases only search-owned plans; equal matrix/RHS/
algorithm plans remain shared with live Graph commands and pre-existing expert
caches. Frozen definitions retain matrix identity and expected facts, not a
baseline plan. Unlike FFT, SpMM needs real dense bindings to create its selected
plan: freeze, resolve and materialize create none; prepared-frame binding (or the
ordinary executor's first native preparation) creates and checks only that plan.
Importing preparation facts does not measure restoration time. Reported workspace
is plan-owned allocation, not total driver residency or device peak. Older
runtimes retain the legacy matrix cache and reject selected-only artifact import.

Preparation artifacts contain expected JSON facts, not Python executables or
vendor binaries. Restoration checks the semantic/device/component contract;
materialization checks the actual selected plan's component and workspace again.
Library discovery is explicit at this cold boundary. Report annotations label
imported preparation observations as historical and separately record observed
plan recreation. A matching library name/version is not proof of binary identity
or production performance, and the artifact is not an AOT or binary cache.

These providers are opt-in additions alongside `ti.graph.default_recipe_providers()`
at `definition.search_recipes(engine="compileiq", providers=..., ...)`. Supplying
a provider without a matching prepared semantic region does not invent one.
The maintained CompileIQ fork schedules only opaque complete-recipe identities;
Forge owns composition, frozen physical configuration and materialization. A
failed plan reconstruction is not silently replaced with another vendor heuristic.
Installing a library or selecting a measured recipe does not change runtime auto.

Prepared FFT and SpMM region strategies can also compose with the default CUDA
immutable binding-frame executor. Computation regions still have exclusive
semantic coverage; the executor wraps the assembled computation rather than
replacing it. Only explicitly compatible region providers participate, and one
recipe selects at most one executor. Unrelated replacement families do not gain
this capability automatically. Materialization retains the selected fixed plans;
`graph.bind(...)` prepares immutable parameter frames before queued replay. Raw
mapping calls still include preparation. This combination can reduce host
resubmission costs without changing the chosen device algorithm; it is not a
claim that every combined recipe is faster or uses less memory.

Selection reports retain setup/first/steady costs, declared numerical contracts,
component identity and memory scope. CompileIQ's trial memory maximum is not a
driver-observed device peak. Cold materialization, after-evaluator resource
snapshots, requested workspace and pool reservation are different observations;
missing measurements are unavailable, not zero. Production acceptance belongs
to the downstream workload, including its reuse count and accuracy requirements.
See the [Graph API reference](forge_api_reference.en.md) for search, resume,
selection resolution and lifecycle-cost reports.

### Opt-in diagnostic facilities

NVTX 3 annotations use bundled headers, not a required `nvToolsExt` shared
library. They correlate explicit profiling of stages/trials/recipes with GPU
work; annotations are not physical strategies or automatic performance gates.

`ti.hardware.gpu_environment()` explicitly samples driver-provided NVML on an
NVIDIA device. `ti.hardware.capture_trial_environment()` attaches boundary
observations when it encloses `session.run(evaluator)` on the same thread.
Missing NVML or unsupported fields yield structured unavailable values. NVML
memory is device-wide, including other processes, not recipe/process peak memory.
Clock, power and temperature snapshots are not trial means. Sampling uses no
replay polling thread or added device synchronization, but its host time counts
toward the enclosing search budget. Passive `report()` / `telemetry()` do not
implicitly enable it or probe external libraries.

## Packaging and version rules

Use the following rules for every optional provider:

1. Install Forge normally. Install optional vendor packages afterwards in the
   application environment; never copy vendor runtimes into a Forge wheel or
   its package directory. A Forge-owned thin C-ABI adapter may live in the
   existing runtime wheel, but must not link or carry the vendor runtime.
2. Select exactly one CUDA-major package family for a given library in one
   environment. A `-cu12` or `-cu13` suffix describes that vendor package, not
   the Forge wheel.
3. Check the intersection of GPU architecture, driver, provider release, CUDA
   family, operating system, and Forge's operation contract. A working Forge
   CUDA kernel does not prove that an optional provider is compatible.
4. Prefer an explicit absolute library path when the Forge provider supports
   one. Otherwise configure the operating-system loader before starting the
   Python process.
5. Record the selected package version and resolved shared-library path in the
   application deployment manifest. Do not rely on whichever compatible-looking
   DLL or shared object happens to appear first.
6. Probe explicitly, then run a small correctness check with the same dtype and
   operation family as production. A successful probe checks discovery and ABI
   symbols; it is not a numerical or performance qualification.

For a component installed from a Python wheel, this command lists its actual
files without assuming a package layout:

```bash
python -m pip show -f PACKAGE_NAME
```

On Linux, add the directory containing the selected `.so` to the adapter's
RPATH or to `LD_LIBRARY_PATH` before process startup. On Windows, add the
directory containing the selected `.dll` to `PATH` for the current process;
`os.add_dll_directory()` is also suitable when it is called before the native
load. Keep transitive dependency directories visible for the same interval.

Do not use `setx PATH` as a current-shell test: it affects future processes but
not the shell that issued the command. In PowerShell, use `$env:PATH` for a
temporary test.

### Vulkan driver, ICD, and SDK boundary

Normal Vulkan use from an official Forge wheel needs a compatible GPU driver
and Vulkan ICD, but not a user-installed Vulkan SDK. Device extensions and
feature combinations are qualified at runtime and unsupported slices fail
closed. Installing validation layers or a newer SDK cannot add a device
feature that the driver does not expose.

The Vulkan SDK is a source-build/development dependency for headers, tools, and
validation, not an optional execution provider and not a reason to create a
Vulkan-versioned Forge wheel. Any future external Vulkan library must define
its own provider ABI and lifetime contract instead of being loaded implicitly
because the SDK is present.

### Optional CUDA compilation providers

The default Forge CUDA-kernel route still submits PTX to the CUDA Driver JIT.
It requires no CUDA Toolkit and starts no external process. A deployment that
needs offline cubins or compiler-level experimental optimization can select an
external `ptxas` before startup:

```powershell
$env:TI_CUDA_PTXAS_MODE = "external"
$env:TI_CUDA_PTXAS_PATH = "C:\CUDA\bin\ptxas.exe"
$env:TI_CUDA_ARTIFACT_CACHE_PATH = "D:\cache\taichi-cuda-artifacts"
```

On Linux, set `TI_CUDA_PTXAS_PATH` to an absolute path or make `ptxas`
resolvable from the current process's `PATH`. Forge packages no `ptxas`, CUDA
Toolkit, CompileIQ, or Python optimizer in its wheel. The application
environment owns every such tool and version. Other compilation-provider
variables do not change the default Driver JIT route unless
`TI_CUDA_PTXAS_MODE=external` is set.

| Variable | Contract |
| --- | --- |
| `TI_CUDA_PTXAS_MODE` | `driver` (default) or `external` |
| `TI_CUDA_PTXAS_PATH` | Optional absolute `ptxas` path; otherwise use `PATH` |
| `TI_CUDA_ARTIFACT_CACHE_PATH` | Persistent root for cubins, checksums, locks, and worker manifests |
| `TI_CUDA_PTXAS_TIMEOUT_SECONDS` | Bounded timeout for each cache-miss `ptxas` process; default 60 seconds |
| `TI_CUDA_PTXAS_ACF_PATH` | Optional static Advanced Controls File; mutually exclusive with a worker |
| `TI_CUDA_COMPILEIQ_WORKER` | Optional user worker executable or Python script |
| `TI_CUDA_COMPILEIQ_PYTHON` | Separate Python used to run a worker script; it may differ from Forge Python |
| `TI_CUDA_COMPILEIQ_TIMEOUT_SECONDS` | Bounded timeout for each cache-miss worker; default 3600 seconds |

Set every variable before `ti.init()`. After the first module load in a CUDA
session, Forge rejects a change in provider identity. To change configuration,
retire old work, call `ti.reset()`, and initialize with the new values. Cache
keys bind the PTX, GPU target, compiler options, Forge artifact schema, `ptxas`
content and version, and the ACF/worker identity. A cache hit loads the verified
cubin without starting the worker or `ptxas` again. Initial binary hashing, the
worker, and `ptxas` are fixed compilation costs, not scale-dependent kernel
execution costs.

CUDA Advanced Controls Files are applied through `ptxas --apply-controls` and
therefore require `ptxas` 13.3 or newer. A static ACF is appropriate for a
fixed, offline-qualified kernel family. ACF is an experimental compiler
control, so the application must retain its numerical oracle, compile timeout,
target GPU, and `ptxas` version, and disable the configuration after any
compile or validation failure. Forge does not silently execute another
explicit provider after a failure.

For the external PTXAS/ACF process route described in this section, CompileIQ
is not imported into the Forge application. Install the selected upstream
release in a separate supported Python environment and supply a
workload-specific worker:

```powershell
py -3.11 -m venv C:\venvs\compileiq
C:\venvs\compileiq\Scripts\python.exe -m pip install compileiq
$env:TI_CUDA_PTXAS_MODE = "external"
$env:TI_CUDA_COMPILEIQ_WORKER = "D:\app\forge_compileiq_worker.py"
$env:TI_CUDA_COMPILEIQ_PYTHON = "C:\venvs\compileiq\Scripts\python.exe"
```

The selected upstream CompileIQ release may have a narrower Python support
range than Forge. Recheck its Python and CUDA/`ptxas` support table during
deployment. This separate-interpreter constraint does not change the Python
support matrix of the Forge wheel itself.

This process worker is distinct from `ti.graph.compileiq_recipe_search()`.
That optional offline Graph-recipe API requires the modified fork's compatible
V2 complete-recipe capability and main-thread staged-search worker. Acceptance
is based on the protocol epoch, required schemas/API, and self-consistent core
and capability identities; it is not tied to one fork commit or wheel hash.
Forge records the installed Python-source identity and binds it to checkpoints,
so source drift invalidates resume evidence. The qualified fork supports Python
3.10--3.14. A generic upstream installation or the external JSON worker above
is not a substitute for this API. Task-indexed kernel/offload search remains
private qualification infrastructure and is not a public API.

Forge invokes the versioned JSON v1 process protocol as:

```text
PYTHON WORKER --request REQUEST.json --response RESPONSE.json
```

The request contains a temporary PTX path, artifact key, target, entry
manifest, compiler options, and the exact `ptxas` identity. The worker must
atomically write one of these responses:

```json
{"schema_version": 1, "status": "pass"}
```

or:

```json
{
  "schema_version": 1,
  "status": "ok",
  "acf_path": "C:/absolute/path/controls.acf",
  "acf_sha256": "EXPECTED_SHA256"
}
```

`pass` uses ordinary external `ptxas` for this artifact. `ok` verifies and
copies the ACF before invoking `ptxas`. The worker owns representative inputs,
the objective, and correctness and lifecycle gates. Forge has only PTX and
static options during compilation; it does not know an arbitrary kernel's
production inputs or physical invariants and therefore does not run global
autotuning on the application's behalf. A nonzero worker exit, timeout,
invalid JSON/status/path/checksum, or unsupported `ptxas` fails closed.

## Registered providers from the user environment

### cuSOLVERDn device Cholesky

`ti.hardware.linalg.CusolverDnProvider` is an explicit, user-installed dense SPD
solver. Forge supplies a lazy thin C-ABI binding, not the vendor runtime or its
CUDA dependencies. Pass a library file/directory, set `TI_CUSOLVERDN_LIBRARY_PATH`,
or install a compatible NVIDIA cuSOLVER component package. Transitive libraries
must also be discoverable. `ti.hardware.probe("cusolverdn", library_path=...)`
checks symbols/version; it does not qualify execution or select this solver.

```python
with ti.hardware.linalg.CusolverDnProvider(library_path) as provider:
    with provider.cholesky_plan(n, rhs_count=8, dtype=ti.f32) as plan:
        bound = plan.bind(a, rhs, solution)
        bound.factor_and_solve()
        # GPU producers may replace rhs; reuse factors only while A is unchanged.
        bound.solve()
        # GPU consumers may inspect plan.info without a host readback.
        status = plan.status()  # optional, explicitly synchronizes
```

A is scalar f32/f64 `(n, n)`, row-major, with the SPD matrix's **lower triangle**
supplied. It is preserved in a private factor buffer. For one RHS, rhs/solution
have shape `(n,)`; for multiple RHS, shape `(rhs_count, n)` stores **one vector
per row**, avoiding transposes. RHS and solution may be the same ndarray, but
then the RHS is overwritten and must be replenished before another solve. A
must not alias them. Plans have one immutable binding and retained device/host
workspace. Close plans before their provider; close/reset waits for outstanding
work and invalidates saved bound actions.

`factor()` invalidates the previous solve status, `solve()` reuses the factor,
and `factor_and_solve()` queues both. Device `info[0]` is the factorization result
and `info[1]` the solve API's numerical status; `-1` denotes not yet submitted
by this binding. A successful API return is **not** proof of SPD input or a
small residual. Consumers must use factor status and application-owned residual
criteria. In particular, solving after an unsuccessful factorization is not a
valid solution. No SPD scan, residual readback, retry, or implicit fallback is
added to each call. f32 error depends on dimension and conditioning; use f64
when the application's accuracy requires it.

`plan.memory_report()` separates private factor/workspace/status bytes from
unknown vendor/driver residency; `plan.host_workspace_bytes` reports host
workspace, and caller arrays are excluded. Calls use Forge's existing ordered
CUDA submission/lifetime boundary. This path is not kernel-callable, Graph
recordable, or a CompileIQ recipe axis, and does not change runtime auto. Local
execution evidence covers Windows, cuSOLVER 12.1.0, RTX 5090; other library/driver
combinations are not implied qualified. See NVIDIA's [generic Cholesky contract](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxpotrf).

### cuBLAS, cuSPARSE, and cuFFT

These providers use copied stable declarations and runtime symbol loading;
Forge does not use Toolkit headers or link the libraries into the official
wheel. Supply them through either a compatible CUDA Toolkit installation or
NVIDIA component wheels matching the application's CUDA family. For example,
replace `XX` below with `12` or `13` rather than running the command literally:

```text
python -m pip install nvidia-cublas-cuXX nvidia-cusparse-cuXX nvidia-cufft-cuXX
```

The library paths for these three providers are implicit. Configure the system
loader before Python starts; `library_path=` is intentionally rejected. CUDA
12+ cuSPARSE may also require `nvJitLink`, so the selected provider's
transitive-library directory must be visible.

Verify an initialized CUDA runtime explicitly:

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("cublas"))
print(ti.hardware.probe("cusparse"))
print(ti.hardware.probe("cufft"))
```

Recommended lifecycle:

- Reuse the Program-scoped cuBLAS handle and matrix-scoped cuSPARSE
  descriptors/preprocessing instead of recreating them around every call.
- Reuse a fixed-size cuFFT plan. Its workspace and layout are part of the plan
  contract; rebuild on shape, transform kind, dtype, device, or runtime change.
- Keep explicit requests fail-visible. An unavailable or failed explicit
  provider is not permission to copy data to the host or select another
  algorithm silently.

### cuDSS 0.8.x

The platform `taichi-forge-runtime` wheel contains one Forge-owned thin C ABI 1
adapter built against official cuDSS 0.8 headers. It does not link cuDSS, the
CUDA runtime, cuBLAS, or Python, and creates no new wheel variant. Users do not
rebuild Forge. The application environment still supplies the vendor cuDSS
runtime and its transitive dependencies.

Forge's public slice is bound to cuDSS 0.8.x. Install the package matching the
application CUDA family:

```bash
# Choose one, not both, in an environment.
python -m pip install "nvidia-cudss-cu12>=0.8,<0.9"
python -m pip install "nvidia-cudss-cu13>=0.8,<0.9"
```

cuDSS also needs compatible cuBLAS libraries. The current Forge resolver uses
this order:

1. the `library_path=` argument;
2. `TI_CUDSS_LIBRARY_PATH`;
3. known `nvidia` namespace-package locations for the active CUDA-driver
   family.

An explicit path may name the shared library or a directory containing it. It
is exclusive: Forge does not fall back to another candidate when that path is
wrong. On Linux, NVIDIA's cuDSS wheel may contain `libcudss.so.0` without an
unversioned symlink; Forge resolves the versioned library directly.
`library_path` always names the vendor runtime. The wheel-internal adapter is
not part of the public path contract and cannot be overridden.

```powershell
# Optional explicit Windows deployment binding for the current process.
$env:TI_CUDSS_LIBRARY_PATH = "C:\vendor\cudss\bin\cudss64_0.dll"
```

```bash
# Optional explicit Linux deployment binding.
export TI_CUDSS_LIBRARY_PATH=/opt/vendor/cudss/lib/libcudss.so.0
```

Before creating a solver, verify the exact deployment candidate:

```python
import os
import taichi_forge as ti

ti.init(arch=ti.cuda)
path = os.environ.get("TI_CUDSS_LIBRARY_PATH")
report = ti.hardware.probe("cudss", library_path=path)
print(report)
```

The probe transiently loads the adapter and vendor runtime, queries the 0.8.x
version, and releases both. It creates or retains no solver handle, factor, or
workspace. Only `CudssPlan` owns execution-time adapter/runtime handles, which
close deterministically with the plan.

Forge currently requires CUDA Driver API 12.0 or newer and a square scalar f32
CUDA CSR matrix. `CudssPlan` separates `analyze()`, `factorize()` /
`refactorize()`, and `solve()`:

- Reuse analysis while the sparsity pattern is unchanged.
- Reuse factors while both pattern and values are unchanged; use
  `refactorize()` when only values change.
- Keep the plan alive until direct calls and submitted root-Graph actions have
  retired. Close it deterministically afterwards.
- Budget for opaque analysis, factor, and workspace memory. The CSR input bytes
  are not the provider's total peak memory.
- Use `provider="auto"` only with matching Forge admission evidence. Without
  evidence, auto does not probe cuDSS and retains cuSOLVERSp. Use
  `provider="cudss"` when the application intentionally selects the provider.

The recommended physics workload is a repeatedly solved fixed-pattern sparse
system where analysis and usually refactorization are amortized. For a
one-off, small, or frequently remeshed system, measure the complete
analysis-factor-solve lifecycle rather than solve time alone.

#### Complete sparse-solve regions

`ti.linalg.record_sparse_solve(pattern, initial_values, ...)` describes one
square scalar f32 CSR matrix and one or more compact f32 vector RHS/output
pairs. The immutable `SparsePattern` supplies topology; the operation copies
initial values into its own preparation storage. `values=None` declares a
fixed matrix whose factors may be reused. A named `values` binding declares
current numerical values on every invocation, requiring factor/refactor before
solving. These are different semantic contracts, never interchangeable search
choices. All bound arrays must be disjoint; outputs may feed later invocations.

```python
operation = ti.linalg.record_sparse_solve(
    pattern, initial_values,
    values="matrix_values",
    rhs_pairs=(("rhs0", "solution0"), ("rhs1", "solution1")),
    matrix_type="spd", matrix_view="full",
    absolute_tolerance=2e-5, relative_tolerance=2e-5,
    library_path=path,
)
preparation = operation.prepare(max_plans=2)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (
    *ti.graph.default_recipe_providers(),
    ti.hardware.linalg.SparseSolveRecipeProvider(),
)
catalog = definition.recipe_catalog(providers=providers)
```

Pass the same providers to the complete-recipe search/materialization/resolution
APIs. Preparation performs bounded private analysis and numerical warmup, not
benchmarking on caller outputs. The baseline already shares a factor owner
across RHS pairs; alternatives freeze reordering and full-factor/refactor
lifecycles. RHS vectors are solved sequentially with shared workspace, not
silently converted to a dense batched matrix. Default policies remain vendor
choices: different requested phases do not guarantee different kernels or a
speedup. Modified CompileIQ receives opaque complete recipes, not library names
or numerical-policy knobs. Ordinary `SparseSolver` auto behavior is unchanged.

Materialization needs the adapter's optional configuration/allocator extensions
and matching native Graph-owned capture capability. It creates private solver
storage and a retained stream; analysis stays outside replay. Capture records
one numerical update per region and its ordered solves, with per-binding
parameter ownership. Resident-capable native builds upload frozen parameters
once after capture and use device-to-device copies on replay. The report records
this capability; restoration rejects a different capture-storage contract.
Preparation can synchronize and allocate; steady replay adds no input scans,
Python/vendor calls, host error readbacks, or report collection. Workspace
clears and device copies still execute. Known memory distinguishes shared plan
payload, the source's numeric snapshot, and Graph-owned per-binding parameters;
vendor estimates and opaque driver pool/residency are not measured VRAM peaks.

Preparation observations can also contain `preparation_factor_statistics`:
vendor-reported factor nonzeros, superpanel count, and factorization FLOPS from
the initial private snapshot. The optional adapter extension queries these once
at cold preparation; reports read cached facts, never query the solver during
replay. These are not GPU counters or statistics for later changed matrix values.
Missing support and failed individual queries remain unavailable rather than
zero, and do not disable execution with an older adapter. Fewer factor nonzeros
or FLOPS alone do not imply faster execution. See NVIDIA's
[cuDSS data types](https://docs.nvidia.com/cuda/cudss/types.html) for their meaning.

The caller declares finite, nonsingular inputs and the matrix class. Each RHS
must satisfy `||Ax-b||inf <= atol + rtol*||b||inf` in the evaluator; this is not a
per-replay residual check or a promise for arbitrary ill-conditioned inputs.
Windows contracts cover SPD and general non-symmetric changing values, fixed
SPD values, and interleaved bindings. They do not qualify every symmetric-
indefinite/pivoting case, Linux deployment, or production workload.

Preparation/selection reuse stores JSON facts, not CSR data or vendor factors.
A new process supplies the same pattern, initial values and semantic contract,
passes `preparation=...`, resolves the saved selection, and rebuilds only the
selected numerical plan. Provider/device/seed/resource drift is explicit.
Neither Python executables nor CUDA graphs are deserialized from the report.

### OptiX runtime provider

The platform `taichi-forge-runtime` wheel contains three Forge-owned thin
adapters built against pinned official OptiX headers: ABI 93 (OptiX 8.1), ABI
105 (OptiX 9.0), and ABI 118 (OptiX 9.1). They share Forge provider C ABI 1 and
are files in the same wheel. Users do not install an OptiX SDK, a CUDA Toolkit,
or rebuild Forge. The wheel still does not contain `nvoptix.dll` or
`libnvoptix.so.1` and does not gain a CUDA- or OptiX-version variant.

The vendor runtime is normally supplied by the NVIDIA display driver. ABI 93,
105, and 118 respectively require driver branches R555, R570, and R590 or
newer. Forge tries its adapters from newest to oldest and retains the first one
whose ABI the installed runtime accepts. An unsupported newer ABI is a bounded
fallback condition; an error after context or scene creation is not permission
to silently switch implementations.

The adapters embed `compute_75` PTX 8.5 generated by the release-pinned CUDA
12.5.x compiler. This build-time dependency is not packaged and does not imply
a user CUDA Toolkit dependency. The PTX ceiling is audited during the build so
a newer release compiler cannot silently raise the ABI 93 / R555 driver floor.

Runtime discovery uses this order:

1. the `library_path=` argument, when it names `nvoptix.dll` or
   `libnvoptix.so.1`;
2. `TAICHI_FORGE_OPTIX_LIBRARY`;
3. the standard NVIDIA driver search implemented by the OptiX loader.

An explicit path is exclusive and is useful for containers or nonstandard
driver layouts. It always identifies the vendor runtime; Forge adapters are an
internal runtime-wheel resource and cannot be overridden through the public
API. `probe()` transiently loads the adapter and vendor runtime to check the
exact ABI, but it does not create or retain a CUDA or OptiX context.

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)
print(ti.hardware.probe("optix"))

with ti.hardware.ray.load_optix_provider() as provider:
    # Create scenes, retain them through submitted work, then close scenes.
    pass
```

To bind a nonstandard vendor runtime explicitly:

```python
vendor_runtime = "/opt/nvidia/lib/libnvoptix.so.1"  # or nvoptix.dll
print(ti.hardware.probe("optix", library_path=vendor_runtime))
with ti.hardware.ray.load_optix_provider(vendor_runtime) as provider:
    pass
```

The provider owns an OptiX context. Scenes retain that provider; submitted
Graph work retains both. Close scenes before the provider and do not reuse any
object after `ti.reset()`. `validation=True` is recommended for development,
not as a default performance setting.

## Explicit optional runtime execution providers

The standard runtime wheel contains Forge-owned thin adapters for the following
three libraries. An adapter contains and links no vendor code. `probe()` loads
the user runtime transiently; creating a provider instead retains the selected
runtime and exposes a bounded execution ABI. All routes are explicit host-side
resources: none is a Graph action, kernel intrinsic, or automatic rewrite.

### cuSPARSELt recommended configuration

cuSPARSELt accelerates matrix multiplication where one operand satisfies the
provider's 50% structured-sparsity contract. It is not a general sparse-matrix
solver and is not a replacement for ordinary CSR SpMV.

Install one CUDA family and inspect the actual shared-library location:

```bash
# Choose one.
python -m pip install nvidia-cusparselt-cu12
python -m pip install nvidia-cusparselt-cu13
python -m pip show -f nvidia-cusparselt-cu13
```

Probe an explicit file or directory. If `library_path` is omitted, Forge reads
`TI_CUSPARSELT_LIBRARY_PATH`, then checks the installed NVIDIA package files,
then asks the operating-system loader:

```python
report = ti.hardware.probe(
    "cusparselt", library_path=r"C:\absolute\path\cusparseLt64_0.dll"
)
probe = next(
    item for item in report.operations
    if item.descriptor.operation_id == "runtime.probe.cusparselt"
)
assert probe.discovery == "available"
```

Execute with an explicit retained plan. `A` must already satisfy exact 2:4
sparsity; Forge does not prune or silently modify the numerical operator. `B`
uses row-major `(n, k)` transposed storage:

```python
with ti.hardware.tensor.CusparseLtProvider(runtime_path) as provider:
    with provider.matmul_plan(m, n, k) as plan:
        plan.compress(a).execute(b_transposed, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

The fixed plan can also be captured into a root CUDA Graph:

```python
provider = ti.hardware.tensor.CusparseLtProvider(runtime_path)
plan = provider.matmul_plan(m, n, k)
plan.compress(a)  # Explicit, already-valid 2:4 weight snapshot.
recording = plan.record(alpha=0.75, beta=0.25)
builder = ti.graph.GraphBuilder()
builder.append_native(recording)
graph = builder.compile()
frame = graph.bind({"b": b_transposed, "c": output, "d": output})
graph.run(frame)
```

For A that changes on each replay, use a separate plan and
`plan.record(a="a", ...)`, then include `"a": a` in the bindings. That command
captures compression followed by matmul; neither bind nor capture executes
the mathematics or advances C/D feedback. B/C/D values remain live in both
modes, and C/D may alias. A/D and B/D aliasing is rejected at binding/capture.
Immutable binding-frame recipes are supported on runtimes exposing this
capture capability, without Python provider callbacks during replay.

The recording retains the plan, compressed data and scratch. While any
recording/Graph holds a lease, `plan.compress()` and `plan.close()` are rejected.
For a new snapshot, retire the old bindings/Graphs/recordings first, or create a
new plan/weight epoch. Do not mix refreshing and snapshot recordings on one
plan: their shared compressed buffer would invalidate the snapshot meaning.
Runtime reset retires the vendor plans before the device Program is destroyed.
There is no standalone recording execution, nested sequential/AOT recording,
automatic pruning, value-change detection, or implicit algorithm search.
Snapshot reuse and per-replay refresh have different input contracts; they
must not be presented as interchangeable optimization candidates without a
common explicit weight-lifetime contract.

For a complete shared-current-A region, use the semantic entry instead of
searching raw plan parameters:

```python
operation = ti.linalg.record_sparse_matmul(
    m, n, k,
    products=(("b0", "c0", "d0"), ("b1", "c1", "d1")),
    alpha=0.75, beta=0.25, activation="relu",
    absolute_tolerance=1e-3, relative_tolerance=3e-3,
)
preparation = operation.prepare(max_algorithms=8)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (
    *ti.graph.default_recipe_providers(),
    ti.hardware.tensor.SparseMatmulRecipeProvider(),
)
# Pass providers to the usual definition.search_recipes(...), with the
# caller's target, budget, workload and evaluation contract.
```

Each product computes `D_i = activation(alpha * A @ B_i.T + beta * C_i)`.
All products use the same positive, multiple-of-16 M/N/K and compact scalar
FP16 arrays, with at most `2**31-1` elements per matrix. One product is also
allowed. A must already satisfy row-wise 2:4 sparsity; no values are inspected
or pruned. Every invocation reads current A/B/C. Only each product's own C/D
may alias; no output may overwrite another product's inputs or outputs.

Preparation creates host descriptors only, freezes actual algorithm attributes
and compressed/scratch/workspace sizes, and reports bounded enumeration or
unavailable candidates. It neither measures performance nor calls vendor
autotuning on caller data. Every materialized region owns its plan and one
compressed-A/workspace allocation, used by ordered products. The baseline
already uses legal compression reuse and vendor-default settings. Alternatives
change the frozen algorithm and fused/separate ReLU dataflow, never the library
route. Separate helpers use the known semantic extents.

Save `preparation` together with the search selection artifact. A new process
can recreate the same operation with `preparation=...`, freeze an equivalent
definition and call `resolve_recipe(selection_artifact, providers=providers)`.
Restoration does not repeat enumeration: it reconstructs the selected plan and
checks component, device, actual configuration and resource facts at cold
boundaries. It does not deserialize executable objects or reuse another
algorithm's compressed bytes. The Forge report carries these dataflows and
limitations; known memory excludes opaque vendor state and driver peak.

This route needs the adapter's optional configured-plan execution table and
native shared-A capture support; the legacy explicit-plan ABI remains usable
without that extension. Development validation covers Windows, cuSPARSELt
0.9.1 and a supported NVIDIA GPU, not every driver/library combination or
production workload. No new steady-replay probe, validation or synchronization
is introduced, and ordinary automatic selection is unchanged.

Follow the selected release's supported GPU list and driver requirements.
cuSPARSELt 0.8 supports CUDA 12.9/13.0; 0.9 dropped CUDA 12.9 support.
The vendor runtime and its CUDA dependencies remain outside Forge's portable
wheel. Package availability is not a compatibility test. See the
[cuSPARSELt release notes](https://docs.nvidia.com/cuda/cusparselt/release_notes.html).

The application adapter should own this lifecycle:

1. Create the handle, dense/structured matrix descriptors, matmul descriptor,
   algorithm selection, and plan.
2. Check or deliberately prune the structured operand according to the exact
   data type, layout, transpose mode, alignment, and supported architecture.
3. Query compressed storage and workspace sizes, compress the structured
   operand, and retain both compressed data and plan.
4. Execute many compatible matmuls. Recompress when the structured operand's
   values change; rebuild descriptors/plan when shape, strides, dtype,
   operation, device, or library ABI changes.
5. Destroy resources only after the stream work that uses them has completed.

Recommended application policy (this is not a Forge API argument):

```yaml
provider: cusparselt
activation: explicit
sparsity_policy: already_valid_2_of_4
automatic_pruning: false
plan_cache_key: [device, library_version, dtype, shape, strides, transpose]
candidate_preparation: descriptors_only
weight_lifetime: explicit_snapshot_or_current_per_invocation
fallback_owner: application
```

There is no fixed minimum reuse count or universal speedup gate. Measure the
actual compression, product and memory costs. Preparation and plan creation
belong outside replay; compression of changing A must remain in the execution
dataflow. Forge's semantic provider does not call `cusparseLtMatmulSearch()`.

For physics workloads, automatic pruning is normally unsafe: changing a mass,
stiffness, Jacobian, contact, or constraint matrix to satisfy 2:4 sparsity
changes the numerical operator. Use cuSPARSELt only when the model or learned
operator already defines the structured pattern, or when the application has
an explicit approximation policy with residual, conservation, and stability
checks. Dense-ish repeated local/block operators and batched constitutive or
reduced-order transforms are better candidates than irregular global CSR
systems.

Admission should include the full amortized cost:

```text
plan + prune/check + compression + repeated matmul + extra memory
```

Require the application's numerical contract, then compare device time, host
submission, synchronization and memory separately on its real workload.
Retain legal physical alternatives and mark negative scopes; unexpected losses
need implementation/timeline attribution, not a blanket family rejection.

### cuTENSOR recommended configuration

Install the package matching the application CUDA family:

```bash
# Choose one.
python -m pip install cutensor-cu12
python -m pip install cutensor-cu13
```

Use an explicit path or set `TI_CUTENSOR_LIBRARY_PATH`. Without either, Forge
checks the installed `cutensor-cu13`/`cutensor-cu12` package files before the
system loader:

```python
report = ti.hardware.probe("cutensor", library_path="/opt/cutensor/lib/libcutensor.so.2")
```

The current execution surface is compact row-major scalar `f32` contraction
with `f32` or `tf32` compute. Modes explicitly define the contraction:

```python
with ti.hardware.tensor.CutensorProvider(runtime_path) as provider:
    with provider.contraction_plan(
        (m, k), "ik", (k, n), "kj", (m, n), "ij", (m, n), "ij"
    ) as plan:
        plan.execute(a, b, c, d, alpha=1.0, beta=0.0)
        ti.sync()
```

For repeated contractions, `plan.record(alpha=..., beta=...)` is a root CUDA
Graph recording, with symbolic `a`, `b`, `c`, `d` bindings (configurable at
record creation):

```python
recording = plan.record(alpha=1.0, beta=0.25)
builder = ti.graph.GraphBuilder()
builder.append_native(recording)
graph = builder.compile()
bindings = graph.bind(dict(a=a, b=b, c=c, d=d))
graph.run(bindings)
```

The recording retains the prepared vendor plan and exact workspace. Plan and
provider close are rejected while their dependents remain live; retire Graphs,
builders, definitions and recordings before closing the plan. Runtime reset
retires these resources before finalizing the CUDA Program. Capture does not
execute the contraction; it does not advance `beta*C` feedback. C/D may share
storage only with identical layouts/modes, and D must not alias A/B. Shape,
dtype and storage legality are resolved at binding/capture, not scanned during
steady replay. The fixed recording also composes with immutable binding frames.
No standalone `recording.execute()` or nested sequential/AOT recording is
provided. This execution route does not itself expose contraction strategy
search or change automatic selection.

For complete contraction search, use the semantic entry rather than a fixed
expert plan:

```python
operation = ti.linalg.record_contraction(
    (19, 5, 7), "kmi", (3, 19, 11), "jkn", "imjn",
    alpha=0.75, beta=0.25, activation="relu",
    absolute_tolerance=3e-5, relative_tolerance=3e-5,
)
preparation = operation.prepare(workspace_limit_bytes=32 << 20)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (*ti.graph.default_recipe_providers(),
             ti.hardware.tensor.ContractionRecipeProvider())
session = definition.search_recipes(
    engine="compileiq", providers=providers, target=target, budget=budget,
    workload_context=workload, evaluation_contract=evaluation_contract,
    backend_environment=environment,
)
decision = session.run(evaluator)
```

`target`, `budget`, `workload`, `evaluation_contract`, `environment` and `evaluator`
are caller-owned, as in the [complete-recipe search contract](graph_runtime_optimization.en.md).
Default binding names are `a`, `b`, `c`, `output`; output/C shape is inferred
from the declared output modes. Shared output modes are batches. Each reduced
mode must occur in both inputs with the same extent. Repeated modes, implicit
broadcasting and scalar outputs are not supported. Dataflow helpers follow
Forge's 12-index limit and require at most `2**31-1` elements per tensor.
`compute="f32"` or `"tf32"` is a semantic choice, never a search axis.

Preparation creates descriptors, not candidate GPU scratch or performance
measurements. The baseline uses original input layouts and the vendor default
plan. Candidates physically reorder A, B or both, and can separate the product
from a fused alpha/beta/activation epilogue. These buffers refresh every replay;
they are not caches of input values. Neutral alpha/beta and singleton-only
permutations do not create artificial candidates. Workspace limits must be
positive: the legacy adapter's zero means its estimate, not zero scratch.

Save `preparation` and `decision.selection_artifact.to_dict()` as JSON. In a new
process, recreate the same operation with `preparation=preparation`, freeze it,
then use `definition.resolve_recipe(selection, providers=providers)` and
`definition.materialize(resolved, providers=providers)`. Restoration verifies
the semantic/device/component contract and the selected plan's required
workspace; it does not rediscover the candidate catalog. It reconstructs a
vendor plan request, **not a serialized or guaranteed bitwise-identical vendor
kernel binary**. Opaque vendor kernels and transitive-library behavior remain
outside that identity claim. Environment/measurement applicability must include
the caller's relevant driver and vendor dependency versions. Reports preserve
declared dataflows, imported-preparation provenance and actual evaluator facts
separately; synthetic tests do not confer production qualification.

The cuTENSOR vendor runtime and its CUDA dependencies stay outside the portable
wheel; Forge's thin bundled adapter and capture bridge remain inside. Current
Forge execution/recording covers contraction, not the vendor's broader
reduction, permutation or elementwise APIs. Windows capture has been exercised
with cuTENSOR 2.7/CUDA 13; this is not a claim of testing every supported vendor
version or platform. Resource reports distinguish known workspace bytes from
opaque vendor state and do not claim to measure driver peak VRAM.

Recommended adapter policy:

- Cache descriptors and plans by operation, dtype/compute type, layout, shape,
  workspace limit, device, CUDA version, and cuTENSOR version.
- Set an explicit workspace budget and query the plan's actual requirement.
- Start with JIT disabled for predictable startup and Graph compatibility.
  Enable it only for a repeatedly executed contraction after profiling.
- A persisted plan cache is valid only for the same cuTENSOR version, CUDA
  version, and matching GPU architecture/multiprocessor configuration. Reject
  cache mismatches rather than silently reusing them.
- Keep small fixed-shape elementwise or contraction kernels handwritten when
  provider setup and general-layout dispatch cost are not amortized.
- Do not infer Tensor Core use from the library name. Dtype, compute descriptor,
  selected plan, and device determine the actual hardware route.

### AmgX recommended configuration

AmgX is a full configurable algebraic-multigrid/Krylov solver, not a kernel
intrinsic. Use a compatible user-installed library, or build an official NVIDIA
release whose CUDA and architecture support matches the deployment:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
  -DCMAKE_NO_MPI=ON
cmake --build build --config Release --target amgxsh
```

Replace the architecture list with the deployment GPUs. Recent AmgX releases
use `CMAKE_CUDA_ARCHITECTURES`; check that release's notes before using older
`CUDA_ARCH` examples. Set `CMAKE_NO_MPI=ON` for a single-GPU adapter. A
distributed build additionally needs a compatible MPI implementation. Add the
resulting `amgxsh.dll` or `libamgxsh.so` directory to the runtime loader path.
Windows is supported upstream but has more limited upstream test coverage, so
keep it behind an explicit deployment qualification.

AmgX has no default Python-package search. Pass the built library explicitly or
set `TI_AMGX_LIBRARY_PATH`; compatible CUDA, cuBLAS, and cuSPARSE dependencies
must remain visible to the loader:

```python
report = ti.hardware.probe("amgx", library_path="/opt/amgx/lib/libamgxsh.so")
```

Execution accepts contiguous host i32 CSR topology and f32/f64 numeric values
from host arrays or scalar CUDA Taichi ndarrays. AmgX
owns device upload, hierarchy, and solver resources; keep the solver when the
topology is reused and provide the exact application-owned configuration:

```python
with ti.hardware.linalg.AmgxProvider(runtime_path) as provider:
    with provider.solver(offsets, columns, values, "PCG_V.json", config_file=True) as solver:
        solution, info = solver.solve(rhs)
        assert info["converged"]
```

For GPU producers and consumers, bind caller-owned device arrays once:

```python
# offsets/columns are host i32 arrays; values_gpu/rhs_gpu/solution_gpu are
# scalar CUDA Taichi ndarrays of matching f32/f64 dtype and fixed 1D shape.
with provider.solver(offsets, columns, values_gpu, config) as solver:
    bound = solver.bind_device(rhs_gpu, solution_gpu, values=values_gpu)
    # After a GPU producer changes numeric values, refresh explicitly:
    bound.replace_coefficients()
    solution, info = bound.solve()  # solution is solution_gpu, not a numpy copy
```

Binding validates dtype, shape and runtime ownership and retains buffers.
Subsequent calls use their live contents without repeating pointer discovery.
Changing coefficients does not implicitly update the solver: call
`bound.replace_coefficients()` or `solver.replace_coefficients(values_gpu)`.
The zero-initial-guess policy is fixed at bind; pass `zero_initial_guess=False`
to read the current solution buffer as the initial guess. `bound.close()`,
solver close and runtime reset invalidate the binding.

When coefficients and RHS are ready together, `bound.update_and_solve()` refreshes
the bound values and solves under one retained submission. This removes the
intermediate Forge producer wait; it does not skip vendor setup or convergence
work. It requires `values=...` at binding and no intervening Forge operation.
Callers must not mutate the buffers concurrently. The separate methods remain
available, and fewer waits do not guarantee lower latency for every workload.

The adapter normally recomputes a full residual after solving. Applications that
do not need that additional observation can choose a fixed policy at creation:

```python
solver = provider.solver(offsets, columns, values_gpu, config, compute_residual=False)
bound = solver.bind_device(rhs_gpu, solution_gpu, values=values_gpu)
solution, info = bound.update_and_solve()
assert info["residual_norm"] is None  # omitted, not zero or a cached residual
```

`compute_residual=True` remains the default. Opting out does not modify AmgX's
configuration or its existing convergence checks; status and iteration count
are still returned, including nonconvergence. This is an observation policy,
not an algorithm or CompileIQ search axis. An older Forge adapter can still use
the default path; an unsupported opt-out is rejected at solver creation using
an adapter capability bit, without a Forge commit pin or patched AmgX runtime.

This is **device-buffer interoperability, not an asynchronous or zero-copy
solver**. The stable AmgX API copies into/from vendor-owned storage. Forge
retains the existing producer synchronization before a vendor call; AmgX's
solver control and residual query remain host-controlled. Device output uses
the existing external-submission lifetime tracking. It is not Graph-recordable
or CompileIQ-searchable. CSR topology stays on the host; no device topology
readback is hidden inside binding. No vendor speedup or peak-VRAM reduction is
implied by accepting device buffers: measure the actual configured AmgX build
and application workload. Vendor hierarchy, vector copies and internal
workspace still contribute to VRAM.

Forge owns the thin adapter, buffer/lifetime integration, and its diagnostic
call policy, not an AmgX fork.
Local Windows checks with unmodified AmgX 2.5.0 exercised f32/f64 device inputs,
coefficient updates, and subsequent GPU consumers. Traces showed reduced bulk
host/device transfers, not a faster Krylov algorithm or lower vendor workspace.
AMG coefficient setup can still allocate temporary storage and synchronize
internally even when inputs reside on the GPU. These vendor costs do not imply
that Forge requires patched libraries or changes the caller's solver settings.

`replace_coefficients()` always refreshes solver setup after replacing numeric
values. The adapter uses `AMGX_solver_resetup` when that optional/deprecated C
symbol is exported, and otherwise performs a full `AMGX_solver_setup`. This
keeps execution compatible with runtimes that retain the stable setup API but
omit the resetup entry point; the fallback is correct but may rebuild more
state.

For repeated coefficient updates with fixed CSR topology, tune AmgX's own
`structure_reuse_levels` in the AMG or AMG-preconditioner scope. `0` rebuilds
the hierarchy; positive values retain progressively more existing level
structure. Reused levels keep their prolongation/restriction operators while
the coarse matrix is recomputed, so Forge never raises this setting
automatically. Some AmgX releases also accept `-1` to retain all levels; treat
that as release-qualified rather than a portable default. Admit any nonzero
setting only after the expected coefficient-change envelope passes residual,
convergence, iteration-count, worst-case update-time, and peak-memory gates.

Close every plan/solver before its provider and do not reuse either after
`ti.reset()`. Provider close fails while a child resource is live. Explicit
selection surfaces load, numerical, and lifecycle errors rather than silently
falling back.

Recommended physics starting points:

- SPD elliptic/Poisson-like systems: start from the shipped `PCG_V.json` or
  `PCG_AGGREGATION_JACOBI.json` and verify the symmetry/positive-definite
  contract.
- Nonsymmetric systems: start from `FGMRES_AGGREGATION.json` or a shipped
  BiCGSTAB configuration; do not select PCG by matrix size alone.
- Keep setup and hierarchy objects while the sparsity topology is unchanged.
  Use `structure_reuse_levels` only when the selected AmgX release documents
  it and the application validates that lifecycle.
- Gate on residual/convergence, iterations, setup time, solve time, and peak
  memory. AMG hierarchy memory can dominate the original CSR storage.
- Store the exact JSON configuration with the deployment. AmgX's large tuning
  surface makes a library-version-only performance claim meaningless.

## Unregistered candidate: NCCL

NCCL is intentionally not part of this adapter mount and has no Forge probe or
execution API.

### NCCL recommended configuration

NCCL is relevant only to multi-GPU or multi-node communication. It does not
accelerate a single-GPU kernel. The recommended Forge candidate scope is Linux,
where NVIDIA's install guide supplies `libnccl2` and `libnccl-dev` packages:

```bash
sudo apt install libnccl2 libnccl-dev
```

An unpinned repository install may upgrade CUDA. Pin the NCCL/CUDA package
versions when preserving an older application stack. An adapter should retain
one communicator per participating device/process group, bind collectives to
explicit CUDA streams, propagate asynchronous errors, and abort/close every
communicator on partial initialization failure.

Physics candidates include halo exchange, distributed vector reductions, dot
products, and coarse-grid/global synchronization in an already partitioned
solver. Admission must measure communication and synchronization with the
actual PCIe/NVLink/network topology. A local compute microbenchmark cannot
qualify NCCL.

## Troubleshooting

| Symptom | Check | Required action |
| --- | --- | --- |
| Probe reports unavailable | Active backend, exact provider ID, shared-library file, transitive libraries | Fix discovery; do not enable auto selection by assumption |
| Windows DLL is present but will not load | Current-process `PATH`, `os.add_dll_directory()`, architecture, dependent DLLs | Configure before load and bind one CUDA-major family |
| Linux `.so` is present but will not load | `LD_LIBRARY_PATH`/RPATH, SONAME, dependent `.so` files | Use the versioned SONAME when the package omits symlinks |
| Provider loads but execution fails | dtype, shape, layout, device, stream, provider ABI/version | Surface the provider failure; do not silently fall back after explicit selection |
| Correctness differs | matrix properties, pruning/precision, transpose/layout, stale plan or values | Treat the candidate as invalid for that numerical contract; diagnose correctness before comparing performance |
| First call is slow | plan creation, JIT, analysis, compression, allocation | Separate setup from steady state and use the production reuse count |
| Memory grows | live plans/scenes/factors, workspaces, caches, in-flight Graph leases | Inspect provider memory reports where available and close owners after retirement |
| Performance is unstable | synchronization, cold caches, clock/power state, algorithm search, topology | Compare balanced fresh-process runs at the actual workload scale; report variability and device/host/memory trade-offs without a universal positive-speedup gate |

Native Windows adapters that fail during deep provider plan creation should
also check the host thread/executable stack reserve. Treat a larger reserve as
a provider-version-specific deployment workaround, not as a Forge runtime
requirement.

## Deployment acceptance checklist

Before enabling an external provider in production, retain evidence for all of
the following:

- exact Forge build/runtime identity and active backend;
- GPU UUID/architecture and driver version;
- provider package version, shared-library content identity, and transitive
  dependency family;
- operation shape, dtype, layout, topology, and reuse/update policy;
- numerical oracle or solver residual/convergence gate;
- setup and steady-state timings with explicit synchronization;
- worst-case result and variability, not only the best or median sample;
- provider-owned, workspace, compressed/factor, and peak memory budget;
- resource close/reset behavior and in-flight submission lifetime.

Do not turn a local benchmark into an automatic global heuristic. Automatic
selection requires an exact-scope, fail-closed admission contract; otherwise
keep the provider explicit.

## Optional Vulkan FFT plans

The current source exposes `ti.hardware.fft.VulkanFftPlan`. It requires a native
runtime containing the FFT bridge and the separate
`taichi_forge_vkfft_provider_abi1_vkfft134` DLL/SO. The adapter compiles VkFFT 1.3.4
with a matched static glslang/SPIRV-Tools distribution. Execution needs the Vulkan
loader/driver, not CUDA, the Vulkan SDK, or a shared glslang runtime. Standard
runtime builds enable `TI_BUILD_VKFFT_PROVIDER` and install the adapter under
`taichi_forge_runtime/_lib/hardware_providers`, with upstream notices under
`_lib/licenses/vkfft`. Older artifacts may omit it; probe the installed adapter
instead of assuming a source checkout describes an already-published wheel.
Builders can supply an offline `TI_VKFFT_ROOT` and matched static compiler paths
in `cmake/TaichiVkfftProvider.cmake`; otherwise the build fetches pinned VkFFT
sources. No source download or C++ build occurs on user-side plan creation.

```python
ti.init(arch=ti.vulkan)
data = ti.ndarray(ti.f32, shape=(2, 16, 8, 2))
# Populate data with interleaved real/imaginary scalars before execution.
with ti.hardware.fft.VulkanFftPlan(
    data, (16, 8), batch_count=2, direction="inverse",
    normalization="inverse",
) as plan:
    plan.run()  # In place, on Forge's ordered compute queue.
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(data="signal"))
    graph = builder.compile()
    bindings = graph.bind({"signal": data})
    graph.run(bindings)
    memory = plan.memory_report()
    build_and_allocation_facts = plan.statistics()
```

Explicit use resolves the adapter from the runtime package without global
provider enablement or discovery during ordinary import/replay. To override it,
pass `adapter_path` or set `TI_VKFFT_LIBRARY_PATH`; an invalid explicit override
fails without falling back to a bundled library. The legacy
`ti.hardware.fft.is_available()` / `cache_statistics()` remain cuFFT-only;
`ti.hardware.probe("vkfft", ...)` checks the adapter ABI without creating a plan
or qualifying a device/workload. Passive reports inspect only known open plans.

This slice supports in-place compact C2C f32, rank 1--3 and explicit batching.
Dimensions may contain only prime factors 2, 3, 5, 7, 11 and 13; larger factors
await upstream error-cleanup qualification, not a speed threshold. The default
`normalization="none"` leaves both directions unnormalized; `"inverse"` divides
the inverse by the transform volume. Storage, shape, direction and normalization
are frozen per plan. A Graph binding must reference the original array.

Plan creation may JIT and synchronize lookup-table initialization. Replay uses a
retained secondary GPU command sequence with one root-ordered host call per FFT
action by default; this is not enclosing native Graph capture or
`ti.linalg.record_fft()`'s CUDA out-of-place contract. Closing a plan rejects
future calls but already submitted command buffers retain their resources.
Requested allocations exclude caller storage and opaque driver objects; neither
closing the handle nor the initialization allocation peak proves device VRAM
retirement/peak. No production speedup or all-driver compatibility is claimed.

### Explicit Vulkan FFT recipe search

Default `plan.record()` nodes require the original plans and compact ndarrays
to remain open during search. For recipe-owned plans, append
`plan.record(recipe_owned=True)`, freeze the Graph, then close the original plans.
The frozen definition retains expected plan facts and caller storage, not native
FFT plans. Baseline compile, partial replacements and complete recipes recreate
only their selected plans at materialization; adapter and baseline physical facts
are checked there, never at replay. Existing live recordings are not detached or
closed implicitly, and ordinary `record()` lifetime semantics are unchanged.
Add `ti.hardware.fft.VulkanFftRecipeProvider()` alongside
`ti.graph.default_recipe_providers()` in `definition.search_recipes(...)`.
The existing complete-recipe evaluator, named metrics, report, checkpoint and
`resolve_recipe()` interfaces apply unchanged; CompileIQ sees only complete IDs.

The provider composes two physical mechanisms:

- Independent FFT batches may share one tile's scratch, with a separate tail
  application when needed. More repeated dispatches can cost device time; small
  transforms may have no scratch saving at all. Tile choices remain internal.
- A mixed, straight-line buffer Graph can be recorded as one secondary sequence
  and embedded in the runtime's ordered primary. `Graph.bind()` prepares fixed
  argument images and native commands once per published version. Replay does
  not invoke Python FFT actions, reupload arguments, or create a separate queue
  submission per segment. Raw mapping calls explicitly include preparation.

Batch recipes require the adapter's optional recipe extension. Whole-Graph
recording additionally requires its optional inline-record symbol and a matching
native bridge; an older adapter is not silently treated as that physical recipe.
One workspace lane and publication-qualified owned bindings are supported, not
SNode/texture/host-return kernels or device-controlled Graph topology. Native
command ownership retains arrays, argument images and FFT resources until their
parent submission retires, including when a materialized Graph is closed early.

New processes first rebuild equivalent baseline facts and storage, then resolve
the selected recipe. Obtaining these facts still requires initial plan creation;
recipe-owned recordings allow those source plans to be closed after freeze and
before search/resolve. This is not FFT binary serialization, a global plan cache,
or zero-cost baseline restoration. Concurrent materialized Graphs own independent
plans; close/drop unused Graph owners to release their plan leases. Caller
baseline allocations, plan-requested scratch, per-binding argument bytes and
unknown driver command/pipeline memory are distinct costs. Windows local tests
cover the implementation; Linux and production performance are not qualified by
those tests. Ordinary runtime selection is unchanged.

## Explicit FidelityFX Parallel Sort (Vulkan)

```python
keys = ti.ndarray(ti.u32, shape=1_048_579)
values = ti.ndarray(ti.u32, shape=1_048_579)
# Populate both arrays before running the plan.
with ti.hardware.sort.VulkanParallelSortPlan(
    keys, values, compiler_path=r"C:\VulkanSDK\<version>\Bin\dxc.exe"
) as plan:
    plan.run()  # Asynchronous, ascending stable sort; modifies original arrays.
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    bindings = graph.bind({"keys": keys, "values": values})
    graph.run(bindings)
    facts = plan.statistics()
    memory = plan.memory_report()
```

This bounded API supports nonempty, fixed, one-dimensional u32 keys and an
optional distinct u32 payload of the same shape. Without payload, omit `values`
and bind only `keys`. Stability preserves the order of equal keys. The source
preloads are guarded for exact-capacity storage, including non-512-sized tails;
the caller need not pad buffers. It does not add signed/floating keys, descending
order, dynamic device counts, or arbitrary payload layouts.

Forge ships the MIT FidelityFX Parallel Sort source and its own bindings. Supply
a DXC executable with SPIR-V support explicitly; JIT compilation happens during
plan creation. No FidelityFX framework, CUDA toolkit, extra Vulkan runtime, or
runtime compiler is bundled or loaded implicitly. The native bridge uses the
active Forge Vulkan device, queue, buffers, and completion-retention machinery.
The device must support compute subgroup basic/arithmetic/ballot/shuffle.
Missing compiler/native bridge/capabilities fail at explicit creation, without
changing ordinary sort. `ti.hardware.capability("sort.radix.fidelityfx")` is a
static contract, not a compiler or device qualification probe.

Pipelines, descriptors, workspace, and the forty-dispatch secondary sequence
are prepared once. Root Graphs retain a host call per sort action; this is not
enclosing Graph capture or a new CompileIQ fixed-sort/provider-route axis.
Binding publication requires the original arrays. Closing a plan invalidates
future calls; already submitted commands retain their GPU resources. Runtime
reset cannot redirect an old handle to a new plan.

There is no host staging, readback, or terminal device copy in the sort sequence.
Requested workspace is one key scratch, optional payload scratch, and compact
histogram/scan tables; reports exclude caller storage, allocator padding, opaque
driver allocations and physical VRAM peak. The lower workspace and retained host
recording can trade off against more device work than Forge's default radix8
implementation. Local Windows comparisons found device regressions, so ordinary
`ti.algorithms.sort` remains unchanged. No production speedup, AMD-wide advantage,
Linux qualification, or complete-recipe search support is implied.

## CUTLASS C++ complete matmul addon

`ti.hardware.source_providers.CutlassMatmulRecipeProvider` discovers prepared
`ti.linalg.record_matmul` regions and uses the existing addon C ABI to capture
complete CUDA Graph regions at a cold boundary. It is not an ordinary matmul
auto route. CompileIQ sees complete recipes, not library names, tiles, or split
counts. Forge's main compilation path remains unchanged; NVCC builds only the
explicit, separately owned addon.

The current scope is one matrix operation on compact scalar f32 ndarrays,
either operand's transposed storage, identity/ReLU, and
`D := activation(alpha * op(A) @ op(B) + beta * D)`. Inputs stay live across
replay; output must not alias either input. SIMT f32 does not silently switch
to TF32, but split-K reassociates accumulation. Applications must validate finite
inputs against their declared tolerances. Batching/rank extensions, other dtypes,
arbitrary epilogues, and the CuTe Python DSL are outside this addon.

| Complete physical strategy | GPU stages | Current requested scratch |
| --- | --- | --- |
| Direct fused | GEMM + epilogue, one kernel | Zero |
| Lower-workspace split-K | Partial products → reduction + epilogue, two kernels | `64 * m * n` bytes |
| Wider-parallelism split-K | More partial products → reduction + epilogue, two kernels | `512 * m * n` bytes |

These sizes describe the implementation, not a stable kernel-configuration API.
The explicit `workspace_limit_bytes` candidate budget defaults to 32 MiB.
Strategies exceeding it are not generated. More partitions can increase global
memory traffic and reduction work enough to outweigh the parallelism benefit.

Building requires caller-owned CUTLASS C++ sources, compatible CUDA Toolkit/NVCC,
and a host compiler. Windows local checks used CUTLASS 4.6.2. For example, from an
MSVC-configured shell in a source checkout:

```powershell
python python/taichi_forge/hardware/source_providers/cutlass/build.py `
  --cutlass-root D:/dependencies/cutlass-4.6.2 `
  --nvcc C:/CUDA/bin/nvcc.exe --target-code sm_120 `
  --output D:/addons/cutlass
```

`sm_120` is an example target, not a universal deployment choice. The builder
downloads nothing and does not instantiate the full CUTLASS profiler. It emits
the standalone binary, source-provider manifest, and `CUTLASS-LICENSE.txt`.
The manifest binds the entire header tree, binary, NVCC/PTXAS, static CUDART,
SM/PTX targets, and declared driver path, without pinning Forge's commit HEAD.
Static linkage does not remove Toolkit/driver compatibility requirements. The
addon is not a mandatory portable-wheel dependency and is not loaded on import.

The compiled addon itself does not require a runtime Toolkit compiler, but its
GPU and driver must satisfy the manifest. This integration reuses cuBLASLt-based
semantic preparation and the existing baseline, so a complete search still needs
a compatible, user-configured cuBLASLt runtime. The entire example is therefore
not a driver-only deployment.

```python
from taichi_forge.hardware.source_providers import CutlassMatmulRecipeProvider

provider = CutlassMatmulRecipeProvider(
    "D:/addons/cutlass/cutlass_source_provider.json",
    workspace_limit_bytes=32 << 20,
)
operation = ti.linalg.record_matmul(
    m, n, k, activation="relu",
    absolute_tolerance=5e-5, relative_tolerance=5e-5,
)
operation.prepare(heuristic_limit=2)
builder = ti.graph.GraphBuilder()
builder.append_native(operation)
definition = builder.freeze()
providers = (
    *ti.graph.default_recipe_providers(),
    ti.hardware.linalg.MatmulRecipeProvider(),  # retain existing alternatives
    provider,
)
session = definition.search_recipes(providers=providers, target=target, budget=budget)
decision = session.run(evaluator)  # application-owned comparable observations
```

Reports retain complete-recipe identities, resource facts, and source-build
provenance. Resolving a saved selection requires a compatible provider again.
Library/ABI/shape/alias checks, scratch allocation, and C ABI calls occur during
preparation, binding, or capture, not steady replay. Replay has no Python provider
callback or added synchronization. Requested scratch is not total VRAM: driver
module state remains unknown, and the runtime allocator may retain a retired
trial's high-water allocation for reuse. Local evidence shows size-dependent
tradeoffs, not superiority over all cuBLASLt strategies. Defaults are unchanged;
no Linux, release-qualification, or production-acceleration claim is added.

## Official references

- [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuSPARSELt getting started](https://docs.nvidia.com/cuda/cusparselt/getting_started.html)
- [cuTENSOR documentation](https://docs.nvidia.com/cuda/cutensor/index.html)
- [AmgX source and build guide](https://github.com/NVIDIA/AMGX)
- [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
- [OptiX SDK downloads and release requirements](https://developer.nvidia.com/designworks/optix/download)
- [CUDA compiler Advanced Controls](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [NVIDIA CompileIQ](https://developer.nvidia.com/cuda/compileiq)
- [FidelityFX Parallel Sort source and MIT license](https://github.com/GPUOpen-Effects/FidelityFX-ParallelSort)
- [CUTLASS C++ Windows build](https://docs.nvidia.com/cutlass/4.6.2/media/docs/cpp/build/building_in_windows_with_visual_studio.html)
- [CUTLASS 4.6.2 source and BSD-3-Clause license](https://github.com/NVIDIA/cutlass/tree/v4.6.2)
