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
| VkFFT 1.3.4 | Optional ABI1 Vulkan JIT adapter | Current runtime build configuration; older artifacts may omit it | `ti.hardware.probe("vkfft")` or explicit library path | Fixed-storage `VulkanFftPlan` or root Graph; no FFT recipe search |
| cuDSS 0.8.x | Registered bundled-adapter ABI | Forge adapter; user vendor runtime | `ti.hardware.probe("cudss", library_path=...)` | Domain auto/explicit or root Graph; not kernel-callable |
| OptiX ABI 93/105/118 | Registered bundled-adapter ABI | Forge adapter; user/driver vendor runtime | `ti.hardware.probe("optix", library_path=...)` | Explicit scene/launch or root Graph; not kernel-callable |
| Vulkan driver/ICD | D0 backend dependency, not a D1 provider | OS/GPU driver installation | `ti.init(arch=ti.vulkan)` plus capability queries | Kernel and documented native Vulkan APIs |
| cuSPARSELt 0.8.x-0.9.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.probe(...)` or `ti.hardware.tensor.CusparseLtProvider` | Explicit FP16 2:4 matmul plan; no Graph/kernel/auto route |
| cuTENSOR 2.0.x-2.7.x | Registered bundled-adapter ABI | Forge adapter; user optional package | `ti.hardware.tensor.CutensorProvider` / `ti.linalg.record_contraction` | Retained root Graph capture and complete contraction dataflows; no kernel intrinsic or implicit auto rewrite |
| AmgX stable C API | Registered bundled-adapter ABI | Forge adapter; user source build | `ti.hardware.probe(...)` or `ti.hardware.linalg.AmgxProvider` | Explicit host-CSR solver; no Graph/kernel/auto route |
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
| Vulkan VkFFT | Explicit fixed-storage plan or root Graph recording | Vulkan JIT/source adapter; no built-in complete FFT-recipe search or CUDA binding-frame integration is implied. |
| cuBLASLt matmul region | `ti.linalg.record_matmul(...)`, then `operation.prepare()` | CUDA compact scalar-f32, fixed shape and optional strided batch. Explicit `ti.hardware.linalg.MatmulRecipeProvider()` composes frozen algorithm/workspace choices, real operand packing, and separate/fused ReLU. The expert retained-plan API remains private. |
| cuTENSOR contraction region | `ti.linalg.record_contraction(...)`, then `operation.prepare()` | Explicit `ti.hardware.tensor.ContractionRecipeProvider()` composes real input permutations and vendor/separate epilogues; includes retained workspace and immutable binding frames. |
| cuSPARSELt / AmgX | Explicit provider plans described below | No complete-recipe provider or general Graph recording route is currently exposed. |

Prepare mathematical operations before freezing the Graph. SpMM, FFT, matmul and contraction require
explicit finite-input / f32 tolerance contracts; Forge does not scan values on
each replay. FFT forward and inverse are both unnormalized, so applying both
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

Follow the selected release's support table. Current cuSPARSELt documentation
requires compute capability 8.0 or newer and, for the current release line, a
CUDA 12.9-or-newer software stack with a compatible driver. Older package
releases have different requirements; package availability is not a
compatibility test.

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
algorithm_search: offline_opt_in
min_matmuls_per_compression: 4
fallback_owner: application
```

Treat `min_matmuls_per_compression: 4` only as a conservative starting gate;
qualify the exact workload and increase it when compression or setup dominates.
Run `cusparseLtMatmulSearch()` only as offline/initialization autotuning for an
operation that will be repeated. Do not put search, pruning, compression, or
plan creation in the simulation step loop.

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

Require numerical acceptance first. Then require a positive worst-case timing
result for the expected reuse count. High timing variance is acceptable only
when the slowest qualified sample still clears the application gate.

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
intrinsic. Build it as an application dependency from an NVIDIA release whose
CUDA and architecture support matches the deployment:

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

Execution accepts contiguous host scalar CSR arrays and host vectors. AmgX
owns device upload, hierarchy, and solver resources; keep the solver when the
topology is reused and provide the exact application-owned configuration:

```python
with ti.hardware.linalg.AmgxProvider(runtime_path) as provider:
    with provider.solver(offsets, columns, values, "PCG_V.json", config_file=True) as solver:
        solution, info = solver.solve(rhs)
        assert info["converged"]
```

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
| Correctness differs | matrix properties, pruning/precision, transpose/layout, stale plan or values | Fail the numerical gate before considering performance |
| First call is slow | plan creation, JIT, analysis, compression, allocation | Separate setup from steady state and use the production reuse count |
| Memory grows | live plans/scenes/factors, workspaces, caches, in-flight Graph leases | Inspect provider memory reports where available and close owners after retirement |
| Performance is unstable | synchronization, cold caches, clock/power state, algorithm search, topology | Use fresh-process AB/BA runs and require a positive worst-case application gate |

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
action; this is not enclosing native Graph capture, `ti.linalg.record_fft()`'s
CUDA out-of-place contract, or a new CompileIQ route. Closing a plan rejects
future calls but already submitted command buffers retain their resources.
Requested allocations exclude caller storage and opaque driver objects; neither
closing the handle nor the initialization allocation peak proves device VRAM
retirement/peak. No production speedup or all-driver compatibility is claimed.

## Official references

- [cuDSS documentation](https://docs.nvidia.com/cuda/cudss/index.html)
- [cuSPARSELt getting started](https://docs.nvidia.com/cuda/cusparselt/getting_started.html)
- [cuTENSOR documentation](https://docs.nvidia.com/cuda/cutensor/index.html)
- [AmgX source and build guide](https://github.com/NVIDIA/AMGX)
- [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
- [OptiX SDK downloads and release requirements](https://developer.nvidia.com/designworks/optix/download)
- [CUDA compiler Advanced Controls](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [NVIDIA CompileIQ](https://developer.nvidia.com/cuda/compileiq)
