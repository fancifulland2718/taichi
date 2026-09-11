# Taichi Forge API Reference

> Covers the **published Taichi Forge 0.6.2 contract** and public APIs marked
> **0.6.3 in development**. This page lists Forge-only public API entry points.
> New options added to Taichi-compatible APIs, such as
> `ti.init(...)` keywords and `@ti.kernel(...)` keyword options, stay in
> [Forge options](forge_options.en.md).
> API introduction versions are indexed separately in
> [release notes](release_notes.en.md). Experimental, in-development, and
> source-build-only labels continue to limit support.

Taichi Forge keeps the vanilla Taichi DSL model, but adds APIs for compile
control, native device primitives, graph replay, display submission, sparse
layout experiments, and diagnostics. The APIs below are grouped by module and
list their call position, parameters, and current limits.

## `taichi_forge` Top-Level APIs

Import Forge as:

```python
import taichi_forge as ti
```

### `ti.hardware` capabilities and explicit probes (0.6.3 in development)

The stable ordinary-user layer consists of `HardwareCapability`,
`HardwareProviderStatus`, and `HardwareExecutionReport`. Use `status(id)`,
`provider_status(id)`, or `execution_report()` to read only `available`, active
`backend`, `selected_provider`/`selected`, hardware qualification `route`, and
the structured `reason`. These passive calls never load, enable, benchmark, or
select an optional provider.

`operations()`, `capability(operation_id)`, `providers()`, and `report()` are
schema-v4 diagnostic contracts. They additionally expose planner/provider
details such as dependency tier, workspace/lifetime/update policy, ABI,
performance scope, resource effects, and native facts. Every operation has an
`activation_mode` that separates `explicit_hardware_api`,
`explicit_kernel_intrinsic`, `domain_api_auto_provider`, and
`compiler_automatic`. `graph_integration` independently reports `inline`,
`root_ordered`, `backend_recorded`, `stream_captured`, `opaque`, or
`unsupported`. In particular, `root_ordered` means that a Forge root Graph
preserves ordering and lifetime while replay calls the provider again; it does
not claim CUDA Graph capture or a persistent Vulkan command buffer.

`probe(provider_id)` explicitly probes a D1 `lazy_external` provider. The
cuBLAS, cuSPARSE, and cuFFT probes check exact symbols through a transient
native handle. cuDSS uses the wheel-internal Forge adapter to transiently load
and validate the user vendor runtime. Bundled adapters similarly audit
cuSPARSELt 0.8.x-0.9.x, cuTENSOR 2.0.x-2.7.x, and the AmgX stable C API. Their
separate provider objects expose explicit execution plans, but no automatic,
Graph, or kernel route. Probe handles close before return
and no probe changes later selection. If an actual algorithm already owns a
plan, passive reports observe that state without invoking the loader. Unknown
operations/providers and unimplemented probes fail closed.

Installation ownership, version binding, loader configuration, and recommended
selection gates for user-managed libraries are documented in
[Optional external hardware providers](external_hardware_providers.en.md).
That guide distinguishes execution providers, non-executing probes, and
native-adapter candidates; installing a runtime never selects an algorithm.

### Core kernel hardware routes (0.6.3 qualification)

The hardware catalog also describes existing D0 CUDA/Vulkan kernel routes.
They add no dependency, wheel variant, Python native executable, or new DSL
syntax. The call boundary is part of the contract:

- Explicit inside `@ti.kernel`: `ti.atomic_*`, `ti.simt.warp` on CUDA,
  the implemented `ti.simt.subgroup` subset on Vulkan,
  `ti.simt.block.SharedArray`, and `ti.block_local`. These operations lower
  inline and cannot be called from Python as native commands. Unsupported
  dtype/operation/device combinations fail during admission or compilation.
- Automatic compiler/runtime selection: recognized CUDA block and Vulkan
  subgroup reductions aggregate lane values before publishing fewer global
  atomics. Vulkan list generation can use subgroup ballot aggregation only
  when `spirv_listgen_subgroup_ballot=True` and the device qualifies. Both
  retain their ordinary atomic fallbacks and expose no public call.
- Layered selection: `ti.block_local` is an explicit cache hint, while the
  later choice of PTX `cp.async` for a qualified read-only copy is automatic.
  The current block-local qualification covers supported gather/read-cache
  patterns; sparse pointer-SNode scatter/write-back is excluded after a
  reproducible CUDA correctness failure.

All catalog entries use `hardware_acceleration="implementation_defined"`:
code generation may choose native atomics, CAS, subgroup instructions, or
shared memory according to the exact operation and device. An API name alone
is not evidence of a particular instruction.

### `ti.hardware.linalg.gemm_f32` (0.6.3 in development)

An explicit D1 cuBLAS provider for compact row-major f32 matrices:

```python
if ti.hardware.linalg.is_available():  # explicit transient provider probe
    ti.hardware.linalg.gemm_f32(a, b, output, alpha=1.0, beta=0.0)

    recording = ti.hardware.linalg.CublasGemmRecording(
        rows, columns, inner, alpha=1.0, beta=0.0
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    builder.compile().run({"a": a, "b": b, "output": output})
```

The operation is `output = alpha * a @ b + beta * output`; inputs have shapes
`(rows, inner)` and `(inner, columns)`, and output has shape
`(rows, columns)`. All arrays are distinct, compact scalar f32 ndarrays on the
current CUDA runtime. Execution uses a column-major cuBLAS SGEMM with swapped
operands to preserve the documented row-major result, binds the Program's
default CUDA stream, and reuses one handle per Program.

The compatible cuBLAS shared library is lazy-loaded only by real execution;
`cublas_is_available()` uses the explicit transient probe (`is_available()` is
its compatibility alias), and passive `report()`
does not load it. Forge copies the stable ABI declarations but includes no
Toolkit header, link dependency, bundled library, package dependency, build
switch, or wheel variant. The API is direct/root-Graph only: it is not
kernel-callable and ordinary matrix multiplication is never rewritten.

### CUDA sparse provider selection, explicit SpMV, and cuDSS (0.6.3 qualification)

Two existing `ti.linalg` domain APIs already select optional vendor libraries
on CUDA:

- `SparseMatrix.spmv(ndarray, method="auto")` considers f32 scalar-CSR
  cuSPARSE SpMV (and the qualified fixed-block BSR slice). Without matching,
  stable matrix-scoped cost evidence, automatic selection fails closed to the
  embedded CUDA CSR/BSR kernel. `method="provider"` remains the explicit
  provider request. The matrix resource caches its provider
  handle, dense-vector descriptors, workspace, and optional SpMV preprocessing
  across calls. This automatic domain route remains direct Python execution.
- `SparseMatrix @ ndarray` uses the same automatic route.
- `ti.hardware.load_provider_admission_evidence(path, case="cuda-spmv")`
  accepts only a strict Forge fresh-process qualification artifact. Attach the
  immutable result with `matrix.set_provider_profile(profile)`, or pass `None`
  to clear it. Admission-schema-v2 profiles match the exact sparse-topology
  fingerprint, device UUID, Python extension, split native runtime and runtime
  bitcode build identity, Python provider-contract hash, provider ABI/version,
  and amortization scope; older profiles fail closed. Users cannot install
  handwritten timing fields. `matrix.set_provider_profile(profile,
  expected_reuse=N)` may supply the current positive reuse count and
  re-evaluates both provider and embedded-baseline first-use costs; omitting it
  adopts the evidence assumption.
- `ti.hardware.linalg.spmv_f32(matrix, input, output)` executes the same stored
  matrix operation into caller-owned output. `CusparseSpmvRecording(matrix)`
  makes that manual operation a root-Graph backend command with explicit
  read/write effects and a matrix-generation lifetime lease. It re-records one
  runtime-ordered cuSPARSE command per Graph run and reuses the matrix-owned
  handle, descriptors, workspace, and preprocessing.
- `ti.hardware.load_provider_admission_evidence(path,
  case="cuda-cudss-solve")` can supply the exact-workload profile to
  `ti.linalg.SparseSolver(provider="auto", provider_profile=profile)`. Auto
  considers user-managed cuDSS 0.8.x only after the profile additionally
  matches both the wheel-internal adapter and loaded vendor-runtime binaries,
  the solver contract, and expected solve
  reuse, and its amortized cost clears the conservative margin against
  cuSOLVERSp. `expected_reuse=N` on `SparseSolver` may replace the evidence
  assumption for the current workload and re-evaluates both routes. Both providers'
  first analysis/factorization cost is
  included. Without a profile, auto does not probe or load cuDSS and stays on
  embedded cuSOLVERSp. CUDA 11 and older, missing/incompatible cuDSS, f64,
  ineligible layouts, or a scope/cost mismatch also retain cuSOLVERSp.
  `provider="cudss"` and `provider="cusolver_sp"` are explicit,
  non-performance-gated choices.
- `ti.hardware.linalg.CudssPlan` exposes explicit `analyze()`, `factorize()` /
  `refactorize()`, and `solve(rhs, solution)` stages. `CudssSolveRecording` is
  a `root_ordered` action over factors produced from the plan's stored matrix:
  each root-Graph run reissues one runtime-ordered solve.
  `plan.record_refactor_solve(values="matrix_values", rhs="rhs",
  solution="solution")` instead records one fixed-pattern f32 action that
  reads current CSR values and the right-hand side, invalidates the old
  factors, refactorizes, and writes the solution as one provider transaction.
  The values, rhs, and solution arrays must be distinct. A plan permits only
  one unretired refactorize-and-solve transaction; overlapping reuse fails
  closed. A provider execution failure invalidates the factors and keeps the
  transaction reserved until submission retirement; after retirement the same
  action can recover by fully factorizing the bound values. Factors produced
  from explicit Graph values cannot be consumed by a
  later solve-only recording until the stored matrix is refactorized. Neither
  recording is stream capture or kernel-callable, and both are unsupported in
  structured Graph and AOT. Explicit cuDSS also requires CUDA Driver API 12.0
  or newer.

All routes lazy-load user-provided compatible libraries only when their domain
objects or explicit plans are used. An explicit cuDSS `library_path` is
exclusive: probing and execution do not fall back to an environment or default
library candidate. Forge does not install cuDSS, add a Python
package requirement, link or bundle a vendor library, or create a wheel
variant. The existing runtime wheel carries the Forge-owned thin C-ABI adapter;
public `library_path` always names the vendor runtime and cannot override that
internal adapter. This is domain-level or explicit-plan selection, not a
compiler rewrite of arbitrary kernels. None is kernel-callable. Provider
analysis, factorization, and numerical failures after explicit selection stay
visible rather than silently falling back.

### `ti.hardware.tensor` and `ti.hardware.linalg.AmgxProvider` (0.6.3 in development)

Three user-runtime adapters expose explicit retained execution resources:

- `CusparseLtProvider(library_path=None).matmul_plan(m, n, k)` creates an FP16
  row-major 2:4 plan. Call `compress(a)` for an already-valid sparse `(m, k)`
  operand, then `execute(b_transposed, c, d, alpha=..., beta=...)`.
  `b_transposed` has shape `(n, k)`; dimensions are multiples of 16.
- `CutensorProvider(library_path=None).contraction_plan(...)` creates a compact
  scalar-f32 contraction from explicit shape/mode pairs for A, B, C, and D.
  `compute="f32"` and `compute="tf32"` are supported; the plan reports and owns
  its exact workspace.
- `AmgxProvider(library_path=None).solver(row_offsets, column_indices, values,
  config, config_file=False)` creates a scalar f32/f64 host-CSR solver.
  `solve(rhs, solution=None, zero_initial_guess=True)` returns the host solution
  and immutable convergence facts; `replace_coefficients(values)` reuses the
  fixed topology and refreshes solver setup. The adapter uses the optional
  vendor `AMGX_solver_resetup` fast path when exported and otherwise falls back
  to a full `AMGX_solver_setup`; hierarchy reuse remains controlled by the
  application-owned AmgX configuration.
  Numeric values may also be CUDA scalar ndarrays. `bind_device(rhs, solution,
  values=None, zero_initial_guess=True)` returns an `AmgxDeviceBinding` whose
  `solve()` and explicit `replace_coefficients()` reuse fixed device buffers.
  Topology remains host i32 CSR; binding does not imply asynchronous execution,
  zero-copy storage, Graph recording or automatic search.

All require an initialized CUDA runtime and load the user library only when the
provider is constructed. Child plans/solvers must close before their provider;
all objects become invalid after `ti.reset()`. AmgX remains a direct Python
call with no Graph route. cuSPARSELt and cuTENSOR also have separate explicit
recording interfaces; none becomes an automatic provider choice merely by
installing its runtime. See the external-provider guide for installation, path,
numeric, and memory gates.

### `ti.graph.VulkanBufferCommand` and `VulkanBufferCommandRecording` (0.6.3 in development)

These D0 APIs describe and submit one Vulkan RHI buffer-command sequence:
`fill_u32()`, `copy()`, `buffer_barrier()`, and `memory_barrier()`. Execute it
manually with `recording.execute(bindings)`, or append it to a root Graph with
`GraphBuilder.append_native(recording, admission="auto")`.

The current contract accepts runtime-owned Vulkan `ti.ndarray`, canonical dense
fields, and compact program-owned `ti.experimental.ndarray_view` ranges. Byte
offsets are relative to the supplied view; disjoint ranges in one allocation
are permitted, but overlapping copies and noncompact views are rejected. No
implicit packing or field-to-ndarray copy is performed. `graph.bind()` prepares
fixed command packets without executing them; layout/range checks occur there,
not on each qualified replay. Tree destruction or runtime reset invalidates old
bindings; content-only updates do not require rebinding.

The contract requires four-byte-aligned ranges, explicit barriers, the runtime-ordered compute queue,
and `rerecord` replay. It is not callable inside `@ti.kernel`, is not qualified
inside a structured `Sequential` or for AOT serialization, and is not itself a
RasterPass or AS provider. Bounds errors, overlapping copies, backend/device
mismatches, Graph use after reset, and recordings over 4096 commands fail.
The route uses only D0 runtime code already present in official wheels and does
not add a wheel variant.

### `ti.hardware.image.VulkanImageCopyRecording` (0.6.3 in development)

This D0 low-level command copies one complete, format- and extent-matched
Vulkan color `ti.Texture` into another:

```python
ti.hardware.image.copy(destination, source)

recording = ti.hardware.image.VulkanImageCopyRecording(
    source="input", destination="output"
)
builder = ti.graph.GraphBuilder()
builder.append_native(recording, admission="auto")
graph = builder.compile()
graph.run({"input": source, "output": destination})
```

The recording declares an exact source READ and destination WRITE, retains
runtime Texture leases through asynchronous submission, and orders preceding
and following kernels without host readback. Aliasing, different formats or
extents, depth/stencil images, stale runtimes, and non-Vulkan backends fail
closed. It records the native Vulkan image-copy path, but Vulkan does not
promise which physical engine executes a copy, so the catalog labels hardware
acceleration `implementation_defined` rather than claiming a dedicated copy
unit. Offset/region copies, buffer-image transfers, blits, and raw public
layout transitions remain deferred until their bounds, format, and effect
contracts are complete. No dependency or wheel variant is added.

### Vulkan `ti.Texture` hardware-sampling qualification (0.6.3 in development)

Explicit `ti.Texture.sample_lod()` and `fetch()` calls inside Vulkan kernels
are automatically lowered to SPIR-V image/sampler instructions. Query the
contract with `ti.hardware.capability("sampling.texture.vulkan")`. The current
slice covers 1D/2D/3D sampled textures and format-matched `rw_texture` storage
images. `ti.hardware.sampling.SamplerConfig` selects immutable min/mag filter
and U/V/W address state at texture creation:

```python
sampler = ti.hardware.sampling.SamplerConfig(
    min_filter="nearest",
    mag_filter="nearest",
    address_mode_u="clamp_to_edge",
    address_mode_v="clamp_to_edge",
)
grid = ti.Texture(ti.Format.r32f, (nx, ny), sampler=sampler)
```

The current choices are `nearest`/`linear` filtering and `repeat`,
`mirrored_repeat`, or `clamp_to_edge` independently per axis. Vulkan sampler
objects are cached by immutable configuration. Textures currently have one
mip level and normalized coordinates; anisotropy and comparison sampling are
not exposed. `sample_lod()` uses the sampler while exact integer-coordinate
`fetch()` ignores it. Floating filtering does not promise cross-device
bitwise determinism.
Ordinary field or ndarray access is never converted to texture sampling. CUDA
also supports explicit sampled textures over Driver-API arrays/texture objects,
including cached Graph capture with generation-owned sampler lifetimes. Fixed
`Graph.bind()` validates Texture type and dimensions before publication. In-place
uploads preserve the capture; replacing the Texture selects or captures another
bounded executable slot. Uploads retain their existing explicit synchronization
behavior; this does not introduce a captured upload command. CUDA RW textures,
texture AOT serialization, and the separate immutable-binding-frame recipe remain
outside this support statement. No wheel variant is added.

### `ti.hardware.graphics.VulkanGraphicsPipeline` (0.6.3 in development)

This is the low-level, renderer-neutral Vulkan raster interface. The caller
provides SPIR-V shader binaries, exact vertex layouts, runtime-owned
`ti.ndarray` vertex/index buffers, and runtime-owned `ti.Texture` color/depth
attachments:

```python
with ti.hardware.graphics.VulkanGraphicsPipeline(
    vertex_spirv,
    fragment_spirv,
    vertex_bindings=(ti.hardware.graphics.VertexBinding(0, 20),),
    vertex_attributes=(
        ti.hardware.graphics.VertexAttribute(0, 0, ti.Format.rg32f, 0),
        ti.hardware.graphics.VertexAttribute(1, 0, ti.Format.rgb32f, 8),
    ),
) as pipeline:
    pipeline.draw(
        color_target,
        {0: vertices},
        draw=ti.hardware.graphics.Draw(3),
    )
```

`record()` creates an automatically admissible root-Graph backend-command
node with exact read effects for vertex/index buffers and write effects for
attachments. Kernel work that produces geometry, graphics submission, and
later kernel reads are ordered without an implicit host wait. The API exposes
pipeline state and draw recording only: it has no camera, material, light,
scene, mesh ownership, shader compiler, presentation, or renderer policy.
It is Vulkan-only, is not callable from `@ti.kernel`, accepts precompiled
SPIR-V rather than adding a shader-toolchain dependency, and adds no official
wheel variant. Driver-owned pipeline and shader-module memory is reported as
opaque rather than estimated.

### `ti.hardware.raster.RasterPass` (0.6.3 in development)

A compatibility and qualification adapter over the existing GGUI renderer:

```python
with ti.hardware.raster.RasterPass((1280, 720)) as raster:
    raster.set_camera(camera).ambient_light((0.2, 0.2, 0.2))
    raster.mesh(vertices, indices=indices, normals=normals, two_sided=True)

    recording = raster.record()  # freezes draw topology and resource bindings
    recording.execute()          # graphics submission without host readback
    rgba = raster.color_numpy()  # separate explicit synchronous observation

    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()     # explicit segmented root-Graph execution
```

`mesh_instance()`, `particles()`, `lines()`, `point_light()`, and `clear()` are
also available. A recording binds the same resource objects but rereads their
latest contents on replay. The current slice is Vulkan-only and uses a hidden
window's provider-owned color/depth targets plus built-in GGUI shaders. One
execution can be consumed by either `color_numpy()` or `depth_numpy()`, not
both. This is a Python native executable and is not kernel-callable.
An explicit root `GraphBuilder.append_native(..., admission="explicit")`
records it as an opaque segmented node and preserves ordering and lifetime;
it does not fuse GGUI helper dispatches into the enclosing backend Graph.
`admission="auto"`, structured Graph regions, and AOT reject it until helper
dispatches and provider-owned output bindings participate in an exact effect
contract. This D0 API adds no dependency or wheel variant.

### `ti.hardware.ray` BLAS/TLAS and batch query (0.6.3 in development)

The low-level D0 Vulkan hardware ray-query API separates fixed-topology
triangle BLAS resources from a fixed-order instance TLAS:

```python
with ti.hardware.ray.TriangleBLAS(vertices, indices) as blas:
    instances = (
        ti.hardware.ray.RayInstance(blas, transform=transform_a, mask=0xFF),
        ti.hardware.ray.RayInstance(blas, transform=transform_b, custom_index=1),
    )
    with ti.hardware.ray.InstanceTLAS(instances) as tlas:
        tlas.trace(rays, hits)
        tlas.refit(updated_instances)  # same BLAS count and order

        builder = ti.graph.GraphBuilder()
        builder.append_native(tlas.record(ray_count), admission="auto")
        graph = builder.compile()
        graph.run({"rays": rays, "hits": hits})
```

`TriangleBLAS.record_build()`/`record_refit()` and
`InstanceTLAS.record_build()`/`record_refit()` are `root_ordered` native
actions. A `RayInstance` contains one BLAS reference, a finite row-major 3x4
transform, an 8-bit mask, and a 24-bit custom index. TLAS build/refit preserves
the number and order of referenced BLAS resources; BLAS refit is vertex-only
and preserves vertex count and index topology. Batch query reads the TLAS and
writes caller-owned hits without host readback.

`TriangleScene` remains the single identity-instance compatibility wrapper:

```python
with ti.hardware.ray.TriangleScene(vertices, indices) as scene:
    scene.trace(rays, hits)

    # Vertex-only BLAS update; count and index topology stay fixed.
    scene.refit(updated_vertices)

    recording = scene.record(ray_count)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="auto")
    graph = builder.compile()
    graph.run({"rays": rays, "hits": hits})

    refit = scene.record_refit(vertices="positions")
    builder = ti.graph.GraphBuilder()
    builder.append_native(refit, admission="auto")
    builder.compile().run({"positions": updated_vertices})
```

Vertices are f32 and indices are i32, using scalar `(N, 3)` or AOS vector-3
layout. Rays are f32 `(N, 8)` values
`[ox, oy, oz, tmin, dx, dy, dz, tmax]`; hits are f32 `(N, 4)` values
`[t, primitive_id, instance_custom_index, hit_flag]`, with misses encoded as
`[-1, -1, -1, 0]`. Input indices must be nonnegative and in range; the provider
does not read them back to validate mesh topology.

For integer IDs and triangle barycentrics, both `TriangleScene` and `InstanceTLAS`
provide `trace_typed(rays, hits, hit_indices)` and `record_typed(ray_count)`.
The caller supplies non-overlapping device ranges: f32 `hits` `(N, 4)` contains
`[t, u, v, 0]`, and i32/u32 `hit_indices` `(N, 4)` contains
`[primitive_index, instance_index, instance_custom_index, hit_flag]`.
AOS vector-4 arrays are also accepted. `instance_index` is the zero-based TLAS
ordinal, not the caller's custom index; the single-instance wrapper uses zero
for both. Triangle vertex weights are `[1-u-v, u, v]`; `t` is a ray parameter,
not a metric distance unless the direction is normalized. Misses write
`[-1, 0, 0, 0]` and `[-1, -1, -1, 0]` respectively.
Use u32 for the full unsigned index range (absent indices are `UINT32_MAX`);
i32 interprets the same bits as signed integers.
Typed output is written directly by the ray shader. It uses 16 additional
caller-owned bytes per ray and no intermediate conversion buffer.
Append `scene.record_typed(count)` as a root native Graph node and bind
`rays`, `hits`, and `hit_indices`; subsequent kernels can consume both outputs.
This remains runtime-ordered native rerecording, not Vulkan command-buffer
replay. Legacy `trace` / `record` retain their float4 output contract.
Query inputs and outputs also accept canonical dense fields and compact
program-owned `ndarray_view` subranges, including disjoint regions of one
SNodeTree. Shape/dtype contracts are unchanged; there is no implicit packing.
Only 4-byte data alignment is required: descriptor offsets are aligned during
preparation and a subrange shader addresses the remaining offset. Unaligned
output uses 2 KiB (4 KiB typed) of workgroup memory to coalesce writes, not an extra
global conversion buffer. Tree destruction invalidates prepared query bindings.
Geometry construction, BLAS build/refit, and `TriangleScene.refit` accept the same
compact storage kinds with f32 vertices and i32 indices in scalar `(N, 3)` or
packed vector-3 `(N,)` layout. Subranges retain their byte offsets. Geometry is
copied device-to-device into the provider's retained build buffers; field inputs
do not require an intermediate ndarray or host readback. Fixed shape/counts are
qualified when preparing the native packet, not rescanned on bound execution.
`TriangleScene.refit` refreshes both its BLAS and one-instance TLAS bounds.
With independent owners, record BLAS build/refit followed by TLAS build/refit
before querying displaced geometry; Forge does not discover external TLAS users.
`graph.bind(...)` prepares immutable native packets without tracing rays or
building acceleration structures.
Reusing that binding does not revalidate shapes, dtypes, or non-aliasing;
`binding.update(...)` prepares a replacement. Provider close/reset and in-flight
resource retention remain enforced by the native owner.

The provider requires Vulkan 1.2 plus buffer-device-address,
`VK_KHR_acceleration_structure`, and `VK_KHR_ray_query`. Its SPIR-V shader is
embedded at build time, so it adds no Vulkan SDK runtime dependency and no
official wheel variant. These batch methods are not callable inside
`@ti.kernel`; inline queries instead use `ti.ray_query` with `InstanceTLAS`.
Neither route replaces ordinary kernels or collision detection automatically.
Topology-changing updates and procedural geometry remain outside this contract.

### `ti.hardware.fft.CufftPlan1D` / `CufftPlanND` (0.6.3 in development)

An explicit D1 single-GPU cuFFT provider for C2C, R2C, and C2R transforms:

```python
if ti.hardware.fft.is_available():  # explicit transient provider probe
    with ti.hardware.fft.CufftPlan1D(length, batch_count=batch) as plan:
        plan.execute(source, spectrum, direction="forward")

        recording = plan.record(
            direction="inverse", input="spectrum", output="signal"
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording, admission="auto")
```

Complex values use a final scalar-f32 pair axis storing `[real, imag]`; R2C/C2R
use the Hermitian `length // 2 + 1` extent. `CufftPlan1D` covers compact 1D
transforms, while `CufftPlanND` covers batched rank-2/rank-3 transforms with
explicit embed, element stride, and batch distance. Inverse C2C and C2R output
is unnormalized. The fixed-size plan owns cuFFT's internal workspace, binds the
runtime default CUDA stream, and can be replayed through a root Graph. Closing
the plan invalidates its recordings.

`is_available()` is an explicit transient probe. Actual provider enablement
happens only when a plan is created; a missing or incompatible library fails
that plan without disabling the CUDA backend. Forge neither bundles nor links
cuFFT, adds no mandatory package, and adds no official wheel variant. Users
install and expose a compatible shared library separately. In-place operation,
independently arbitrary per-axis strides, callbacks, LTO, multi-GPU, kernel
invocation, and AOT serialization remain unsupported.

### `ti.experimental.ndarray_view(source, *, slices=None, access="readwrite")`

Creates an explicit non-owning zero-copy dense storage view over a qualified
Forge `Ndarray`, `DenseNdarrayView`, or root-dense field. `slices` optionally
supplies one positive-step Python `slice` per logical index axis and preserves
rank. View composition combines byte offsets and per-axis strides without
allocation or copy.

The returned view is accepted by `ti.types.ndarray(...)` kernel arguments on
CPU, CUDA, and Vulkan. Compact internal storage is eligible for CUDA Graph
capture/replay; positive affine views use ordinary CUDA Graph execution and
Vulkan command record/replay. Unsupported layouts raise before submission;
there is no staging fallback. The current contract is read-write, binds no
gradient owner, and does not support ArgPack nesting, negative or broadcast
strides, overlap, axis permutation, integer indexing, or external ownership.
Ordinary compact ndarray calls and positive-affine views use separate compiled
specializations: compact calls keep canonical addressing in the generated
kernel, while affine views load the validated runtime offset/stride metadata.
Symbolic Graph ndarray arguments use the runtime-affine specialization so one
compiled Graph may safely bind either qualified layout.

See [Experimental zero-copy dense storage views](storage_views.en.md) for the
layout matrix, lifetime behavior, Graph paths, and examples.

### `ti.interop.from_dlpack(source, *, element_shape=(), access="readwrite", copy=False)`

Imports a qualified DLPack producer as a managed, strict zero-copy `ExternalDenseView`. CPU and CUDA-host storage are accepted by a CPU runtime; CUDA and CUDA-managed storage are accepted by a CUDA runtime. Vulkan, cross-device import, noncompact external affine layout, `copy=True`, and unsupported access modes raise instead of materializing a copy.

The returned view can be used as a compatible ndarray kernel argument and supports `close()` plus the context-manager protocol. It keeps the DLPack capsule owner alive through in-flight work and remains safe to close after runtime reset.

### `ti.interop.from_external(source, *, provider=None, element_shape=(), access="readwrite", copy=False)`

Provider-neutral managed-tensor entry point. `provider=None` and
`provider="dlpack"` currently delegate to the same strict implementation as
`from_dlpack()`. Passing an existing open `ExternalDenseView` returns that
view; reinterpretation options are rejected.

### `ti.interop.import_external_allocation(provider, memory_handle, **options)`

Imports a raw external allocation through a qualified provider. The current
`"vulkan_cuda"` provider requires `arch=ti.cuda`, a dedicated
Vulkan-exported buffer allocation, its byte size and 16-byte physical-device
UUID, plus paired binary-semaphore handles. `opaque_win32` is accepted on
Windows and `opaque_fd` on Linux. `allow_unsynchronized=True` permits an
explicit caller-synchronized import. No copy fallback is provided.

The returned `VulkanCudaExternalAllocation` creates compact AOS typed-offset
views with
`allocation.view(dtype=..., shape=..., element_shape=(), offset_bytes=0)`.
Views share one Graph access epoch. `import_vulkan_cuda_allocation()` remains
as the provider-specific compatibility spelling, and
`current_cuda_device_uuid()` returns the active CUDA device's 16-byte UUID.

### `ti.interop.capabilities()`

Preserves the schema-v1 top-level DLPack fields and adds
`interop_schema_version=2` plus a `providers` map. Provider records report
availability, accepted handles/layouts, synchronization modes, Graph access
epochs, and strict copy-fallback policy.

Historical NumPy, PyTorch, and Paddle kernel-argument signatures remain supported. The explicit interop API is strict; historical adapters preserve their established fallback behavior. See [Zero-copy dense storage and interoperability](zero_copy_interop.en.md) for the support matrix and synchronization contract.

### `ti.compile_kernels(kernels)`

Location: `taichi_forge.lang.misc`; exported as `ti.compile_kernels`.

Pre-materializes and submits a batch of kernel specializations before a hot
loop.

```python
ti.compile_kernels([
    init_kernel,
    (step_kernel, (positions, velocities, dt)),
    (render_kernel, (frame,), {"exposure": 1.0}),
])
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `kernels` | Iterable of `kernel`, `(kernel, args)`, or `(kernel, args, kwargs)` tasks. `args` must be a tuple/list and `kwargs` must be a dict. |

Returns: number of kernel specializations submitted.

Limits:

- The Python frontend still materializes each specialization on the calling
  thread because AST transformation mutates frontend runtime state.
- Arguments determine the specialization and cache key.
- Reuse is limited to the current runtime, arch, compile options, source hash,
  and backend cache partition.

Alias: `ti.parallel_compile(kernels)`.

### Extended `ti.ad.FwdMode` field seeds

Taichi Forge 0.6.0 accepts one dense `ScalarField`, `VectorField`,
or `MatrixField` as the `param` group. The field must have dual storage. `seed`
may be either:

- an array shaped as `param.shape + element_shape`, where `element_shape` is
  `()` for scalar, `(n,)` for vector, and `(n, m)` for matrix fields; or
- a flat sequence with the same number of values, interpreted in C row-major
  order with field indices before element indices.

This host contract is identical for AoS and SoA placement and covers 0-D and
N-D fields. The default seed is available only when the whole parameter group
contains one scalar value. `FwdMode` still accepts exactly one parameter group
per context; use multiple contexts for multiple groups. Loss entries remain
scalar fields.

### Automatic-differentiation order boundary

Forge currently supports first-order reverse AD through `ti.ad.Tape()` or a
manual `kernel.grad()` call, and first-order forward AD through
`ti.ad.FwdMode()`. First-order forward and reverse results are regression-tested
against finite differences on CPU, CUDA, and Vulkan.

Arbitrary higher-order AD is not part of the current contract. Nested or
concurrent Tape/FwdMode contexts, `kernel.grad()` inside Tape, and
forward-on-reverse (`kernel.grad()` inside FwdMode) raise `TaichiRuntimeError`
before compiling or submitting the unsupported operation. A Tape whose body
raises performs cleanup but does not run adjoints from a partial primal trace.
Dynamic `return` inside a non-static `if` or loop remains a frontend error;
compile-time `ti.static` specialization does not imply a general higher-order
control-flow guarantee. Explicit gradient Graphs remain manual, first-order
operations and must run outside automatic AD contexts.

### Integer values in <code>ti.types.rw_texture</code>

On Vulkan, storage-image load/store follows the shader-visible sampled type of
the declared format:

| Format family | <code>load()</code> element type | Required <code>store()</code> element type |
| --- | --- | --- |
| r16u, rg16u, rgba16u, r32u, rg32u, rgba32u | <code>ti.u32</code> | <code>ti.u32</code> |
| r16i, rg16i, rgba16i, r32i, rg32i, rgba32i | <code>ti.i32</code> | <code>ti.i32</code> |
| supported normalized and floating-point formats | <code>ti.f32</code> | <code>ti.f32</code> |

The 32-bit value type is the shader ABI; it does not change the physical
channel width. Values written to a 16-bit image must remain representable by
that image format. Three-channel RGB storage images are not part of this
contract.

### Argument and quantized-type boundaries

- Ndarray tensor elements are scalar, rank-1 vector, or rank-2 matrix values.
  Arbitrary rank tensor elements are rejected by ordinary kernel annotations,
  Graph, and low-level Graph Args before backend compilation. StructNdarray is
  supported by ordinary kernels but not by the current serialized Graph schema.
- Quantized integer and fixed-point widths are in `[1, 32]`. Quantized float
  exponent width is in `[1, 8]`; its significand field is at most 24 bits when
  signed and 23 bits when unsigned, and its compute type must be `ti.f32`.
  A `ti.f64` quant-float/shared-exponent contract is not supported.
- External NumPy/Torch arrays must be contiguous. C-contiguous values are used
  directly; Fortran-contiguous NumPy arrays use an explicit copy/copy-back
  adapter. Arbitrary-stride views are rejected instead of reaching a backend
  `TI_NOT_IMPLEMENTED` path.
- Graph sampled textures validate dimension, and Graph RWTextures validate both
  dimension and format against the kernel annotation before compilation.

### `ti.compile_profile(clear_on_enter=True)`

Location: `taichi_forge.tools.compile_profile`; exported as
`ti.compile_profile`. The returned type is also exported as
`ti.CompileProfile`.

Context manager for compile-time profiling.

```python
with ti.compile_profile() as prof:
    ti.compile_kernels([(step_kernel, (x, y))])

prof.dump_csv("compile.csv")
prof.dump_chrome_trace("compile.json")
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `clear_on_enter` | Clear existing compile timing records when entering the context. |

Useful methods:

| Method | Meaning |
| --- | --- |
| `dump_csv(path)` | Write C++ compile timing records as CSV. |
| `dump_chrome_trace(path)` | Write Chrome trace JSON. |
| `python_events()` | Return recorded Python-side compile events. |
| `dump_python_csv(path)` | Write Python-side compile events as CSV. |
| `records(include_cpp=True, include_python=True)` | Return merged timing records. |
| `top_n(n=10, include_python=True)` | Return the largest timing records. |

Limits:

- This is a development and diagnosis API. It is not meant to be left in a
  hot loop.
- Availability of C++ pass-level timing depends on the active runtime build.
- While profiling is active, SNode tree add/destroy records include coarse
  lifecycle-lock, backend materialization/synchronization/resource-release,
  executable-retirement, and kernel-definition-retirement scopes. They are
  diagnostic phase boundaries, not stable per-pass ABI names.

## `taichi_forge.runtime`

Runtime observability is available as `ti.runtime`. Startup attribution is
available before `ti.init()`; Program statistics and capabilities require an
active Program. These APIs do not change kernel, Graph, or submission
semantics.

### `ti.runtime.configure_startup_profile()` and `startup_profile()`

Startup profiling is disabled by default. Set `TI_STARTUP_PROFILE=1` before
import to include Python import, split-runtime discovery/load, pybind import,
and runtime-bitcode configuration. Alternatively call
`ti.runtime.configure_startup_profile(True, clear=True)` or pass
`ti.init(startup_profile=True)` to attribute subsequent init phases. The
immutable schema-v1 snapshot contains ordered timestamp events and matched
phase durations for reset, configuration, backend selection, Program creation,
runtime materialization, and primitive-capability registration.

The disabled path does not read a clock. The profiler is a low-frequency
diagnostic, not a steady-state kernel/Graph tracer; enabling it after package
import cannot reconstruct import or native-loader time. `startup_profile()`
does not require a CUDA driver or active Program.

### `ti.runtime.stats()`

Returns an immutable `RuntimeStatistics` snapshot:

```python
snapshot = ti.runtime.stats()
print(snapshot.backend, snapshot.program_domain)
print(snapshot.submission.kernel_submissions)
print(snapshot.memory.device_raw_bytes)
```

Statistics schema v2 groups submission, synchronization, memory, transfer,
Graph, display, first-fault, and trace counters. Counters are cumulative for
one Program generation and a snapshot remains valid after a later
`ti.reset()`. A new Program has a different `program_domain` and fresh
Program-owned counters.

An optional measurement is `None` when the active backend or build cannot
observe it. Zero means that the measurement is available and no activity was
observed. In particular, do not convert unavailable device-memory or backend
wait data into a measured zero. Taking a snapshot does not intentionally wait
for GPU completion.

`snapshot.memory.host_allocator` is a process-owned host-pool snapshot:

| Fields | Meaning |
| --- | --- |
| `requested_live_bytes`, `peak_requested_live_bytes` | Requested bytes not yet released through the pool, and its lifetime peak. This is not RSS. |
| `reserved_bytes`, `committed_bytes` | Current OS mapping size; committed bytes equal reserved bytes for Windows reserve+commit. Linux reports `None` because anonymous-mapping residency needs an OS RSS/page query. |
| `capacity_bytes`, `used_bytes`, `available_bytes` | Allocator-owned capacity, bump-cursor consumption including alignment, and still-allocatable tails. |
| `alignment_waste_bytes`, `unreclaimed_released_bytes`, `wasted_bytes` | Alignment loss, released slab bytes that the current policy cannot reuse, and their sum. |
| `*_chunk_count` | Current total, adaptive slab, request-larger-than-the-next-slab, and exclusive mapping counts. |
| `peak_reserved_bytes`, `peak_used_bytes`, `peak_wasted_bytes`, `peak_chunk_count` | Host-pool lifetime peaks; they intentionally survive Program reset. |

The older flat `host_requested_live_bytes`, `host_raw_bytes`, and
`host_capacity_bytes` fields remain compatibility aliases. New measurement
code should prefer `host_allocator`. `ti.tools.memory_pool_stats()` exposes
the same host values in its legacy dictionary and remains a diagnostic
snapshot, not a reset or allocator-control API.

The default host policy starts with a 16 MiB slab and grows geometrically up
to the existing 1 GiB ceiling. A request larger than the next slab receives a
request-sized, alignment-safe large mapping without advancing that sequence.
For release
diagnosis only, setting `TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS=0` before importing
or initializing Taichi restores the legacy fixed-1-GiB slab policy. This
environment switch is an internal rollback gate, not a stable `ti.init`
option or a long-term allocator-control API.

#### Memory growth and ownership boundaries

The current runtime applies these bounded contracts to long-lived state that
it owns:

- An OS mapping is released after every valid request in a non-exclusive host
  allocator chunk has been released. A partially live chunk is not unmapped at
  the cost of invalidating pointers.
- Python kernel specialization defaults to 1,024 resident entries per Program
  generation. Set a positive budget with
  `ti.init(kernel_specialization_limit=...)`. At the limit, compiled paths keep
  working and only new specializations are rejected. `ti.reset()` establishes a
  new Program generation and budget. Destroying an SNodeTree returns an
  unpinned generation binding to this budget after its runtime and backend
  owners retire. On CPU/CUDA/Vulkan, a direct root-dense Field template of the
  same Python kernel can reuse an executable across serial tree generations
  when layout, dtype, runtime-local places, compile policy, and native
  dependencies all match; the new generation creates only a validated
  binding. Pointer/bitmasked/dynamic/hash SNodes, hidden field captures,
  grad/dual, data-oriented templates, and layout mismatches remain fail-closed
  and generation-compiled. A stale Graph lease pins its old executable until
  release but is never retargeted.
- Temporary-source LRUs, compile/timeline traces, and raw kernel-profiler
  history have fixed capacities. Capacity exhaustion evicts, counts drops, or
  reports a clear error instead of growing without bound. Long-running profiler
  sessions should call `ti.profiler.clear_kernel_profiler_info()` periodically.
- Execution state for a destroyed SNodeTree is reclaimed at a safe
  synchronization boundary, the Python runtime-object registry does not hold
  dead wrappers strongly, and the weekly version-check thread starts at most
  once per Python process.

SNodeTree lifecycle churn and executable-producing kernel churn are separate
contracts. Repeated create/destroy cycles with a bounded peak live topology
reuse tree slots and keep tree metadata bounded. The root-dense intersection
above also makes compile cardinality follow unique kernel/layout identities,
not historical generation count; unsupported intersections still repeat
frontend/backend compilation. Short unit tests cover the bounded tree
invariant, while longer qualification runs exercise 10,000 tree-only and
executable-producing generations. Lifecycle telemetry distinguishes historical
materializations, resident specializations, template hits, binding creation,
reclaims, retired/pinned handles, and budget rejections.

Ordinary `ti.init()`, kernel, Graph, and UI runtime paths use in-process worker
threads and do not launch persistent helper subprocesses. The `ti` CLI,
diagnostic tools, source builder, and application-created child processes are
explicit caller-visible operations; applications still own the lifetime of
their multiprocessing workers.

Bounded does not mean constant RSS. Live fields, ndarrays, and Graphs, chunks
with live allocations, driver context/pool high-water marks, and the selected
number of specializations still consume memory. The on-disk offline cache is
not host RSS. Leak diagnosis should compare `requested_live_bytes`, current and
peak chunk statistics, OS RSS, and application object lifetimes instead of
using the process peak alone.

### `ti.runtime.capabilities()`

Returns immutable `RuntimeCapabilities` for the active Program. It describes
which observability mechanisms are implemented, including bounded tracing,
Chrome trace export, backend wait/lock telemetry, device-memory telemetry, and
CUDA mempool telemetry. It is not a hardware feature query and does not
predict whether a particular algorithm or kernel is supported.

### `ti.runtime.trace(path, *, max_threads=16, events_per_thread=4096)`

Creates a one-shot context manager that records bounded host-side runtime
events and exports Chrome/Perfetto-compatible JSON:

```python
with ti.runtime.trace("runtime.json") as trace:
    graph.run(arguments)
    render_kernel(frame)

print(trace.summary.recorded_events, trace.summary.dropped_events)
```

Tracing is disabled by default. An enabled session allocates a fixed
`max_threads * events_per_thread` event buffer; it never grows or blocks
when full. At most 64 thread shards and 1,048,576 total events are accepted.
Threads that cannot claim a shard and events that exceed capacity increment
`dropped_events`.

Current trace events cover host submission, Program synchronization, and bulk
transfer boundaries. They are not GPU timestamps and do not replace a CUDA,
Vulkan, or vendor GPU profiler. Trace-on overhead is observable, so use it for
bounded diagnostic windows rather than leaving it around a production hot
loop.

Only one runtime trace context may be active in a Python process; nested and
concurrent contexts are rejected. The context preserves a workload exception
while still attempting stop/export. If `ti.reset()` occurs inside the
context, the trace finishes and exports the Program generation active on
entry; work on a newly initialized Program is not mixed into that session.

The legacy `ti.compile_profile()` and C++ timeline facilities have separate
owners, schemas, and purposes. Compile profiling measures compilation;
`ti.runtime.trace()` measures bounded runtime host events. Private Program
methods whose names begin with `_runtime_` are implementation details, not
public APIs.

### CPU scheduler telemetry

`ti.profiler.set_cpu_scheduler_telemetry(True, reset=True)` opens an explicit
diagnostic window for the CPU ThreadPool. Query it with
`ti.profiler.query_cpu_scheduler_telemetry()` and close it by setting `False`.
The schema-v1 mapping reports submitted/completed/queued jobs, requested and
completed chunks, requested and joined workers, underfilled and nested-serial
jobs, exceptions, maximum queue/thread occupancy, and aggregate queue,
execution, and submitter-wait nanoseconds.

Telemetry is disabled by default. The disabled scheduler path performs one
relaxed flag load per ThreadPool invocation and does not read a clock or update
per-chunk counters. Enabled telemetry is intentionally observable and is for
short CPU investigations, not continuous production monitoring. A reset
returns the preceding counter window; resetting while jobs are active is a
diagnostic boundary rather than a transactional event partition. The API does
not alter thread admission or infer NUMA/CCD topology.

### `ti.real_func(fn)`

Location: `taichi_forge.lang.kernel_impl`; exported as `ti.real_func`.

Decorator for compiling a Taichi function as a real callable function instead
of always inlining it like `@ti.func`.

```python
@ti.real_func
def bsdf_eval(normal: ti.types.vector(3, ti.f32), wi: ti.types.vector(3, ti.f32)):
    return max(0.0, normal.dot(wi))
```

Parameters: a Python function using Taichi function syntax.

Limits:

- Intended for reducing compile pressure from large repeatedly used functions;
  it is not a general runtime acceleration switch.
- Current support is LLVM-oriented and non-autodiff.
- `ti.experimental.real_func` remains as a deprecated alias. Use
  `ti.real_func`.

## `taichi_forge.algorithms`

Import through:

```python
import taichi_forge as ti

ti.algorithms.experimental_reduce(...)
```

These APIs are Python-scope native primitives. They cannot be called inside
`@ti.kernel` or `@ti.func`. When the current backend and input layout support a
native path, they call CUDA device APIs, native Vulkan code/shaders, or native
CPU/C++ implementations directly. Otherwise, supported routes fall back to
Taichi helper kernels.

### CompileIQ boundary (0.6.3 in development)

The algorithms module exposes no independent CompileIQ search builders. The
historical reduce-provider and segmented-scan search implementations remain
private qualification tools; ordinary primitives, explicit `method=`, and
`method="auto"` keep their existing contracts. The public search entry point is
`ti.graph.compileiq_recipe_search(graph)`, and it accepts only complete
Forge-owned Graph recipes.

### Primitive capability queries

| API | Return | Contract |
| --- | --- | --- |
| `primitive_capability(name)` | `PrimitiveCapability` | Static immutable schema-v1 contract for one family; valid before `ti.init()`. |
| `primitive_capabilities()` | tuple of `PrimitiveCapability` | All current families in stable catalog order. |
| `resolve_primitive_capability(name)` | `ResolvedPrimitiveCapability` | Current Program/backend provider resolution; requires `ti.init()`. |

`PrimitiveCapability` exposes `schema_version`, `name`, `entry_points`,
aggregate `dtypes/ranks/layouts/storages`, role-specific `operands`,
`methods`, `stability`, `determinism`, `atomic_order_dependent`, `ad`,
`graph_replay`, `aot`, `workspace`, and `fallback`. Each
`PrimitiveOperandCapability` exposes `name`, access mode, dtype/rank/layout/
storage tuples, and machine-readable constraints. Each
`PrimitiveMethodCapability` exposes its public method name, backend set,
provider probes, implementation kind, and whether final support depends on the
concrete input.

Each `ResolvedPrimitiveMethod` reports `program_available`. This is a
provider-level result, not concrete request validation. The operation still
checks dtype, shape, layout, storage, device features, and workspace capacity
before writing. The returned dataclasses are frozen snapshots.

Automatic-AD behavior is part of the descriptor. Tape keeps the established
complete-primal-plus-adjoint gate. FwdMode uses verified helper-kernel
fallbacks for transform, reduce-sum, gather, scatter, and scatter-add; scan and
grouped-reduce reject. Discrete/non-differentiable families reject automatic AD
before writing. See [Native algorithms](native_algorithms.en.md) for the full
matrix.

### Sort

#### `ti.algorithms.sort(keys, values=None, *, stable=True, descending=False, method="auto", precision="exact", workspace=None)`

Stable sort dispatcher for 1D key arrays, optionally carrying values.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `keys` | 1D ndarray, dense field, or supported member view containing sort keys. |
| `values` | Optional payload array with matching length. |
| `stable` | Require stable ordering. |
| `descending` | Sort in descending order when the selected method supports it. |
| `method` | `"auto"` or an explicit backend route such as legacy, CPU native, CUDA native, or Vulkan native where available. |
| `precision` | Sort precision policy. `"exact"` is the portable default. |
| `workspace` | Optional `SortWorkspace` for repeated calls. |

Limits:

- Method support depends on arch, dtype, and input layout.
- Some explicit native methods may reject unsupported dtypes, descending order,
  or non-contiguous inputs.
- The vanilla-compatible `parallel_sort()` entry point is still available.

#### `ti.algorithms.sort_by_key(key_parts, values=None, *, stable=True, order="lexicographic", method="auto", workspace=None)`

Sort by one or more key arrays.

Limits:

- `order="lexicographic"` is the current supported ordering.
- Key parts must be scalar 1D arrays with matching length.
- Whole StructNdarray payloads are not treated as sort keys; member views may
  be accepted by supported native paths.

### Prefix Sum

#### `ti.algorithms.PrefixSumExecutor(length).run(input_arr)`

Inclusive in-place prefix sum over `input_arr`.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `length` | Number of items handled by this executor. Fixed at construction. |
| `input_arr` | 1D numeric input/output array, dense field, or supported member view. |

Limits:

- Native scan paths support CPU, CUDA, and Vulkan where the runtime primitive
  is available.
- Native numeric inputs support common scalar integer and float types.
- Fallback field helper support is narrower.

### Device-resident valid-prefix composition

#### `ti.algorithms.device_prefix(values, extent, *, workspace=None)`

`DevicePrefix` pairs a fixed-capacity scalar 1D ndarray with a
`ti.DeviceExtent`. Its `compact()`, `scan()`, `reduce()`, `sort()`, `unique()`,
`run_length_encode()`, `grouped_reduce()`, and `bucket_builder()` methods pass
the device-written valid count from one operation to the next without reading
it in Python:

```python
workspace = ti.algorithms.DevicePrefixWorkspace(capacity)
source = ti.algorithms.device_prefix(values, source_extent, workspace=workspace)
active = source.compact(flags, compacted, compacted_extent)
active.sort()
active.scan(scanned)
```

The arrays retain their fixed physical capacity. Only entries below the paired
extent count are semantically defined; providers may overwrite the inactive
suffix with neutral values or sort sentinels. The wrapper reuses existing
CPU/CUDA/Vulkan primitive providers and workspace objects and does not claim
that every provider executes only the active count. Supported scalar dtypes
are `i32/u32/i64/u64/f32/f64`; individual operations keep the narrower dtype
and semantic constraints of the underlying primitive. A workspace is reusable
but must not be used concurrently. `clear()` releases its Python-owned staging
and child workspaces.

`ti.algorithms.DevicePrefixSequence(capacity)` records the same operations as
one logical fixed-topology native Graph node. Declare symbolic inputs with
`sequence.input(values_arg, extent_arg)`, chain the returned prefix methods,
then append the sequence with `builder.append_native(sequence)`. Counts remain
on device across every recorded operation; runtime arguments still use the
normal Graph ndarray names. A sequence becomes immutable after it is appended
and compiled, and its workspace must not be shared by concurrent sequences.
Provider selection, workspace topology, and operation routing are fixed when
the native node is materialized, so steady-state replay does not repeat the
operation-kind/provider decision. `memory_report()` exposes current/peak
workspace bytes and confirms zero replay allocations. The current core Prefix
providers do not expose a complete enclosing-backend command recipe.
`append_native(..., admission="auto")` therefore rejects the fragmented plan;
the default explicit admission keeps a diagnostic segmented route whose
manifest and `physical_plan()` report loose helpers and queue topology. This
contract does not imply backend Graph recording or cross-operation kernel
fusion.

### Device worklists

#### `ti.algorithms.DeviceWorklist(capacity, dtype=ti.i32, *, workspace=None, telemetry=True, transition_mode="staged", unique_key_capacity=None)`

Owns fixed-capacity front/back scalar storage, paired `DeviceExtent` state,
mandatory publication state, optional counters, and reusable primitive
workspace. Supported value dtypes are
`i32/u32/i64/u64/f32/f64`; capacity is a positive Python integer no greater
than `2^31-1`. The object is tied to the active runtime generation and rejects
use after `ti.reset()`.

The direct atomic-producer lifecycle is:

1. `prepare_next()` resets back extent and counters on device.
2. A kernel calls `device_worklist_append()` using
   `*worklist.append_arguments()`.
3. `commit_next(dispatch_state=None)` finalizes count/counters and swaps front
   and back without synchronizing.

`telemetry=False` does not allocate, bind, or write accepted/rejected/conflict/
winner counters. Staged mode retains generated/overflow/generation in 12 bytes.
`transition_mode="direct"` requires `telemetry=False`, retains only overflow
and generation (8 bytes), and uses `device_worklist_append_direct()` so the
producer publishes its extent during reservation. It supports only atomic
append transitions; `finalize_next()` is invalid in this mode.

`unique_key_capacity` is an explicit direct-mode contract for a bounded dense
integer key domain. It allocates one persistent i32 generation tag per key and
one scalar epoch. A producer calls
`device_worklist_append_unique_direct(*worklist.unique_append_arguments(), key,
value)` to append each emitted key at most once in the current transition.
Tags are not cleared between transitions. Invalid keys, an ABI mismatch,
capacity overflow, source overflow, and epoch exhaustion fail closed through
sticky overflow. Output order remains unspecified.

`graph_args()` preserves that ABI. `transition_arguments(step)` returns the
alternating source/target pair for a static multi-stage Graph, and
`DeviceWorklistSequence(args).prepare(target=...)` advances the epoch before
each bounded producer. `commit_direct_transitions(steps)` changes Python front
ownership only after completion. This enables incremental frontier and halo
generation but does not itself remove an application full-domain copy/select;
the caller must update retired/default state as part of its own active-domain
kernel. A worklist has one completion-ordered lane and cannot be written by
concurrent submissions.

On Vulkan, an adjacent recordable `prepare()` and bounded consumer are lowered
into one backend Graph region. The prepare kernel resets the target and
publishes the source extent's indirect packet in the same dispatch, so the
steady-state pair contains two physical dispatches and no loose/helper launch.

A direct worklist can eliminate a later standalone `prepare_next()` when an
already globally ordered boundary kernel calls
`device_worklist_recycle_direct(*worklist.recycle_arguments())` exactly once
after the preceding producer has finished and after that boundary has consumed
the old front. Call `commit_recycled_next()` after the boundary dispatch to
swap the host-side front identity. This is a low-level ordering contract: it
does not insert a device-wide barrier inside a producer, cannot be mixed with
staged transitions, and resets the lean shared overflow statistic while the
published `next_extent` retains that transition's overflow state. Explicit
`statistics()` merges the current published extent, so the latest overflow
remains observable without adding work to submission.

For a unique direct worklist, use
`device_worklist_recycle_unique_direct(*worklist.unique_recycle_arguments())`.
It advances both generation and the dense-tag epoch and fails closed on either
counter's exhaustion. The ordinary recycle argument accessor rejects a unique
worklist.

`fixed_graph_args(builder, name)` is the provider-owned counterpart to
`graph_args(name)`. It privately binds address-stable worklist resources into
that builder and removes them from the public runtime argument dictionary.
The compiled Graph is generation/front-parity qualified, supports only one
workspace lane, and runtime-orders asynchronous reuse, adding a completion
fence when the backend exposes pending work. After an odd number
of committed transitions, use the separately compiled Graph for the other
front parity. This form expresses ownership; it does not skip native resource
retention or promise a speedup.

`device_worklist_append(values, extent_state, generated, overflow, capacity,
value)` is a `@ti.func` and returns the reserved slot or `-1` on overflow. The
overflow-free path performs one reservation atomic per item. The append order
is unspecified, so consumers that require order must call stable `select()` or
deterministic `resolve_conflicts()`. One producer owns each transition; the
worklist does not serialize independent Graph submissions that write it. The
capacity scalar must equal the physical values capacity; a mismatched Graph
binding fails closed with sticky overflow before any value write.

`select(flags, *, method="auto", dispatch_state=None)` stably filters the
current front into the back and commits it. `resolve_conflicts(keys, *,
priorities=None, ordinals=None, policy="first", method="auto",
strategy="auto", sort_method=None, key_capacity=None,
dispatch_state=None, output_shape="compact_winner_list")` accepts integer keys
and chooses one winner per key.
Policies are `first`, `claim`, `min_priority`, and `max_priority`; priority
policies require an i32 priority ndarray. Ties are resolved by ordinal, then
source index. The returned `DeviceConflictResult` exposes device-owned keys,
values, priorities, ordinals, extent, counter arrays, selected `strategy`, sort
provider, and key capacity.

Conflict algorithm selection and sort-provider selection are independent.
`strategy="radix_grouped"` uses the CPU/CUDA/Vulkan native stable-sort
provider selected by `sort_method`; the legacy `method` argument remains its
backward-compatible alias. `strategy="dense_atomic"` requires a bounded
integer `key_capacity` and uses deterministic multi-pass arbitration followed
by a key-domain scan. Out-of-domain keys are rejected and set overflow rather
than accessing scratch out of bounds. `strategy="auto"` admits the dense path
only for a sufficiently compact declared key domain and uses a conservative
small-CPU crossover; otherwise it selects radix. The dense path currently owns
extent publication and therefore rejects `dispatch_state`. Neither strategy
silently falls back to a host round trip. Radix winner reduction assigns one
scan to each sorted key run; a few exceptionally long runs reduce parallelism
and should be treated as a separate workload shape when benchmarking.

`output_shape="dense_winner_table"` requires explicit `dense_atomic`, a
bounded `key_capacity`, and `telemetry=False`. It returns only a dense
source-index table with `0x7fffffff` for empty keys. The worklist front is not
replaced, and no compact extent, scan, compact, or winner-list materialization
is created.

`resolve_conflicts_from_mask(keys, active_flags, *, priorities=None,
ordinals=None, policy="first", key_capacity=...)` is the fixed-domain form of
that contract. `keys`, `active_flags`, and any attributes have worklist
capacity; inactive slots are ignored and winner values are original source
indices. It skips candidate compaction and attribute gathering. Omitting
`ordinals` uses source index as the deterministic final tie break and removes
the ordinal arbitration pass and buffer. This explicit low-level path requires
`telemetry=False`, returns only `dense_winner_sources`, reports an active
out-of-domain key through overflow, and does not imply that a generic dense
claim is faster than a fused workload-specific atomic kernel.
For `min_priority`/`max_priority`/`claim` without custom ordinals, CPU and CUDA
use a packed 64-bit priority/source arbitration route; signed priority order
and source-index tie breaking are preserved exactly. Vulkan uses the portable
32-bit multi-pass route. `DeviceConflictResult.arbitration_route` reports
`packed_priority_source_u64` or `portable_multi_pass`. The packed route removes
one arbitration dispatch and adds four scratch bytes per key; route selection
does not change the result contract.

#### `ti.algorithms.DenseConflictClaimTable(key_capacity, source_capacity, *, priority_range=None, policy="min_priority")`

Owns a packed dense claim table for callers that already have reset, producer,
and materialization kernels and therefore cannot benefit from a self-contained
multi-pass resolver. `policy` is `min_priority` or `max_priority`; source index
is the deterministic final tie break. A declared signed-i32 `priority_range`
selects a portable u32 encoding when the complete priority/source domain fits.
That route is supported on CPU, CUDA, and Vulkan. Wider or full-i32 domains use
u64 on CPU/CUDA; Vulkan rejects them explicitly and callers may retain
`resolve_conflicts_from_mask()` as the portable fallback.

The low-level functions are fusion points, not hidden dispatches:

- call `device_dense_conflict_reset_slot(claims, key)` for every key in an
  already ordered dense reset kernel;
- call `device_dense_conflict_begin(overflow, generation)` exactly once in the
  same reset kernel;
- call `device_dense_conflict_claim(*table.claim_arguments, key, priority,
  source)` directly in the candidate producer;
- call `device_dense_conflict_winner_source(*table.winner_arguments, key)` in
  the consumer/materializer; it returns `-1` for an empty key.

There is no device-wide barrier between reset and claim inside one kernel, so
they must remain globally ordered dispatches; each may be fused with existing
work on its side of that boundary. Invalid keys, sources, priorities, or a
forged capacity set sticky overflow without accessing the table. `reset()` is
the standalone convenience path. `statistics()` is an explicit synchronized
overflow/generation snapshot. `graph_args(name)` and `runtime_arguments(name)`
provide a stable symbolic Graph ABI with no replay allocation. The table owns
one packed value per key plus two i32 scalars; no decoded i32 winner table is
materialized.

`statistics()` and `snapshot()` are explicit synchronized observations.
`execution_report(dispatch=None, target="current")` additionally joins the
latest counters with a bounded-dispatch snapshot and reports useful, executed,
skipped, encoded, overflow, and exact-grid state. `memory_report()` reports
owned front/back, extent, counter, and reusable workspace bytes.

#### `ti.algorithms.DeviceWorklistSequence(args, *, workspace=None)`

Records exactly one transition over the symbolic bundle returned by
`worklist.graph_args(name)`: `prepare_next()`, `finalize_next()`, `select()`,
`resolve_conflicts()`, or `resolve_conflict_winner_table()`. Direct mode records
`prepare_next()` but intentionally rejects `finalize_next()`. Append the result with
`GraphBuilder.append_native()`. Staging is allocated before submission and the
compiled action neither allocates nor reads the count on host during
steady-state replay. The first execution may still compile kernels and prepare
native provider workspace.
The sequence fixes its conflict strategy, sort provider, and workspace at
materialization. It is immutable after compilation and its workspace is
serially reusable, not concurrently shareable.

Bind runtime values with `worklist.runtime_arguments(name)`. Atomic-producer
graphs also pass `include_capacity=True` because `append_arguments()` includes
a scalar capacity argument. With full telemetry,
`DeviceWorklistGraphArgs.observe(builder, name=...)` attaches counters to
terminal Graph observation; lean arguments reject observation rather than
materializing hidden state.
`decode_observation(mapping)` returns `DeviceWorklistStatistics` after ticket
completion. On Vulkan, an adjacent recorded `finalize_next()` and matching
bounded consumer automatically share a Graph-owned packet and remove the
consumer preparation dispatch. A matching producer-owned
`DeviceDispatchState` on both sides remains supported for explicit
finalize/select/claim packet publication. On CUDA, `launch_state` is only a
compatibility adapter for capacity, extent identity, and block geometry; the
exact Driver API 12.4 route reads the extent through Graph-owned controls and
does not consume the external packet. Inspect
`dynamic_work_capabilities()` for the selected physical route.

### Primitive Algorithms

#### `ti.algorithms.DeterministicScatterReducePlan(indices, num_groups)`

Builds an explicit reusable reduction topology from host-visible integer
destination indices. Valid sources are stably grouped by destination while
preserving source ordinal within each group; negative and out-of-range indices
are ignored. `plan.bind(values, output)` accepts matching one-dimensional
scalar or vector ndarrays/root-dense fields with component dtype
`i32/u32/i64/u64/f32/f64`, and reduces each group left-to-right in one indexed
dispatch on CPU, CUDA, and Vulkan. `binding.run()` executes it and
`binding.graph_action()` returns a native Graph action.

This is an opt-in reproducibility and qualification route. It never replaces
atomic scatter-add automatically and is not differentiable. Independent
concurrent execution requires independent bindings. The order is repeatable
for the same backend/build contract; the API does not promise identical bits
across backends/compiler modes or improved numerical accuracy. `report()`
exposes stable topology bytes, component shape, implementation route, and zero
ordered-value/workspace bytes. Atomic assembly remains the default performance
route; use this stable-serial plan only when reproducibility is required.

These functions return a workspace when a workspace object is useful for replay
or reuse. Pass an explicit workspace to keep scratch buffers and native plans
alive across frames.

| API | Purpose |
| --- | --- |
| `experimental_compact(values, flags, output, count, *, method="auto", workspace=None)` | Stable compact. Writes items whose `flags[i] != 0` into `output`, and writes the count to a device scalar. |
| `experimental_run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | Encode consecutive equal integer-key runs into unique keys and i32 lengths. |
| `experimental_unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Select the first value from each consecutive equal-value run. |
| `experimental_unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Select one key and the first payload from each consecutive key run. |
| `experimental_segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | Sum each segment described by an immutable `SegmentedLayout`. |
| `experimental_segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | Inclusive or exclusive sum within each segment. |
| `experimental_reduce(values, output, *, op="sum", method="auto", workspace=None)` | Reduce 1D `values` into scalar `output[0]`. `op` supports `"sum"`, `"min"`, and `"max"` where the selected backend supports them. |
| `experimental_histogram(values, bins, *, method="auto", workspace=None)` | Histogram integer values into fixed bins. |
| `experimental_transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | Compute `dst = src * scale + bias`. |
| `experimental_gather(src, indices, dst, *, method="auto", workspace=None)` | Indexed read: `dst[i] = src[indices[i]]`. |
| `experimental_scatter(src, indices, dst, *, method="auto", workspace=None)` | Indexed write: `dst[indices[i]] = src[i]`. |
| `experimental_scatter_add(src, indices, dst, *, method="auto", workspace=None)` | Indexed add: `dst[indices[i]] += src[i]`. |
| `experimental_bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | Build grouped output for integer bucket keys. |
| `experimental_grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | Reduce values by integer group key. |

Common limits:

- Inputs are expected to be dense, contiguous, and shape-compatible on native
  paths. Sparse and complex SNode trees are not treated as native-compatible.
- StructNdarray support is member-view based; whole tensor/member semantics are
  intentionally narrower than ndarray scalar paths.
- `experimental_scatter_add()` with duplicate floating-point targets can differ
  across backends because atomics do not guarantee the same accumulation order.

#### Consecutive RLE and Unique

RLE/Unique has deliberately explicit consecutive semantics. It never sorts or
hashes implicitly. Arbitrary input preserves run order; sorted input therefore
produces global sorted unique keys. `unique_by_key` selects the first payload
in each run. StructNdarray raw payloads are accepted by unique-by-key. Dense
MatrixField payloads currently require matching input/output shapes and
`ti.i32` elements.

The current RLE/Unique contract supports integer keys (`i32/u32/i64/u64`).
`size` is an optional Python integer selecting the active prefix
`[0, size)` of fixed-capacity storage. It defaults to the full input capacity;
`size=0` represents a logical empty input even though Taichi dense arrays
cannot have physical shape zero. Output capacity must still be at least the
physical input capacity.

`size` changes logical results but not physical scratch capacity or compact
dispatch extent: the boundary kernel clears the inactive tail and the compact
provider still processes fixed-capacity arrays. This avoids Graph rebuilds and
reallocations. If active sizes are usually much smaller than capacity, bucket
workloads by capacity or use smaller storage/workspaces.

`run_count`/`count` remains on device (one-element i32 ndarray or scalar i32
field). Only output entries below that count are defined; reading the count in
Python synchronizes that scalar. Input/output aliasing is rejected before
submission. These discrete operations reject Tape/FwdMode before writing.

`method="auto"` composes a boundary kernel with the existing CPU native,
CUDA driver-only, or Vulkan native compact provider; dense-field fallback uses
`field_scan`. `RunLengthWorkspace(max_items=None)` owns reusable flags and,
for RLE, start buffers. The minimum scratch is 4 bytes/item for Unique and
12 bytes/item for RLE, plus the selected compact provider's temporary storage.
One workspace is not safe for concurrent calls; give each concurrent producer
or Graph its own workspace.

#### Segmented Reduce and Scan

`SegmentedLayout` validates reusable topology on the host and normalizes it to
device-resident i32 offsets and segment IDs:

```python
layout = ti.algorithms.SegmentedLayout.from_offsets(
    np.array([0, 4, 4, 11], np.int32),
    capacity=16,
)
workspace = ti.algorithms.SegmentedWorkspace(
    max_items=16, max_segments=3
)
ti.algorithms.experimental_segmented_scan(
    values, layout, scanned, workspace=workspace
)
```

`from_offsets()` requires at least `[0, end]`, starts at zero, accepts
nondecreasing offsets and repeated offsets for empty segments, and treats the
last offset as `num_items`. `from_segment_ids(ids, num_segments, size=None,
capacity=None)` accepts a nondecreasing active prefix of IDs in
`[0, num_segments)`; missing IDs represent empty segments and the inactive
fixed-capacity tail is normalized to `-1`. Construction from a Taichi
array/field reads topology to the host and therefore synchronizes. Construction
is intended to be outside the hot loop; reuse does not read topology back.

The public properties are `encoding`, `num_items`, `capacity`,
`num_segments`, `max_segment_length`, and `topology_bytes`. Both operations
require matching scalar 1D plain ndarrays or root-dense fields with
`i32/u32/i64/u64/f32/f64` elements. Reduce output has exactly one element per
segment and must be disjoint from input. Scan output has exactly `capacity`
elements and may be exactly in-place or disjoint. Only the scan prefix below
`num_items` is defined; the padded tail is capacity storage, not another
segment. Matrix fields, StructNdarray views/raw payloads, and sparse SNodes are
not supported by the current contract. Only `op="sum"` is implemented.

Reduce `method="auto"` uses the grouped ndarray provider where possible and
the stable segment-local `serial` method for dense fields. Integer results are
exact; grouped floating sum can vary with backend atomic order, while explicit
`serial` follows stable left-to-right segment order. Reverse AD is supported
only by the grouped ndarray sum path; FwdMode and serial/dense-field AD reject
before writing.

Scan accepts `auto`, `serial`, and integer-only `global_scan`. Float scan
always uses stable left-to-right accumulation. Integer auto uses the
zero-scratch serial path on CPU/Vulkan and on ordinary short CUDA segments; it
uses global scan only for a CUDA layout with at least 65,536 active items and
a segment of at least 4,096 items. The coarse choice is observable through
`SegmentedWorkspace.last_scan_method`, and explicit `method=` remains
available for controlled tuning. Scan rejects automatic AD before writing.

`SegmentedWorkspace` reuses child scan/reduce plans and scratch.
`workspace_bytes_current` / `workspace_bytes_peak` exclude immutable
`layout.topology_bytes`. A short serial scan needs no workspace allocation;
global scan may allocate provider scratch and one value per segment. A
workspace is not concurrently shareable. Share the immutable layout but use
one workspace per producer or Graph.

### Device-Side Numeric Checks

These APIs launch device-side checks from Python scope and return a result
object whose read method synchronizes one scalar.

| API | Return | Purpose |
| --- | --- | --- |
| `count_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count nonzero predicate values. |
| `any_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check whether any predicate is true. |
| `all_if(flags, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check whether all predicates are true. |
| `nan_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count NaN values. |
| `inf_count(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Count infinite values. |
| `all_finite(values, *, method="auto", workspace=None)` | `DeviceCheckResult` | Check that all values are finite. |
| `index_bounds_check(indices, upper, *, lower=0, method="auto", workspace=None)` | `DeviceCheckResult` | Count indices outside `[lower, upper)`. |
| `max_abs(values, *, method="auto", workspace=None)` | `DeviceMetricResult` | Compute max absolute value. |
| `max_abs_delta(values, reference, *, method="auto", workspace=None)` | `DeviceMetricResult` | Compute max absolute difference. |

Result objects:

| Type | Methods / fields |
| --- | --- |
| `DeviceCheckResult` | `device_scalar`, `kind`, `to_int()`, `to_bool()`, `ok()` |
| `DeviceMetricResult` | `device_scalar`, `kind`, `to_float()` |

Limits:

- These calls are Python-scope native methods, not kernel-scope DSL functions.
- `to_int()`, `to_bool()`, `ok()`, and `to_float()` read one scalar back to the
  host and therefore synchronize that scalar.
- Native routes cover dense ndarray, dense field, and supported StructNdarray
  member views. Non-dense and sparse SNode trees are not native check targets.
- Vulkan metric fast paths currently prioritize `f32`; unsupported `f64`
  metric routes fall back or reject depending on the selected method.

The same check functions are also available under `ti.algorithms.check`.

### Workspaces

Reusable workspace classes:

```python
workspace = ti.algorithms.ReduceWorkspace(max_items=n)
ti.algorithms.experimental_reduce(values, out, workspace=workspace)
```

| Workspace | Used by |
| --- | --- |
| `SortWorkspace(max_items=None, device=None)` | `sort()`, `sort_by_key()` |
| `CompactWorkspace(max_items=None)` | `experimental_compact()` |
| `RunLengthWorkspace(max_items=None)` | `experimental_run_length_encode()`, `experimental_unique()`, `experimental_unique_by_key()` |
| `SegmentedWorkspace(max_items=None, max_segments=None)` | `experimental_segmented_reduce()`, `experimental_segmented_scan()` |
| `ReduceWorkspace(max_items=None, cache_native_plans=True)` | `experimental_reduce()` |
| `HistogramWorkspace(max_items=None, max_bins=None)` | `experimental_histogram()` |
| `TransformWorkspace(max_items=None, cache_native_plans=True)` | `experimental_transform()` |
| `IndexedCopyWorkspace(max_items=None, cache_native_plans=True)` | `experimental_gather()`, `experimental_scatter()` |
| `ScatterAddWorkspace(max_items=None, max_groups=None)` | `experimental_scatter_add()` |
| `BucketBuilderWorkspace(max_items=None, max_bins=None)` | `experimental_bucket_builder()` |
| `GroupedReduceWorkspace(max_items=None, max_groups=None)` | `experimental_grouped_reduce()` |
| `CheckWorkspace(max_items=None)` | device-side checks returning `DeviceCheckResult` |
| `MetricWorkspace(max_items=None)` | device-side metrics returning `DeviceMetricResult` |

Common fields and methods:

- `workspace_bytes_current`
- `workspace_bytes_peak`
- `clear()`

### Primitive Sequences

#### `ti.algorithms.primitive_sequence()`

Creates a replayable sequence of Forge-defined native primitives.

```python
seq = ti.algorithms.primitive_sequence()
err = seq.max_abs_delta(values, reference)
seq.prewarm()
seq.run()
print(err.to_float())
```

Common methods:

| Method | Purpose |
| --- | --- |
| `prewarm(repeat=1)` | Build and warm native plans without treating the run as a measured hot replay. |
| `run(repeat=1)` | Replay the recorded native sequence. |
| `scan(input_arr, *, executor=None)` | Add an in-place prefix-sum primitive. |
| `count_if(...)`, `any_if(...)`, `all_if(...)`, `nan_count(...)`, `inf_count(...)`, `all_finite(...)`, `index_bounds_check(...)` | Add device check primitives. |
| `max_abs(...)`, `max_abs_delta(...)` | Add metric primitives. |
| `sort(...)`, `sort_by_key(...)` | Add sort primitives where supported. |
| `reduce(values, output, *, op="sum", method="auto", workspace=None)` | Add a reduce primitive. |
| `histogram(values, bins, *, method="auto", workspace=None)` | Add a histogram primitive. |
| `transform(src, dst, *, scale=1, bias=0, method="auto", workspace=None)` | Add an affine transform/copy primitive. |
| `gather(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed read primitive. |
| `scatter(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed write primitive. |
| `scatter_add(src, indices, dst, *, method="auto", workspace=None)` | Add an indexed add primitive. |
| `compact(values, flags, output, count, *, method="auto", workspace=None)` | Add a compact primitive. |
| `run_length_encode(keys, unique_keys, run_lengths, run_count, *, size=None, method="auto", workspace=None)` | Add a consecutive RLE primitive. |
| `unique(values, output, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Add a consecutive unique primitive. |
| `unique_by_key(keys, values, unique_keys, unique_values, count, *, mode="consecutive", size=None, method="auto", workspace=None)` | Add consecutive unique-by-key with first-payload semantics. |
| `segmented_reduce(values, layout, output, *, op="sum", method="auto", workspace=None)` | Add a fixed-topology segmented sum. |
| `segmented_scan(values, layout, output, *, inclusive=True, op="sum", method="auto", workspace=None)` | Add an inclusive/exclusive segmented sum scan. |
| `bucket_builder(keys, values, offsets, output, *, method="auto", workspace=None)` | Add a bucket-builder primitive. |
| `grouped_reduce(keys, values, output, *, op="sum", method="auto", workspace=None)` | Add a grouped-reduce primitive. |
| `clear()` | Clear owned workspaces and captured native plans. |

Useful properties include `call_count`, `direct_plan_count`,
`fused_plan_count`, `fused_plan_method`, `workspace_bytes_peak`, and
`workspaces`.

Limits:

- Primitive sequences are for Forge-defined native primitives only.
- They are not arbitrary user native callbacks.
- Sequence contents must keep the underlying arrays/workspaces alive for replay.

### Diagnostics and Cache Helpers

| API | Purpose |
| --- | --- |
| `clear_default_workspaces()` | After submissions quiesce, clear default workspace caches isolated by Program and Python thread. |
| `legacy_helper_auto_fallback_enabled()` | Query whether legacy helper fallback is enabled. |
| `set_legacy_helper_auto_fallback_enabled(enabled)` | Enable or disable automatic legacy helper fallback. |
| `reset_legacy_helper_auto_fallback_policy()` | Restore the default fallback policy. |
| `legacy_helper_fallback_counting_enabled()` | Query fallback counting state. |
| `set_legacy_helper_fallback_counting_enabled(enabled, clear=False)` | Enable fallback counting and optionally clear existing counts. |
| `clear_legacy_helper_fallback_counts()` | Clear fallback counters. |
| `get_legacy_helper_fallback_counts(reset=False)` | Read fallback counters. |
| `clear_primitive_diagnostics()` | Clear primitive diagnostics. |
| `set_primitive_diagnostics_enabled(enabled, clear=False)` | Enable primitive diagnostics and optionally clear existing records. |
| `get_primitive_diagnostics(reset=False)` | Read primitive diagnostics. |
| `get_primitive_runtime_diagnostics(reset=False)` | Return a schema-v1 provider/dependency/fallback/counter/workspace snapshot without waiting for the device. |
| `get_primitive_workspace_statistics()` | Return schema-v1 Program provider bytes, aliases/errors, and logical per-thread default-cache statistics without waiting for the device. |

These helpers are intended for validation and deployment diagnostics. They are
not performance-critical hot-loop APIs. Provider invocation counts are complete
only for an interval where `set_primitive_diagnostics_enabled(True)` was active.
The `program_provider_bytes*` and `default_cache.logical_workspace_bytes_*`
fields returned by `get_primitive_workspace_statistics()` are different
ownership views that can refer to the same resource and must not be added.
`clear_default_workspaces()` is not supported concurrently with submissions
using the same workspace; stop producers and establish a quiescent boundary
first.

See also [Native algorithms](native_algorithms.en.md).

## AOT APIs

### `ti.aot.Module(arch=None, caps=None)`

Current Forge AOT compilation is same-target only. Omitting `arch` uses the
active runtime architecture; an explicit value must equal the architecture
selected by `ti.init()`. A mismatch raises `TaichiRuntimeError` before creating
the backend builder. Forge does not silently replace the requested target and
does not currently claim cross-architecture compilation.

CUDA AOT uses an explicit artifact target instead of treating the build GPU as
the distribution contract. The default target is compute capability 60 with
its derived PTX 50 floor. Select a newer exact target with, for example,
`caps=[ti.DeviceCapability.cuda_compute_capability(86)]`. The target must be at
least 60 and must be supported exactly by Forge's bundled LLVM NVPTX backend.
Forge records both values in `aot_metadata.json`; the LLVM AOT loader validates
the active device and its selected PTX target before registering any kernel.
Choosing a newer target can enable target-specific codegen but deliberately
narrows the set of loadable GPUs. This mechanism uses the CUDA Driver API and
does not add a CUDART or CUDA Toolkit runtime dependency. CUDA LLVM AOT
artifacts created before this metadata contract must be rebuilt; the loader
rejects a missing sidecar rather than guessing requirements from the build
machine.

GFX AOT artifacts now preserve dense SNodeTree layout identity explicitly.
metadata.json stores every artifact-local root-buffer size, each field's tree
id, and each kernel's sorted tree dependencies. The C API loader allocates all
serialized roots and passes the recorded tree count to kernel registration;
it no longer assumes one root. Artifact-local ids must be contiguous and do
not encode process-local tree generations. A runtime with destroyed-tree holes
is rejected at build time instead of producing an ambiguous artifact. The
legacy C++ get_root_size() view still returns the first root; multi-tree
loaders must use get_root_sizes(). This support remains limited to dense AOT
fields; sparse SNode AOT is outside this contract.

AOT kernel-template instantiation accepts ndarray exemplars on CPU, CUDA, and
Vulkan. Supported exemplars are ordinary scalar/vector/matrix Taichi ndarrays,
C-contiguous NumPy arrays, and contiguous Torch tensors. The specialization
key records element dtype/shape, logical ndim, AOS contiguous byte stride,
gradient presence, and boundary mode. Runtime capacity is deliberately
excluded, so the same ABI at different lengths reuses one artifact. SOA or
structured ndarray views, non-contiguous host arrays, textures, and arbitrary
Python objects fail before compilation. Keys use the filesystem-safe
__tmpl__ convention; signatures over 180 UTF-8 bytes use a deterministic
SHA-256 key to avoid Windows path-length failures.

## Kernel and Graph APIs

Dense Field-specific layouts, lifetime, concurrency, AD, and backend behavior
are documented in [Dense Field Graph](dense_field_graph.en.md).

### `GraphBuilder.dispatch(kernel, *args, template_args=None, label=None)`

`Sequential.dispatch()` provides the same keyword-only `template_args` and
`label` parameters. `template_args` binds a data-oriented `self`, a Field, or another
`ti.template()` parameter at graph definition/compile time. These objects do
not become `Graph.run()` runtime arguments.

```python
builder.dispatch(
    solver.step_kernel,
    slot_arg,
    template_args={"self": solver, "state": solver.state},
)
graph = builder.compile()
graph.run({"slot": 3})
```

Contract:

- Every `ti.template()` parameter must be provided by kernel argument name.
  Unknown, missing, or ordinary scalar/matrix names raise
  `TaichiCompilationError` while the graph is built.
- A Field closed over by a kernel or bound through `template_args` is a
  definition-time binding. Its contents may change between runs, but that
  static Field does not appear in the runtime argument dictionary.
- A JIT Graph `ArgKind.NDARRAY` runtime slot accepts a compatible `ti.ndarray`,
  canonical compact dense scalar/vector/matrix Field, or explicit
  `ti.experimental.ndarray_view()`. Graph creates the runtime storage argument
  automatically. Dtype, ndim, element-shape, or layout mismatches fail
  explicitly without copying or implicit staging. AOT Graph currently still
  requires an owning Ndarray.
- Dense Field dependencies are tracked by SNodeTree id and generation.
  Destroying a referenced tree invalidates the Graph; a later tree that reuses
  the numeric id does not revive it.
- An ndarray or texture may provide a compile exemplar in `template_args`, but
  it still needs a matching `ti.graph.Arg` and a real runtime resource in each
  `run()` call.
- An ndarray exemplar must match the symbolic Arg's dtype, ndim, and element
  shape.
- The Graph retains the compiled kernel, not an additional strong solver
  reference from `template_args`.
- `kernel` is normally a decorated primal kernel. An explicit `kernel.grad`
  object is also accepted for a manually managed gradient Graph; run that
  Graph outside `ti.ad.Tape()` / `ti.ad.FwdMode()`.
- `label` is an optional invocation label (for example,
  `"sweep=3/color=red"`). A label is kept on the Graph dispatch rather than
  the shared compiled kernel. It prevents that dispatch from being composed
  with another dispatch, but it does not disable CUDA/Vulkan backend replay.
  Labels remain stable task metadata; per-replay timing or event evidence
  requires explicit telemetry/profiling rather than an ordinary launch path.

### Task manifests and dispatch labels

`kernel.task_manifest(*args, **kwargs)` returns an immutable tuple with one
`OffloadedTaskManifest` for every compiled backend task in the selected
specialization. `Graph.task_manifest()` returns the corresponding
`GraphTaskManifest` tuple for a Graph containing one JIT CGraph segment.
Graphs with native, observation, or structured-control nodes reject this
query until those node kinds share one serializable task list.

Each manifest separates compiler-requested, backend-selected, and proven
actual grid/block geometry. CPU leaves GPU-shaped selected/actual values
unset and reports `actual_geometry_kind="cpu_runtime_scheduler"`. A Vulkan
device-indirect dispatch reports its static selected capacity but leaves
actual grid/block unset with `actual_geometry_kind="runtime_indirect"`, since
the device packet determines each invocation without host readback. Static
and dynamic shared-memory byte counts are reported separately. The
`range_mapping` field is `cpu_scheduler`, `grid_stride`,
`device_bounded_grid_stride`, `one_to_one`, or `not_applicable`. The CUDA
device-bounded mapping has a device-loaded logical end while retaining a
saturation-capped grid-stride worker envelope. In particular, `one_to_one` is a compiler proof that reducing
an indirect grid also reduces the logical range indices visited by the task;
ordinary GPU range kernels remain grid-strided.

`task_id` is stable for the same specialization cache identity, task ordinal,
task kind, compile configuration, device capabilities, and backend. It is not
a cross-backend, cross-configuration, or cross-release identifier. Manifest
objects are read-only observations: querying them may compile a cold
specialization, but it does not launch work, allocate device telemetry, copy
memory, synchronize, or override launch geometry.

Use `label=` on `GraphBuilder.dispatch()` / `Sequential.dispatch()` when the
same compiled kernel represents different sweeps, colors, or phases. For an
ordinary direct call, `ti.profiler.dispatch_label("phase=...")` is a nestable,
thread-local context manager. Labeled kernel-profiler and optional NVTX names
retain the original task name and append `tf.task=<task_id> label=<label>`.
Labels are limited to 128 UTF-8 bytes and reject NUL or line breaks.
Dispatch labels are currently JIT-only; `AOT Module.add_graph()` rejects a
labeled Graph instead of silently removing its observability metadata.

### Direct-kernel `TaskLaunchPolicy`

`kernel.with_launch_policy(policy)` creates a reusable view of a direct JIT
kernel without changing ordinary calls to that kernel. The first qualified
CUDA/Vulkan slice controls the block size of exactly one top-level parallel
range task:

```python
tuned = update.with_launch_policy(
    ti.TaskLaunchPolicy.block(256, mode="require")
)
report = tuned.report(x, y)  # may compile; never enqueues work
assert report.status == "applied"
tuned(x, y)
```

`TaskLaunchPolicy.auto()` retains compiler/backend selection. A `hint` requests
a block size but preserves an explicit source-level
`ti.loop_config(block_dim=...)`; its report then uses
`status="hint_not_applied"`. A `require` must resolve to the requested block or
fails before device enqueue. `block_dim` accepts values from 1 through 1024
that are powers of two or multiples of 32, matching Taichi's existing loop
configuration contract; device and kernel resource limits may still reject a specialization.
The report exposes the immutable policy, backend, status/reason, and the N0
task manifests containing requested, selected, and actual geometry.
`report.resources` adds one immutable `TaskLaunchResourceReport` per task. It
reports compile-time shared-memory use, the source and value of an exposed
backend block limit, representative geometry-valid block sizes, and a
structured reason when a requested candidate was not selected. These sizes
are probes, not tuning recommendations. CUDA register/local-memory allocation
requires a materialized native function, while SPIR-V register allocation is
driver owned; the no-submit report leaves those fields as `None` and explains
why instead of launching a profiler query.

This contract is deliberately narrower than a raw CUDA/Vulkan launch API. It
does not expose grid truncation and currently supports only primal direct-JIT
kernels with one safe parallel range task. Graph, AOT, automatic
differentiation, multi-range kernels, struct/ndarray/mesh iteration, and
user-visible serial side effects remain unsupported. If the policy actually
changes the block size, kernels using `SharedArray`, block barriers, or other
block-sensitive intrinsics fail closed. A source-owned block declaration that
already equals a `require` remains valid because the policy does not change its
geometry.

On CPU, `hint` reports `fallback_auto` and uses the normal worker scheduler;
`require` fails rather than inventing GPU geometry. A cold CUDA/Vulkan policy
specialization must be prepared on the Python main thread. Call
`tuned.report(...)` before launching it concurrently from worker threads.
Validation is performed once without submission; warm calls reuse the normal
launch path and allocate no telemetry buffer. Each distinct policy is a normal
compiled specialization, participates in the runtime specialization budget
and offline-cache identity, and must be prepared again after `ti.reset()`.
Block tuning is backend- and workload-specific, so always compare against
`auto` with end-of-work synchronization. The reproducible paired harness is
`benchmarks/task_launch_policy_bench.py`.

### Complete Graph execution-recipe search with modified CompileIQ

Use `GraphBuilder.freeze()` and the public whole-Graph objects
`GraphOptimizationTarget`, `GraphSearchBudget`, `GraphRecipeHandle`,
`GraphRecipeManifest`, `GraphRecipeSearchStrategy`, `GraphWorkloadContext`,
`GraphEvaluationContract`, `GraphBackendEnvironment`,
`GraphOptimizationDecision`, and `GraphOptimizationReport` for new code:

```python
definition = builder.freeze()
target = ti.graph.GraphOptimizationTarget(
    objectives=(("device_time_ns", "min"), ("materialized_memory_bytes", "min")),
)
workload = ti.graph.GraphWorkloadContext({"case": "representative-a", "size": n})
evaluation = ti.graph.GraphEvaluationContract(
    {"timing": "steady-replay", "synchronize": "after-sample"}
)
environment = ti.graph.GraphBackendEnvironment(
    {"backend": "cuda", "device": device_identity, "driver": driver_identity}
)
session = definition.search_recipes(
    engine="compileiq",
    target=target,
    budget=ti.graph.GraphSearchBudget(evaluation_limit=32, repeat_count=3),
    strategy=ti.graph.GraphRecipeSearchStrategy(mode="staged"),
    workload_context=workload,
    evaluation_contract=evaluation,
    backend_environment=environment,
)
decision = session.run(evaluator)
json_facts = decision.report.to_json()
human_summary = decision.report.to_markdown()

if decision.status == "selected":
    materialized = definition.materialize(decision.selection)
    portable_selection = decision.selection_artifact.to_dict()
```

`evaluator(graph, recipe_handle)` returns named finite metrics. The reserved
`materialized_memory_bytes` metric is filled from the observed physical
manifest when requested. Compatible independent fragments may be composed into
one candidate up to the evaluation budget. Public recipe summaries never expose
raw provider, block, tile, padding, lane, workgroup, or PTXAS knobs. Multi-
objective results retain a Pareto frontier; declaration order is only a
deterministic final preference. Runtime `auto` and ordinary compile remain
unchanged.

`decision.status` is `selected`, `resumable`, `no_feasible_candidate`, or
`failed`; `decision.next_action` distinguishes applying a selection, resuming
from `decision.checkpoint`, and reviewing incomplete or failed evidence. Resume
must recreate the same definition, provider set, target, workload, evaluation,
and backend-environment contracts. The report embeds a summary of the original
CompileIQ facts, Pareto trade-offs, recipe/physical annotations, structured
failures, termination reason, provenance, and the checkpoint. JSON is the
canonical fact source; Markdown is generated from it.

Persist `decision.selection_artifact`, not a Python materializer. A new process
recreates an equivalent frozen definition and provider catalog, calls
`check_recipe_applicability(...)` with current workload/evaluation/environment
facts to decide whether old measurements still apply, and calls
`resolve_recipe(artifact, providers=...)` to reconstruct the structurally
identical recipe. Contract drift is explicit; Forge never unpickles executable
objects. This is report, search, and selection reuse, not AOT binary reuse.

`ti.graph.compileiq_recipe_search(graph)` is the lower-level complete-recipe
API for callers that manage batches, checkpoints, and materialization contexts
directly. The Graph must be backed by a frozen `GraphDefinition`; the API uses
that definition's whole-Graph catalog and has no fallback to the historical
family executable-space adapter. A Graph without a definition is rejected
instead of exposing map width, control route, provider, block, workgroup, or
backend parameters to CompileIQ. Exact fusion partitions, memory staging,
structured control, recording topology, and other low-level decisions remain
available to Forge materializers and may be composed when their fragments are
compatible. The modified CompileIQ boundary receives only complete opaque
recipe tokens, and materialization always crosses the definition-owned
transactional context.

Task-indexed offload identities, materialization, and qualification remain
private diagnostic infrastructure rather than a public
`ti.compileiq_offload_execution_plan_search` API. Ordinary kernels continue to
use source-owned contracts and explicit `TaskLaunchPolicy`; CompileIQ does not
receive raw workgroup, TLS, register, or PTXAS axes.

The modified CompileIQ wheel is accepted by the V2 protocol/capability contract,
not by an exact fork commit or wheel hash. The installed Python source identity
is nevertheless recorded and bound to checkpoints, so code drift invalidates
resume rather than silently reusing measurements. The current commit and wheel
SHA in the manifest describe one qualified build only.

### Device-resident bounded workloads with `DeviceExtent`

`ti.DeviceExtent(capacity)` owns one stable two-element `i32` device state: a
bounded count and sticky overflow status. The capacity, runtime generation,
and allocation identity are immutable, so the same state can be passed to
ordinary kernels, JIT Graph ndarray arguments, and existing Forge primitives
that write a one-element count ndarray:

```python
extent = ti.DeviceExtent(capacity)

@ti.kernel
def publish(requested: ti.i32, state: ti.types.ndarray()):
    ti.device_extent_publish(state, capacity, requested)

@ti.kernel
def consume(state: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(capacity):
        if i < ti.device_extent_count(state):
            out[i] = i

publish(requested_count, extent.state)
consume(extent.state, output)
```

`device_extent_publish()` is a single-writer Taichi-scope operation. It stores
`min(max(requested, 0), capacity)` and sets overflow when clamping is needed,
without host observation. `device_extent_overflowed()` reads the status in
Taichi scope. `extent.reset()` clears both values on device. For an existing
producer that writes element zero directly (including a compatible primitive
called with `extent.count`), `extent.normalize()` enqueues one small clamp and
status kernel without readback. Integrating `device_extent_publish()` into a
producer avoids that extra launch and is preferred on hot paths.

`extent.runtime_arguments("extent")` returns a zero-copy runtime mapping for a
Graph ndarray argument. Count churn keeps the same binding and therefore does
not require Graph reconstruction or workspace allocation. `snapshot()` and
`check()` are explicit host observation boundaries; `check()` raises on
overflow. Bindings fail closed after `ti.reset()` or owner replacement.

`DeviceExtent` does not by itself change a kernel's grid or suppress a command.
A current consumer may use the bounded count inside a fixed-capacity kernel;
backend-specific exact-indirect or masked-capacity dispatch remains a separate
capability and must not be inferred from this state object.

`extent.dispatch_state(block_dim)` creates a 16-byte
`DeviceDispatchState` for a producer and one bounded consumer. Passing it as
`dispatch_state=` to `DevicePrefix.compact()` and as `launch_state=` to
`GraphBuilder.dispatch_bounded()` lets the Vulkan compact scatter publish the
count and indirect grid together. This removes the consumer-side packet
preparation dispatch. CPU and CUDA do not consume the packet; CUDA uses its
exact logical range and may select 12.4+ adaptive physical control. The state, extent,
capacity, and block dimension must match and stale
runtime generations fail closed.

New recorded worklist pipelines normally do not need that adapter. When an
adjacent `DeviceWorklistSequence.finalize_next()` publishes the same extent,
Graph lowering supplies a Graph-instance-owned four-u32 packet to the
producer, records no preparation dispatch, and reuses that packet for
consecutive consumers with the same capacity and block dimension. An
intervening dispatch or an unqualified producer conservatively restores the
standalone preparation path. `DeviceDispatchState` remains supported for
explicit packet producers and backward compatibility.

### Graph-owned private ndarray storage

`GraphBuilder.private_ndarray(name, dtype, shape)` declares address-stable
storage owned by each compiled Graph instance. The returned symbolic ndarray
can be passed to ordinary dispatches, but its name is removed from the public
run schema and callers must not bind it:

```python
builder = ti.graph.GraphBuilder()
scratch = builder.private_ndarray("scratch", ti.f32, count)
builder.dispatch(stage, input_arg, scratch)
builder.dispatch(consume, scratch, output_arg)
graph = builder.compile(workspace_lanes=2)
graph.run({"input": values, "output": result})
```

`Sequential.private_ndarray()` provides the same contract for a recorded
provider region. A provider that constructs a `RecordableGraphAction` directly
can place `ti.graph.GraphOwnedNdarray(dtype, shape)` in `fixed_bindings`.
Materialization happens per Graph instance, identical declaration objects are
shared across composed regions, and memory appears in
`execution_stats().memory.persistent_internal_storage_bytes`.

Mutable private storage is exclusive by default. An asynchronous replay keeps
its lane leased until the completion fence retires; `workspace_lanes=N`
materializes independent copies for genuine overlap. Set
`exclusive_submission=False` only for storage whose contents are immutable
during every overlapping submission. Runtime reset, duplicate bindings, public
argument injection, and stale instance reuse fail closed.

### Bounded and ordered segmented Graph dispatch

`GraphBuilder.dispatch_bounded()` accepts exactly one dynamic-count source:
a symbolic `extent=` backed at run time by the matching `DeviceExtent`, or a
host-known symbolic i32 `count=`. `capacity` is fixed when the Graph is built.
The payload must include the chosen count argument and have exactly one
provable scalar range task. A device-known payload must mask its semantic body
with `ti.device_extent_count(extent)`:

```python
handle = builder.dispatch_bounded(
    consume,
    extent_arg,
    input_arg,
    output_arg,
    extent=extent_arg,
    capacity=capacity,
    block_dim=128,
    physical_grid="extent",
)
graph = builder.compile()
graph.run({"extent": extent, "input": values, "output": output})
```

`physical_grid` is a per-dispatch policy. `"auto"` keeps the backend's
conservative default, `"extent"` requires physical work to track the clamped
device extent without host readback, and `"capacity"` records the explicit
fixed-capacity baseline. `ti.graph.bounded_dispatch_capabilities(policy)`
reports the route for the same policy before construction. An unavailable
explicit `"extent"` route fails closed instead of silently becoming a capacity
launch.

The lowering contract is backend-honest:

- A host-known count is clamped to `[0, capacity]` before enqueue and uses the
  actual scalar range on CPU, CUDA, and Vulkan. No device readback is involved.
- A standalone Vulkan device-known dispatch writes a three-u32 packet on
  device and uses an exact `dispatchIndirect` grid. A qualified adjacent
  recordable producer can instead publish a shared Graph-owned four-u32 packet
  directly. Both payload specializations use `one_to_one` range mapping, so a
  smaller grid really visits fewer logical indices and a zero count skips the
  payload command.
- CUDA defaults to an exact logical-range specialization on every supported
  driver. It loads and clamps `DeviceExtent` on device, then uses the ordinary
  saturation-capped grid-stride scheduler; no host readback or 12.4 symbol is
  required. The physical thread envelope is not an exact grid, but only
  `[0, count)` enters the range body. With Driver API 12.4 or newer,
  `physical_grid="extent"` additionally trims the physical
  node grid to `min(ceil(count / block_dim), saturation_grid)` and disables the
  payload at zero. Correctness does not depend on that update. The forced
  `physical_grid="capacity"` route remains an A/B and diagnostic baseline.
  `TI_CUDA_BOUNDED_DISPATCH_MODE` remains only as a process-wide diagnostic
  override for callers that leave the policy at `"auto"`. None of these
  routes claims CUDA indirect dispatch or conditional termination.
- CPU defaults to the exact scheduler route. It snapshots and clamps the
  device extent once, skips a zero range, and submits positive work as adaptive
  contiguous JIT loops. CPU chunking is independent of GPU `block_dim`, so LLVM
  retains its loop-vectorization opportunity. Set
  `physical_grid="capacity"` only to force the conservative fixed-capacity
  fallback or to run an A/B diagnostic. `TI_CPU_BOUNDED_DISPATCH_MODE` remains
  a process-wide diagnostic override for `"auto"`. CPU ordered segmented
  dispatch does not yet have an exact lowering: `auto` falls back to its
  globally ordered masked route for that operation, while a forced
  `exact_scheduler` request fails closed.

`ti.graph.bounded_dispatch_capabilities()` reports the selected route before a
Graph is built. The returned handle exposes immutable capabilities and stable
workspace accounting. `handle.snapshot(extent)` is an explicit synchronization
and reports useful, executed, skipped, encoded, and overflow counts; the
host-known handle reports the same data without synchronization. On Vulkan,
`forge_producer_fusion_supported` reports support for provider-qualified
publication specialization; each handle's `preparation_dispatches` reports
whether that particular producer/consumer placement actually used it.

On CUDA 12.4+ adaptive graphs, `TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY` accepts
`auto`, `grouped_stateful`, or `per_node`. `auto` selects the internal
grouped/stateful policy: two or more consecutive payloads with the same
extent, capacity, and block dimension share one updater, and unchanged
grid/enabled state returns before traversing the node array or issuing
persistent node-update calls. A singleton retains the
per-node updater. Any different or intervening dispatch terminates the group.
`per_node` remains the conservative A/B route. Per-segment execution reports
expose actual updater groups, grouped payloads, control bytes, and the last
driver error. Explicit submission telemetry can collect the stateful replay
counters `bounded_update_replays`,
`bounded_update_state_changes`, `bounded_update_cache_hits`, and
`bounded_node_api_calls`. `bounded_max_group_size` is static metadata. Public
`Graph.execution_stats()` is side-effect free and never enables this path;
ordinary replay performs no telemetry readback. Rebinding starts a new control
epoch, so measured counters do not combine different extent allocations. This
changes no public resource ownership.

The CUDA 12.4+ qualification probe is a setup-only safety gate. It permits at
most two bounded retries when a freshly uploaded Graph returns a transient
device-node status without a driver error, restoring the probe counter before
each attempt. `probe_transient_retries` exposes that event. Persistent status
or driver failures still reject the adaptive route; ordinary replay does not
inherit probe retries, clock reads, or host synchronization.

`ti.graph.dynamic_work_capabilities()` returns a schema-v5 report that keeps
the count owner, bounded launch, structured iteration termination, worklist,
and ticket observation as separate axes. The worklist section reports append
ordering, single-writer ownership, stable/deterministic transforms, replay
allocation/readback policy, counters, and the active physical launch route. In
particular, CUDA conditional termination is not reported as exact indirect
grid launch. The bounded section also reports the backend publication/reuse
contract, static route admission and its reason, and whether physical
blocks/threads are available at an explicit observation boundary.

`GraphBuilder.dispatch_ordered_segments()` consumes an i32 offsets ndarray and
the same `DeviceExtent`. It appends one reusable payload specialization per
segment position, with an explicit global order between segments; it does not
clone a kernel specialization per color. The builder injects an internal
five-i32 state as the payload's final argument. Read it in Taichi scope with
`ti.graph.segmented_dispatch_begin/end/index/count()` and mask local indices
against the segment count. Offsets are clamped for safe execution; an explicit
snapshot reports invalid topology through `overflow` and per-segment
`invalid_offsets`. The segment count is fixed in `[1, 4096]`.

These APIs are JIT Graph features. AOT export fails explicitly. Consecutive
standalone Vulkan consumers with the same extent, capacity, and block
dimension share one 12-byte Graph-instance packet and one preparation
dispatch; an intervening action conservatively starts a new packet. An
automatically specialized worklist publication instead owns one shared
16-byte packet and adds no preparation dispatch. Vulkan ordered dispatch uses
32 bytes. CPU/CUDA bounded dispatch uses no Python-owned workspace and ordered
dispatch uses 20 bytes. The default CUDA exact route adds a private 16-byte
argument prefix per payload. The 12.4+ adaptive per-node route uses 32
persistent control bytes per payload; a grouped route uses one 80-byte control
plus one eight-byte node handle per grouped payload. An explicit producer-owned
Vulkan launch state remains an external 16-byte compatibility state and reports
zero internal packet bytes. Exact physical work reduction is not a universal
speedup: a light standalone Vulkan payload can cost more than a fixed Graph
because packet preparation and ordering add a dispatch/dependency. Measure the
complete workload against both fixed Graph and direct execution. The paired
harnesses are `benchmarks/dynamic_workload_bench.py` and
`benchmarks/device_worklist_bench.py`.

### `GraphBuilder.dispatch_indirect(kernel, *args, dispatch_packet, template_args=None, label=None)`

`Sequential.dispatch_indirect()` provides the same API. `dispatch_packet` is
a one-dimensional scalar `u32` Graph ndarray argument. Its first three values
are the device-written `{group_x, group_y, group_z}` command for the target
kernel. The target must compile to exactly one offloaded task.

The current native path is Vulkan Graph replay. It records
`vkCmdDispatchIndirect` without host readback, supports zero-group skipping,
and re-records safely when the packet allocation changes. The packet must be
an owning Taichi ndarray with at least three values; Field, external-storage,
and AOT Graph packets fail explicitly. CPU and CUDA also fail closed instead
of substituting a fixed dispatch. Query
`structured_control_capabilities()["device_control"]["parallel_indirect_dispatch"]`
before selecting this path.

### Structured control

| API | Contract |
| --- | --- |
| `GraphBuilder.while_loop(condition, body, *, predicate, max_iterations, control_inputs=(), carried_state=(), counter=None, status=None, chunk_size=None, vulkan_first_chunk_strategy="auto", masked_execution=False, lowering_mode="auto", name="while")` | Append a fixed-schema bounded loop. `condition` and `body` are nonempty `Sequential` values. `predicate`, optional `counter`, and optional distinct `status` are one-element device ndarrays. |
| `GraphBuilder.if_then_else(condition, then_region, *, predicate, control_inputs=(), else_region=None, lowering_mode="auto", name="if")` | Append a fixed two-way branch. Only the selected branch executes. |
| `GraphBuilder.switch(condition, branches, *, selector, control_inputs=(), default_region=None, lowering_mode="auto", name="switch")` | Append a zero-based fixed branch table with an optional default. |
| `Sequential.while_loop(...)`, `.if_then_else(...)`, `.switch(...)` | Append one structured child to a condition, body, or branch `Sequential`. Definitions form a single-owner tree. The public structured-control depth limit is two; deeper definitions, cycles, or reuse at multiple call sites fail before execution during region construction or Graph compilation. |
| `Graph.control_flow_stats()` | Return immutable `GraphWhileReport` / `GraphBranchReport` values for the latest run. Repeated nested calls retain only the latest invocation of each static definition. Reports include `region_path`, `structured_depth`, and encoded/masked work counts; a qualified Vulkan nested-while outer report additionally exposes `nested_region_path`, `nested_logical_iterations`, and `nested_encoded_iterations`. Native CUDA branch reports are materialized lazily, so requesting them is an explicit synchronization point. |
| `ti.graph.structured_control_capabilities()` | Return the schema-v5 portable and device-control contract for the active backend. The result reports the depth-two portable contract and native-leaf/nested-Vulkan qualification separately from structured submit, bounded chunk/replay limits, terminal observation, queue-submit coalescing, and exact dynamic termination. |

Condition regions combine multiple device values in ordinary Taichi kernels;
structured control does not invoke Python callbacks. Graph treats `status` as
a user-defined integer and reports it independently from the continue
predicate. `max_iterations` is mandatory even when the condition also checks
an iteration budget.

CPU uses exact host control over cached dispatch plans. Eligible CUDA
`while`/`if`/`switch` regions select one device-control route when they are
compiled. Driver API 12.8 or newer uses native conditional Graph nodes. Older
drivers with ordinary CUDA Graph capture use Forge's bounded masked Graph:
the selector is latched on device, all work is submitted once, and a private
entry gate prevents inactive Taichi tasks from reaching payload side effects.
This route performs no per-iteration host readback and preserves exact logical
results, but it still issues the encoded task nodes through the CUDA Graph and
therefore does not claim exact dynamic command termination. It accepts at most
4096 encoded dispatches. The capability report distinguishes
`cuda_conditional_graph` from `cuda_masked_bounded_graph` and reports whether
command issue actually stops after exit.

The same `while`/`if`/`switch` APIs are available on `Sequential`, allowing any
of those region kinds to contain one more structured level. Depth-two
semantics are exact. CPU executes the complete tree with exact host control.
At depth two, the parent normally uses exact portable control and a qualified
`auto` leaf may retain its flat native route: CUDA `while`/`if`/`switch`, or
Vulkan `while`. CUDA and Vulkan additionally qualify the ordered
`while -> while[1..8]` single-ticket shape described below. Other depth-two
shapes use portable parent control and reject asynchronous `submit()`.

Vulkan provides two distinct `while` routes. `portable` retains exact
host-observed replay. A qualified `native_required` region uses bounded
device-controlled masking with at most eight replay chunks and a maximum budget
of 512 iterations per region. An explicit positive `chunk_size` is honored per
region and capped at 64; omitting it selects 64 for the compound route. A
combination that needs more than eight chunks fails during Graph construction.
The automatic first-chunk strategy uses compact per-iteration masking. A region
may explicitly request `compact` or `coarse_conditional` through
`vulkan_first_chunk_strategy`; the latter fails closed unless
`VK_EXT_conditional_rendering` is qualified. Under `auto`, each later chunk
uses one conditional command when the extension is qualified. A loop that
terminates inside an active chunk still masks its remaining iterations; a later
inactive chunk skips its shader dispatches at the conditional-command level.
This preserves exact logical results but does not provide exact dynamic command
termination: commands for the active chunk are already encoded.

For the shared ordered depth-two shape described below, Vulkan can run one
bounded replay during ordinary `Graph.run()` or `Graph.submit()` when tracing
is disabled and the outer mode is `auto` or `native_required`. The outer
condition, body gaps, and every inner condition/body may contain only ordinary
dispatches or qualified recordable actions. Every loop requires a counter;
all predicate, counter, and optional status controls must be mutually distinct
one-element `i32` device ndarrays owned by the same Program.
`VK_EXT_conditional_rendering`, nested runtime binding, and ordinary Vulkan
replay mode must be qualified; the kernel profiler and Vulkan dispatch cache
must be disabled. Every loop budget must be from 1 through 64, each inner chunk
size must be positive and no larger than 64 or its budget, and the additive
complete encoded program must contain at most 4096 actions. Shapes outside
this qualification use exact portable-parent control; an eligible leaf
`while` may still use the flat Vulkan route described above. This optimization
still does not make Vulkan `if`/`switch` native and does not provide exact
dynamic command-stream termination.

One Vulkan `Graph.submit()` may contain multiple qualified
`native_required` `while` regions. Forge enqueues them in program order inside
one runtime transaction, batches their Vulkan queue submissions, and publishes
one final `SubmissionTicket` observation boundary. The fixed eight-slot replay
ring remains the inter-invocation backpressure boundary. First-use resource
materialization may flush a preceding command list before the compound batch;
steady execution uses one transaction batch plus the completion-fence
submission. Vulkan `if` and `switch` remain portable-only.

A qualified depth-two outer `while` may also contain from one through eight
ordered leaf inner `while` regions, with ordinary dispatch gaps between them.
CUDA records one bounded conditional or compatibility-masked Graph launch;
Vulkan records one bounded conditional/compact replay. Each inner region keeps
its own predicate, counter, optional status, iteration budget, and (on Vulkan)
chunk size. All controls must be mutually disjoint, every inner region must be
a leaf, and the additive encoded program remains capped at 4096 actions. This
is the portable single-ticket shape for an outer nonlinear iteration followed
by multiple sequential inner solves or searches; it is not parallel branch
execution.

Qualified Vulkan compound replay prepares bindings, dependencies, retained
resources, and submission guards once per region. Its structured command
buffers use allocation-level effects to place RAW/WAR/WAW barriers and retain
conservative controller/global boundaries. The capability keys
`compound_single_preparation` and `structured_barrier_policy` describe these
active policies. The environment switches
`TI_VULKAN_COMPOUND_SINGLE_PREPARATION=0` and
`TI_VULKAN_STRUCTURED_HAZARD_PLANNER=0` are qualification fallbacks to the
legacy per-chunk preparation and eager per-task barrier paths.

`portable` forces the portable route; `native_required` fails closed when the
selected backend cannot honor its native contract. Portable structured-control
Graphs use `run()` and reject `submit()`. Qualified CUDA
`native_required` while/if/switch regions and qualified Vulkan
`native_required` while regions support `submit()`. An ordered device setter
or Vulkan predicate gate consumes control state without a per-region host
readback. A ticket can expose explicit terminal `GraphBuilder.observe()`
snapshots; synchronous `control_flow_stats()` are unavailable for that
asynchronous submission. Qualified depth-two `while` sequences use the native
single-ticket path above; other depth-two shapes still reject asynchronous
submission.

`GraphBuilder.observe()` takes symbolic scalar `NDARRAY` arguments (`ndim=0`).
Bind an ndarray, a canonical dense scalar field, or its program-owned view.
Fixed bindings validate storage at bind/update; dynamic frames validate before
payload submission. Snapshot packing preserves each scalar's dtype and batches
up to four values per task; it does not convert integer IDs through floats.
Independently materialized recipes own independent packing execution caches.
General affine, sparse, and higher-rank storage are not scalar observations.

Terminal observations are attached to the submission completion by default.
CPU and Vulkan use host-visible snapshot slots; CUDA keeps snapshots in
device-local memory and appends an asynchronous copy into a persistent pinned
host slot before recording the ticket completion. Calling
`ticket.observations()` therefore waits only for that completion and then reads
host memory; it does not enqueue a second device readback. Observation slots
are bounded by `TI_GRAPH_OBSERVATION_SLOTS` (default 4). For diagnosis,
`TI_GRAPH_COMPLETION_ATTACHED_OBSERVATION=0` restores deferred readback.
`Graph.execution_stats().memory.observation_readback_mode` reports the active
route.

Opt-in `submit(telemetry="summary")` records each while region's entry
counter/status and terminal counter/predicate/status on device.
For bounded dispatches it also appends one two-word tail snapshot for each
distinct device extent and correlates the result with the compiled task
manifest, dispatch label, launch geometry, and useful/capacity accounting.
Host-known counts are captured from the immutable submission arguments and do
not allocate device snapshot storage. Ordered segmented dispatch reports the
aggregate extent only; per-segment useful work remains unavailable unless the
offsets are explicitly observed.
`ticket.telemetry()` reads
the packed snapshots only after completion and reports the actual stop
iteration, encoded/masked work, active/skipped chunks, host enqueue time, and
queue-counter window. Device-wide queue deltas are marked non-exact because
external graphics/interop producers can submit in the same window. GPU
durations are explicitly `disabled_by_mode` in summary mode. Use
`telemetry="timestamps"` to add backend whole-ticket and region timestamps;
`telemetry=True` is a compatibility alias. Timestamp mode is marked
measurement-path changed and must not be presented as normal execution wall
time.
For nested sequences, every inner definition has its own stop snapshot;
`logical_invocations` reports how many outer iterations invoked it, while
`logical_iterations` remains the final invocation's stop position.

### `GraphBuilder.compile()`, `Graph.bind(arguments)`, `Graph.run(args, *, trace=False)`, and `Graph.submit(args)`

`compile()` freezes the dispatch/sequential definition at the call and returns
a runtime-bound `Graph`. `run(args)` submits one complete graph invocation and
keeps the established fire-and-continue return contract. `submit(args)` uses
the same execution path and returns a completion ticket. Both `run()` and
`submit()` accept either an exact argument dictionary or a `GraphBindingSet`
published by the same Graph's `bind()` method.

| API | Contract |
| --- | --- |
| `GraphBuilder.compile(*, workspace_lanes=1, workspace_saturation='wait')` | Later changes to the builder or original `Sequential` do not modify the compiled graph. Additional workspace lanes are materialized lazily and only affect Graphs with eligible exclusive Graph-owned internal storage, such as a recorded SolvePlan. Provider-fixed worklist storage deliberately requires one lane and rejects a larger value. `workspace_saturation='raise'` fails instead of waiting when every eligible lane is busy. |
| `Graph.bind(arguments)` | Publish a `GraphBindingSet` from a dictionary whose keys exactly match the declarations. Publication snapshots Python scalar/matrix values and qualifies the current device-resource identities; failure publishes no visible version. Construct BindingSets through this factory rather than by calling their constructor directly. |
| `Graph.run(args, *, trace=False)` | `args` must be a dictionary with exactly the declared keys or a `GraphBindingSet` owned by this Graph; missing or extra keys and cross-Graph BindingSets raise `TaichiRuntimeError`. The default returns `None` and does not allocate a dynamic control-flow trace. Qualified depth-two CUDA/Vulkan structured Graphs use the same native single-submission path as `submit()` and wait once at terminal completion; they do not retain synchronous control-flow reports. |
| `Graph.run(args, *, trace=True)` | Run synchronously and return an immutable `GraphControlFlowTrace`. Its ordered invocations contain a `sequence`, static `definition_path`, dynamic `invocation_path`, optional `parent_iteration`, and the invocation's while/branch report. Unlike `control_flow_stats()`, it preserves every repeated nested invocation. Tracing bypasses strict Vulkan nested replay and uses exact portable-parent execution so each invocation is observable. |
| `Graph.submit(args, *, pacer=None, lane=None, on_saturation='wait', telemetry=False, workspace_lane=None)` | Uses the same exact argument, lifecycle, concurrency, and AD contract as `run()`, returns one `SubmissionTicket`, and can opt into shared admission pacing. `lane` remains the pacer lane; `workspace_lane` optionally pins a Graph-owned execution/workspace lane. `telemetry="summary"` adds per-while and bounded-extent snapshots, queue/submission accounting, and the lazy post-optimization pipeline definition without backend timestamp markers. `telemetry="timestamps"` adds whole-ticket/region GPU timestamps; `True` is its compatibility alias. The default adds no telemetry buffers or report. Structured submission accepts qualified CUDA `native_required` while/if/switch regions and qualified Vulkan `native_required` while regions, including multiple ordered regions and qualified depth-two multi-inner sequences. Portable control and unsupported native combinations fail explicitly. |
| `Graph.prepare_telemetry(mode, *, slots=1)` | Explicitly allocate one to the configured maximum bounded telemetry slots and compile required packed snapshot kernels without executing the Graph or reading user resources. `mode="timestamps"` additionally performs one empty instrumented transaction to move backend event/query initialization outside the first measured submission. Preparation covers currently materialized workspace lanes; later lanes reuse compiled kernels but materialize their own bounded storage. `False` is a no-op. |
| `SubmissionTicket.telemetry()` | Wait if needed and return an immutable schema-v5 `GraphSubmissionTelemetry` when telemetry was requested; otherwise return `None`. Region reports include terminal counters, stop positions, nested invocation counts, and `pipeline` is the ticket-owned `GraphPipelineReport`. Nullable GPU duration fields are never inferred from host wall time. |
| `SubmissionTicket.pipeline_report()` | Return the same immutable pipeline object as `ticket.telemetry().pipeline`, waiting if needed. Returns `None` when telemetry was not requested. |
| `Graph._prewarm()` | Warm the current runtime's backend plan; this internal/advanced entry point does not change the argument contract. |

#### `GraphBindingSet`

Location: `taichi_forge.graph`. This is a stable public API constructed by
`Graph.bind()`. It publishes successfully qualified runtime arguments as an
immutable version for production paths that replay one binding set repeatedly:

```python
bindings_a = graph.bind({"source": source_a, "output": output_a})
bindings_b = graph.bind({"source": source_b, "output": output_b})

graph.run(bindings_a)
graph.run(bindings_b)
graph.run(bindings_a)
```

The A-to-B-to-A sequence above does not reconstruct Python storage descriptors,
owners, layouts, or alias proofs on every replay. The native CGraph still does
a generation-qualified resource-identity lookup and reacquires asynchronous
owners/leases for every submission. `fast_path_qualified` is therefore a
performance diagnostic, not a promise that all native safety boundaries have
been removed. Prepublish one BindingSet per set for a bounded collection of
recurring resources as above; `update()` is an explicit publication boundary
and does not promise replay-level cost.

| API | Contract |
| --- | --- |
| `bindings.revision` | Monotonic revision of the currently published version. A failed update does not change it. |
| `bindings.fast_path_qualified` | Diagnostic indicating whether the current version can reuse a preflattened frame. `False` remains correct and uses the conservative slow path for dynamic providers, temporaries, or other per-submission state. |
| `bindings.snapshot()` | Return an immutable mapping of the current public arguments. Python scalars/matrices are snapshotted by value. Device resources such as ndarrays/textures retain object identity, so their device contents may still change between invocations. |
| `bindings.update(values=None, /, **changes)` | Atomically publish a partial copy-on-write update. The candidate is fully qualified before publication; a dtype/layout/alias/owner failure leaves the old version usable. |
| `bindings.replace(arguments)` | Atomically replace the current version from one complete exact mapping, using the same publication checks as `Graph.bind()`. |
| `bindings.statistics()` | Return current qualification and lifetime diagnostics. Fields may be added; applications must not select correctness routes from blocker strings. |

One BindingSet may be read and updated concurrently; an invocation observes
exactly one complete version. A paced `Graph.submit()` takes its version only
after admission waiting, so an update published during that wait may supply the
submitted version. An older version and its resources remain retained while an
asynchronous submission is in flight and are reclaimed after completion.
`ti.reset()` or runtime reinitialization invalidates both the old Graph and all
of its BindingSets.

Concurrent host calls on one graph queue at the complete-invocation boundary;
independent graphs do not share that lock. This guard does not wait for GPU
completion or imply `ti.sync()`. A single workspace lane preserves the old
completion-fence reuse rule. Multiple lanes remove that device-completion wait
for independent submissions, but do not make one workspace reentrant and do
not promise simultaneous execution on separate backend streams. Recompile
graphs after `ti.reset()`.
Destroying any referenced SNodeTree also makes the Graph stale; rebuild it
after constructing the replacement Field layout.

`Graph.run()` and `Graph.submit()` are primal-only. They raise
`TaichiRuntimeError` inside an active
`ti.ad.Tape()` or `ti.ad.FwdMode()` because a backend Graph invocation is
opaque to automatic AD and would otherwise silently omit gradients or dual
propagation. A user may build an explicit `kernel.grad` Graph and run it
manually outside those contexts; Forge does not claim automatic primal/adjoint
Graph pairing.
The same rule is enforced across threads: automatic AD cannot enter during a
Graph host submission, Graph cannot start during AD setup, and overlapping
runtime-global AD contexts are rejected. These checks do not wait for device
completion.

Runtime keys come from the compiled graph definition. Forge still recovers the
actual `ti.graph.Arg` names when a legacy engine adapter writes directly to the
durable AOT plan, but new code should use the public `template_args=` entry
point above. This compatibility path does not relax the contract, and
undeclared extra keys still raise. Direct access to underscored AOT/native
builder objects is not a public user API.

### `SubmissionTicket`

Location: `taichi_forge.graph`; returned by `Graph.submit(args)`.

```python
ticket = graph.submit(args)
if not ticket.done():
    do_independent_host_work()
ticket.wait()
```

| API | Contract |
| --- | --- |
| `ticket.done()` | Poll this invocation without a device-wide synchronization. Returns `True` after successful completion; repeated calls are safe. A deferred backend error is raised when observed. |
| `ticket.wait()` | Wait only for work ordered through this invocation. It does not imply a global `ti.sync()`; repeated calls are safe and return `None`. |
| `ticket.telemetry()` | Return opt-in immutable submission telemetry after completion, or `None` for a default submission. |
| `ticket.pipeline_report()` | Return the opt-in immutable `GraphPipelineReport`, or `None` for a default submission. |
| `ticket.backend` | Read-only backend name for diagnostics. |
| `ticket.sequence` | Read-only, Program-local monotonically increasing completion sequence for diagnostics; it is not a portable persistence or cross-runtime ordering key. |
| `ticket.workspace_lane` | Read-only Graph-owned workspace lane selected for this invocation. It is independent of the optional SubmissionPacer lane. |

CPU tickets are complete when returned. CUDA and Vulkan tickets may be pending,
although very short work can finish before `submit()` returns. Runtime argument
allocations and Forge native-node owners remain valid until backend completion,
even if application code drops the ticket. `ti.sync()` and `ti.reset()` also
retire pending tickets safely. A ticket is a completion handle, not an
`asyncio` future, callback scheduler, cross-Graph dependency object, or
cross-Program synchronization primitive.

### `SubmissionPacer`

Location: `taichi_forge.graph`. A pacer provides bounded, cooperative
asynchronous admission for `Graph.submit()` and CUDA/Vulkan
`BatchedSolvePlan.submit()` calls that share it:

~~~python
pacer = ti.graph.SubmissionPacer(
    2,
    max_in_flight_per_lane=1,
    max_queued=8,
)
physics = solve_plan.submit(
    rhs, out=x, pacer=pacer, lane='physics'
)
render = render_graph.submit(
    render_args, pacer=pacer, lane='render'
)
physics.wait()
render.wait()
~~~

| API / parameter | Contract |
| --- | --- |
| `SubmissionPacer(max_in_flight, *, max_in_flight_per_lane=None, max_queued=64)` | `max_in_flight` bounds incomplete invocations admitted to the backend. The optional per-lane limit prevents one producer from occupying every slot. `max_queued` bounds calls waiting for admission. |
| `lane` | A non-empty string. Equal names on different objects join one scheduling lane. Without a name, each Graph has a distinct default lane, while workspace clones from one batch plan share its default lane. |
| `on_saturation='wait'` | Wait for capacity. Complete host launch turns are granted by work-conserving round robin across lanes and FIFO within a lane. |
| `on_saturation='raise'` | If a launch turn is not immediately available, raise `TaichiRuntimeError` before submitting backend work. |
| `pacer.statistics()` | Return a schema-v2 snapshot with current and peak in-flight/queued counts, grant/rejection/completion/failure counters, admission wait time, per-lane telemetry, and `contract`. The snapshot first polls ready tickets without blocking. |

The host launch turn for one invocation does not interleave with another paced
invocation. After launch, up to `max_in_flight` invocations may continue
asynchronously on the backend. When admission must wait, one blocked caller
acts as a cooperative progress steward and polls all in-flight completions with
bounded adaptive backoff. A completion from the constrained lane is checked
first when a per-lane cap applies, but a later fast completion can release
global capacity before an older slow invocation finishes. Polling is active
only while callers are waiting; the pacer creates no background worker thread.

The pacer measures capacity in complete invocation counts, not memory bytes,
kernel counts, or estimated GPU time. `max_in_flight > 1` only permits multiple
tickets to remain incomplete at once; the API does not create or guarantee
independent CUDA streams, Vulkan queues, concurrent kernel execution, or device
preemption. Each invocation may retain argument allocations, Graph replay
state, operator numeric generations, and caller resources. Persistent solver
workspaces and submissions that do not share the pacer remain outside its
budget. `statistics()["contract"]` reports these limits in machine-readable
form.

For a large solve or Graph on one GPU, start with `max_in_flight=1`. Increase
it to 2 only when a trace demonstrates useful host overlap, improved device
utilization, and acceptable memory and tail latency. Real-time loops should use
a small `max_queued` and prefer `on_saturation='raise'` when they must degrade
or skip work before enqueue. Coarsen many small tasks into a Graph or batch
instead of using ticket count as a substitute for batching.

This is an explicit cooperative boundary. Submissions that do not use the same
pacer remain outside its control, and it cannot reorder commands already in a
CUDA stream or Vulkan queue. It is not a priority scheduler, dependency graph,
or `asyncio` executor. A pacer belongs to the first runtime generation it binds;
create a new one after `ti.reset()`. A backend completion error makes the pacer
fail closed: later admission is rejected while the first completion failure is
preserved.

### Fatal backend errors and runtime reset

Forge keeps the first context- or device-fatal backend error for the current
Program. Examples include `VK_ERROR_DEVICE_LOST` and CUDA execution failures
such as illegal address, device assertion, or launch failure. The error can
first surface from a kernel or Graph call, `SubmissionTicket.done()` /
`wait()`, `ti.sync()`, or GGUI submission because GPU execution is
asynchronous.

After the first fatal error, later kernel, Graph, completion-recording,
synchronization, and Vulkan display submissions fail fast and refer back to
that first error. Forge does not retry the failed invocation or replace the
root cause with teardown errors. Swapchain out-of-date/suboptimal results,
not-ready polling, invalid arguments, unsupported capabilities, and stale
handles remain operation-local and do not by themselves poison the Program.

Treat outputs from the failed or still-in-flight work as undefined. Stop
application producer threads before cleanup, discard those outputs, and call
`ti.reset()` to retire the old Program. Fault-aware teardown skips unsafe
queue, event, fence, or device waits while still releasing host-owned state
and safely destructible backend handles. This is not in-place recovery:
after a real CUDA context loss or Vulkan device loss, creating usable backend
work in the same process is not guaranteed and a process restart may be
required. Graphs and tickets from the old Program never become valid again.

### `Graph.execution_stats()`

Returns a frozen `GraphExecutionReport` snapshot with schema version 7. The
report is a stable public diagnostic API; do not consume `_graph_stats`
directly in application code.

The top-level report includes:

- architecture and lifecycle state;
- node, CGraph, native-node, dispatch, and compiled-task counts;
- runtime-argument and generation-qualified static-dependency counts;
- a pointer-free static layout fingerprint;
- the last aggregate execution path and fallback reason;
- backend-graph, backend-replay, and ordinary-fallback segment counts;
- temporary, observation, telemetry, internal-storage, workspace-lane, and
  retained provider-generation memory/ownership counters; and
- immutable per-segment reports and counter-completeness state.

Per-segment data distinguishes CPU `ordinary`, CUDA capture/exact
replay/patched replay/recapture, Vulkan record/replay, native dispatch, and
ordinary fallback. It also reports bounded persistent argument bytes, replay
eligibility, fallback classification, retry state, and detailed counters.
CUDA conditional replay additionally reports asynchronous control uploads,
waits caused by the two-batch deferred-resource bound, and the peak number of
deferred batches.

Schema v7 retains the v6 `segment.replay_attribution` shape and adds a
deduplicated `provider_memory` tuple. `memory.provider_generation_*` aggregates
only provider generations retained by the Graph. These requested-byte values
are reported separately from Graph-owned `persistent_bytes`; `None`, a false
completeness bit, and `opaque_component_count` preserve vendor-library and
driver allocations that cannot be measured through the loaded ABI. Runtime
allocator statistics remain the source for raw, committed, and cached device
memory.

Production
CGraph replay leaves it disabled and performs no clock reads or cumulative
counter updates. `execution_stats()` is a side-effect-free snapshot: repeated
calls cannot change replay qualification, enable backend instrumentation, or
add a later readback. Uncollected detailed counters remain zero and explicitly
report `counters_complete=False`. Use `Graph.submit(..., telemetry="summary")`
or `"timestamps"` when a particular execution must be measured; ordinary
Graph replay performs no telemetry readback.

### Hardware provider memory reports

Reusable hardware resources such as `ti.hardware.ray.TriangleBLAS` /
`InstanceTLAS`, `ti.hardware.fft.CufftPlan1D`,
`ti.hardware.linalg.CudssPlan`, `ti.hardware.raster.RasterPass`, and stored
cuSPARSE recordings expose `memory_report()`. The frozen
`HardwareMemoryReport` uses schema version 1 and separates known requested
bytes from opaque driver/provider components. It never treats an unavailable
byte count as zero. Closing a provider changes resident requested bytes to
zero while retaining its known capacity for planning; a prior runtime
generation reports `runtime_invalid`.

### `GraphBuilder.append_native(node, *, prewarm=False, admission="explicit")`

Location: `taichi_forge.graph._graph`; available on the Forge graph builder.

Append a Forge DSL-defined native node to a graph.

```python
builder = ti.graph.GraphBuilder()

seq = ti.algorithms.primitive_sequence()
seq.max_abs_delta(values, reference)
builder.append_native(seq, prewarm=True, admission="auto")

graph = builder.compile()
graph.run({})
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `node` | A Forge-defined native node, such as a `PrimitiveSequence`, `DeviceCheckResult`, or `DeviceMetricResult`. |
| `prewarm` | Compile/warm the native node before storing it in the graph. |
| `admission` | `"auto"` accepts only a backend-integrated action; `"explicit"` also permits a diagnostic segmented provider. |

Recordable native nodes can participate in mixed dispatch regions and
structured control. A provider can declare private workspace requirements;
Graph assigns bounded per-invocation arena storage and keeps those bindings out
of the public runtime argument dictionary. Concurrent tickets use independent
arena slots. Providers that cannot record their action, cannot bind the active
slot, or do not qualify the current backend fail before submission.
Consecutive ordinary CGraph and compatible recordable-provider segments are
compiled as one backend region; conflicting fixed or private bindings fail
before backend work is submitted.

`BackendCommandRecording` actions are the explicit exception: each append
flushes an ordinary CGraph segment and remains a separate native-command node.
The root Graph preserves ordering and lifetime across those nodes, but does not
claim CUDA-Graph/Vulkan-command-buffer fusion. CUDA cuBLAS/cuSPARSE composition
and Vulkan AS-refit/query composition are correctness-qualified in this
segmented form. Backend commands remain unsupported inside a structured
`Sequential`.

The default `admission="explicit"` preserves backward compatibility for
provider-local native nodes. It does not make a segmented provider part of one
backend Graph. Use `admission="auto"` for production selection: a fragmented
plan fails at materialization with its stable reason code. The immutable
manifest and `Graph.physical_plan()` distinguish recordable actions, backend
Graph launches, physical queue submissions, and loose helpers.

When submission telemetry is requested, every compiled native action also has
an immutable `NativeActionManifest`. It contains symbolic public runtime
bindings, provider-derived private runtime bindings, resource effects,
temporary requirements, fixed/private binding names,
lifetime-lease count, recordability, qualified backends, update policy, and
synchronization domain. It never contains provider-owned storage objects,
allocation handles, or host/device addresses. An opaque action remains
distinct from a recordable action that was coalesced into a CGraph stage.

`GraphPipelineReport` describes post-optimization execution-root stages, so its
stage boundaries can differ from source append boundaries. `dispatch_count`
and `physical_dispatch_count` are static compiled-definition counts, not the
dynamic iteration count of one invocation. `declared_temporary_bytes` is the
sum of provider declarations and is not a measured peak allocation. Existing
structured-region timestamps are mapped to their matching stage; an ordinary
CGraph/native stage reports `gpu_duration_ns=None` rather than attributing the
whole-ticket duration to that stage. Whole-ticket GPU timing remains available
on the pipeline report when the backend can provide it.

Pipeline schema v2 also exposes immutable `tasks` and `bounded_dispatches` per
stage, together with `task_mapping_status` and `bounded_mapping_status`.
Ordinary compiled CGraph stages report `available`; a task is the same
`GraphTaskManifest` identity used by profiler/NVTX labels and carries requested,
selected, and invocation-resolved geometry when the backend can prove it. A
bounded report carries logical and, when
unambiguous, physical dispatch indices; label; count source/name; capacity;
block size; selected route and physical launch kind; source/useful/executed/
skipped/encoded counts; overflow; and snapshot status. `None` is intentional:
for example, ordered segments do not claim per-segment useful counts from one
aggregate extent. Device snapshots belong to the ticket slot, so reusing or
mutating a `DeviceExtent` after submission cannot change an earlier report.
Structured while/if/switch stages report `structured_runtime_dependent` and
leave both tuples empty: their selected branch, iteration count, and physical
lowering vary at runtime, so Forge does not concatenate internal CGraph-local
indices into a fictitious physical sequence. Structured region timing remains
available through the existing stage fields.

This observability is opt-in and intended for sampling. The default path does
not query task manifests, materialize the telemetry arena, append a snapshot
kernel, or perform a readback. When enabled, one distinct device extent costs
8 bytes per live telemetry slot plus one small tail snapshot dispatch; complete
report materialization waits for the ticket and reads that slot.

Limits:

- Only Forge-defined DSL native nodes are supported. Arbitrary user native
  callback capture is intentionally not public API.
- Native graph replay is JIT/runtime-oriented. AOT native-node serialization is
  not a current public capability.
- Cross-backend graph execution is not implied. The node must match the runtime
  and resources it was compiled for.

See also [Dense Field Graph](dense_field_graph.en.md) and
[Graph compatibility and migration guide](graph_migration_guide.en.md).

## `taichi_forge.ui`

### `ti.ui.DisplayFrame`

Location: `taichi_forge.ui.display_frame`; exported as `ti.ui.DisplayFrame`.

A display-ready frame object for the GGUI `set_image` submission path. Use it
when the caller already owns a displayable representation and wants to skip
generic input detection and repacking. For ordinary images, `canvas.set_image`
remains the preferred API and selects qualified CUDA-Vulkan shared storage
or optimized device-side staging for Taichi field and ndarray inputs.

Constructors:

| API | Input | Parameters / limits |
| --- | --- | --- |
| `DisplayFrame.from_numpy_rgba8(image, *, copy=False, transpose=True)` | Host RGBA image | `image` must be a `uint8` array with shape `(H, W, 4)`. It must be C-contiguous unless `copy=True`. |
| `DisplayFrame.from_texture(texture, *, transpose=False)` | `ti.Texture` | Texture must belong to a compatible graphics backend. |
| `DisplayFrame.from_packed_u32_ndarray(image, *, transpose=True)` | 2D `ti.ndarray(ti.u32)` | Each element is packed RGBA8. The constructor caches field metadata for repeated submission. |

### `Canvas.submit_frame(frame)`

Location: `taichi_forge.ui.canvas.Canvas`.

Submit a `DisplayFrame` to the window display path.

```python
frame = ti.ui.DisplayFrame.from_packed_u32_ndarray(color_buffer)
canvas.submit_frame(frame)
```

Returns: `True` if the frame was accepted by the display path, `False` if it
was dropped by the window frame policy.

Notes:

- `canvas.set_image(frame)` forwards to `canvas.submit_frame(frame)`.
- Ordinary `canvas.set_image(...)` inputs remain supported. CUDA Taichi field
  and ndarray images are packed directly into a Vulkan-exportable shared buffer
  when device identity and external memory/semaphore capabilities qualify.
  Other CUDA/Vulkan inputs retain the established device staging path; neither
  path requires a per-frame device-to-host round trip.
- Contiguous host `uint8` RGBA NumPy inputs are submitted directly through the
  host RGBA8 path. Use `DisplayFrame.from_packed_u32_ndarray(...)` only when
  the producer already writes packed RGBA8 into a 2D `ti.u32` ndarray.
- CUDA-Vulkan sharing is automatic and fail-closed. If qualification fails,
  `set_image()` preserves the same result contract through the established
  staging path. `window.get_display_stats()` reports the path that submitted.

### Display Statistics

Location: `taichi_forge.ui.window.Window`.

| API | Purpose |
| --- | --- |
| `window.is_headless_display()` | Return whether the window uses the offscreen display sink. |
| `window.get_display_stats()` | Return display submission statistics for `set_image` / `show`. |
| `window.reset_display_stats()` | Reset display submission statistics. |

Use these APIs to measure accepted, submitted, dropped, and reused frames, plus
`zero_copy_render_submissions` and `last_render_zero_copy`, in an engine loop.

See also [Display frame submission](display_frame.en.md).

### Window Edge Layout

Location: `taichi_forge.ui.window.Window`.

`window.configure_layout()` reserves any combination of top, bottom, left, and
right logical-pixel regions around one central render viewport:

```python
layout = window.configure_layout(
    top=ti.ui.EdgeRegion(size=44, resizable=False),
    bottom=ti.ui.EdgeRegion(size=28, minimum_size=20),
    left=ti.ui.EdgeRegion(size=260, minimum_size=160),
    right=ti.ui.EdgeRegion(size=320, minimum_size=180),
    minimum_render_size=(320, 240),
)

with layout.region("top") as bar:
    bar.text("Simulation")
with layout.region("left") as panel:
    panel.text("Scene")
with layout.region("right") as panel:
    panel.text("Solver")
```

`None` disables an edge. Each enabled edge is independently resizable and
collapsible: drag its inner boundary, double-click it to collapse, or click
the small edge handle to restore it. `layout.set_collapsed()`, `toggle()`, and
`disable()` modify one edge, while `layout.state` reports logical and
framebuffer rectangles.

The central rectangle drives the Vulkan/Metal viewport and scissor, camera
aspect, scene/circle/line dimensions, and fullscreen images. The swapchain,
clear, screenshot, and depth-buffer shape stay full-window, so no intermediate
render target or pixel copy is added. Viewport-local interaction uses
`window.get_render_cursor_pos()`, `is_cursor_in_render_viewport()`, and
`is_render_input_available()`; existing `get_cursor_pos()` remains
full-window for compatibility.

### Responsive GGUI Panels and Fonts

Location: `taichi_forge.ui.imgui.Gui`, returned by `window.get_gui()`.

| API | Purpose |
| --- | --- |
| `gui.set_font_scale(scale)` | Use a fixed positive font scale and disable automatic height tracking. |
| `gui.set_font_scale_from_window_height(reference_height, reference_scale=1.0)` | Continuously derive the font scale from the window's logical height. |
| `gui.get_font_scale()` | Return the effective font scale prepared for the current frame. |
| `gui.set_font_size(size)` | Use a fixed font size measured in logical pixels. |
| `gui.set_font_size_from_window_height(reference_height, reference_size=16, minimum_size=12, maximum_size=24)` | Continuously derive a bounded, readable logical-pixel font size. |
| `gui.get_font_size()` | Return the effective logical-pixel font size. |
| `gui.set_font_zoom(zoom)` / `gui.get_font_zoom()` | Set or inspect user zoom layered over fixed or responsive sizing. |
| `gui.adjust_font_zoom(delta)` / `gui.reset_font_zoom()` | Change user zoom or restore it to 1 without discarding the base policy. |
| `gui.enable_font_shortcuts(enabled=True)` | Enable or disable edge-region font shortcuts. |
| `gui.sub_window(name, x, y, width, height=None)` | Create a fixed-width panel whose height follows its visible contents. A numeric height preserves fixed-size behavior. |
| `gui.collapsible_section(name, default_open=True)` | Create an independently expandable section inside the current panel. |

Height tracking applies
`reference_scale * logical_height / reference_height` at each GGUI frame
boundary. It uses the logical display height reported by ImGui, not the
framebuffer pixel height, so HiDPI pixel density does not multiply the scale a
second time. A minimized zero-height window keeps the last valid scale.
The logical-size variant applies the same linear policy and clamps it to
`[minimum_size, maximum_size]`. Its defaults produce 16-pixel text at the
reference height while retaining a readable 12-to-24-pixel range. It derives
the scale from the actual default font size rather than assuming a particular
font atlas. User zoom multiplies either base policy. While the pointer is over
an edge region, `Ctrl+wheel` and `Ctrl++/-` adjust it in 0.1 steps, and
`Ctrl+0` resets it; render-viewport wheel input remains available to camera
controls. Shortcut adjustment is clamped to 0.5--3.0.

```python
window = ti.ui.Window("simulation", (1280, 720))
gui = window.get_gui()
gui.set_font_size_from_window_height(
    reference_height=720,
    reference_size=16,
)

with gui.sub_window("Controls", 0.02, 0.02, 0.3) as panel:
    with panel.collapsible_section("Solver") as section:
        if section:
            section.text("PCG")
            iterations = section.slider_int(
                "Iterations", iterations, 1, 100
            )
    with panel.collapsible_section(
        "Rendering", default_open=False
    ) as section:
        if section:
            section.checkbox("Enabled", True)
```

All size and scale arguments must be finite and greater than zero, and
`minimum_size <= reference_size <= maximum_size`. An auto-height panel uses
Dear ImGui's content measurement every frame, so expanding or collapsing any
number of sections grows or shrinks the panel at the next UI frame boundary
without application-side height calculations. Each open section also gets an
independent widget-ID scope. Font policy changes rendering only; it does not
rebuild the font atlas.
Vulkan and Metal share the calculation and panel behavior, with no GPU readback
or extra GPU submission.

## `taichi_forge.linalg` Sparse Linear Algebra

The module provides fixed CSR/BSR patterns, value-only updates, scale-aware
iterative convergence, provider-neutral MINRES, BiCGSTAB, restarted GMRES,
variable-linear FGMRES, and validated symbolic factorization reuse. The complete usage and
backend matrix is in
[Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md).
The runtime-bound operator API is documented separately in
[LinearOperator and SolvePlan](linear_operator.en.md).

| API | Purpose | Supported boundary |
| --- | --- | --- |
| `ti.linalg.SparsePattern.csr(rows, cols, row_offsets, column_indices)` | Create an immutable scalar CSR pattern from current-runtime `i32` ndarrays. | CPU `f32/f64` values; CUDA/Vulkan `f32` values. Indices are sorted, unique, and in range per row. |
| `ti.linalg.SparsePattern.bsr(block_rows, block_cols, block_size, row_offsets, column_indices)` | Create an immutable BSR pattern. | Block size 2, 3, 6, or 12. Solver support is narrower than SpMV support. |
| `pattern.matrix(values)` / `SparseMatrix.from_pattern(pattern, values)` | Bind independent numeric values to shared immutable indices. | `values` is a one-dimensional scalar Taichi ndarray on the current runtime. |
| `matrix.update_values(values)` | Replace compressed values without rebuilding indices. | Stored scalar count and compressed order must remain unchanged. |
| `ti.linalg.SparseCG(A, b, ..., atol, preconditioner, rtol)` | Solve an SPD system and return `(x, converged)`. | CPU mutable/fixed CSR/BSR; CUDA scalar CSR/fixed BSR; no Vulkan stored CG. |
| `ti.linalg.SparseMINRES(A, b, ..., atol, rtol)` | Solve an explicitly stored symmetric-indefinite system. | CPU mutable/fixed CSR/BSR, `f32/f64`; identity preconditioner. |
| `ti.linalg.SparseBiCGSTAB(A, b, ..., atol, rtol)` | Solve an explicitly stored nonsymmetric system. | CPU mutable/fixed CSR/BSR, `f32/f64`. |
| `ti.linalg.SparseSolver` | Direct LLT/LDLT/LU factorization and solve. | CPU mutable Eigen providers and documented CUDA scalar-CSR route; no Vulkan. |
| `ti.linalg.OperatorTraits(...)` / `.spd()` | Declare mathematical properties without sampling or inference. | CG/PCG require trusted self-adjoint and positive-definite traits; MINRES requires trusted self-adjointness and rejects a declared-singular operator. |
| `ti.linalg.LinearOperator.from_sparse_matrix(A, traits=...)` | Bind fixed CSR/BSR as a runtime-owned linear map. | CPU `f32/f64`; CUDA/Vulkan `f32`; no copy or fallback. |
| `LinearOperator.from_kernel(..., adjoint=...)` / `.from_graph(..., adjoint=..., state=...)` | Bind an exact f32 ndarray kernel ABI or role-qualified compiled Graph; an integer size is square shorthand and a tuple is `(range, domain)`. | CPU, CUDA, Vulkan; explicit adjoint; operator-owned topology/numeric/workspace snapshots. For each distinct dependent pure-dense SNodeTree, `state` supplies at least one representative root-dense scalar/Vector/Matrix Field and retains that tree zero-copy. Missing/extra trees, any sparse/dynamic node in a dependent tree, destroyed storage, or cross-runtime state fails closed. Keys are diagnostic labels; matching is tree-granular. |
| `operator.graph_action(input_arg, output_arg, *, adjoint=False)` | Record one compiled-kernel or direct-dispatch compiled-Graph operator apply into a Graph root or structured `Sequential` body. | CPU/CUDA/Vulkan f32. Generic and legacy compiled Graphs preserve ordered multi-dispatch forward actions; the generic form records an explicitly registered adjoint. Compatible values-only generations rebind at launch and are pinned per submission without an outer-Graph rebuild. Topology/schema/SNode/runtime changes fail closed. A single recorded action does not guarantee a speedup; the intended benefit is multi-action composition. |
| `ti.linalg.FieldLinearOperator(matvec_kernel)` | Wrap the callback-only `(x, y)` field ABI used by `MatrixFreeCG` and `MatrixFreeBICGSTAB`. | Field-shaped legacy contract; no provider capability, resource-generation, storage-view, composition, or SolvePlan adaptation. |
| `ti.linalg.vector_view(field, *, indices=None, offset=None, length=None, stride=1)` | Declare a canonical root-dense scalar/Vector/Matrix field as a runtime-bound scalar-flat vector, optionally with an indexed subset/permutation or immutable range. | 1D/2D/3D, `f32/f64`, subject to operator/provider/backend dtype support. `indices` are nonempty, in-range, unique `i32` values frozen at construction. A range is host-known, positive-stride, bounds-checked, and mutually exclusive with indices. Compact `stride=1` ranges retain qualified direct storage; strided/indexed views use device staging. Sparse SNodes and noncanonical layouts fail explicitly. |
| `ti.linalg.vector_io_capabilities()` / storage-view metadata | Inspect the versioned storage, layout, execution mode, zero-copy eligibility, and indexed-topology contract. | Compiled kernels directly bind compact and rank-one scalar affine runtime storage on CPU/CUDA/Vulkan. Compiled Graphs directly bind compact storage and preserve zero-copy affine execution through backend-qualified dispatch. Native CSR/BSR accepts compact direct storage on CPU/CUDA; Vulkan dense fields and solve boundaries use reusable device staging. |
| `operator.apply(x, out=None, *, alpha=1, beta=0, addend=None)` / `operator @ x` | Synchronously compute `out = alpha * A(x) + beta * addend`. | Scalar one-dimensional ndarray, scalar-linearizable dense field/view, or qualified `DenseNdarrayView`; general coefficients on CPU and for CUDA/Vulkan f32 ndarrays; `beta=0` does not read addend; input/output may not alias. GPU addend/output alias uses one persistent scratch. |
| `operator.parameterized_affine(other=None, *, alpha, beta, alpha_range, beta_range)` / `.update_parameters(...)` | Build `alpha*A + beta*B`, or an identity shift when `other` is omitted, and atomically publish coefficient generations. | CPU `f32/f64`; Program-bound CUDA/Vulkan `f32`. Closed ranges are mandatory and bound trait proofs. Updates require the exact current version; cached Graph actions rebind without rebuild and pin in-flight generations. |
| `operator.scaled(...)`, `.shifted(...)`, `operator + other`, `.compose(...)`, `.adjoint()`, `block_diagonal(...)`, `identity(...)` | Construct minimal linear-operator algebra. | CPU `f32/f64`; Program-bound f32 scale/shift/sum/compose on CUDA/Vulkan. GPU fixed-layout block diagonal requires direct-affine f32 leaves and uses consecutive zero-staging subviews. With recordable leaves, the container is itself a Graph/SolvePlan action with private derived subviews and no block-sized temporary. A recordable shift adds one in-place `axpby`, with no identity-sized temporary. Explicit adjoint capability is required. |
| `ti.linalg.inverse_block_diagonal(inverse_blocks, block_size, *, assume_spd)` | Build a recordable fixed-linear scalar or small-block inverse preconditioner from caller-supplied row-major f32 inverse blocks. | CPU/CUDA/Vulkan; block size 1-4 with a specialized kernel per size and constant-size topology metadata. Forge does not read back, invert, regularize, or infer SPD. Compatible numeric updates copy only inverse values and rebind into cached Graph submissions. |
| `ti.linalg.SmallBlockInverseBuilder(block_size, block_count, regularization=0, pivot_tolerance=...)` | Build row-major f32 inverse blocks on device through `build()` or one-dispatch `graph_action()`. | CPU/CUDA/Vulkan, sizes 1-4. Pivot tolerance is relative to the largest absolute regularized block coefficient; f32-unrepresentable controls fail. Per-block device status is 0 success, 1 non-finite, or 2 singular/ill-conditioned; failed output is zero. No implicit status readback or SPD inference. |
| `ti.linalg.qualify_operator(operator, reference=..., ...)` | Generate versioned, JSON-serializable provider-neutral protocol evidence. | Records oracle/adjoint/generalized apply, synchronous timing, resource stamps, and native counters; unsupported paths do not fall back. |
| `summarize_operator_qualifications(reports)` | Build a deterministic backend/provider support matrix from detached reports. | Schema-v1 JSON dictionary preserving passed/failed/unsupported status for every check. |
| `ti.linalg.experimental.qualify_solve_plan(plan_or_factory, rhs, reference=..., ...)` | Generate versioned correctness, lifecycle, and execution evidence for one single or independent-batch plan. | Separates build/first/warm wall time and qualified async submit; records true residual, A/M identity, iteration/work/resource/pacer telemetry; device time is never inferred. |
| `summarize_solve_qualifications(reports)` | Build a deterministic solver/backend/provider/policy matrix from detached reports. | Schema-v1 JSON dictionary retaining checks, timing availability, normalized work metrics, and original telemetry. |
| `ti.linalg.experimental.PreconditionerPlan(target, action, method=..., behavior=..., selection=...).setup()` | Establish provenance and compatibility for a fixed-linear approximate inverse or a bounded variable-linear action table. | `action` is one operator for `fixed_linear` or a 1-32 operator sequence for `variable_linear`; the latter uses `selection="cyclic"` for FGMRES. CPU/CUDA/Vulkan; target updates are stale by default. A recordable fixed-linear action can be consumed by Graph PCG while each ticket pins its exact approved target/action pair. A variable table validates every action before publishing any generation. |
| `preconditioner.pin()` / `.apply(r, out=None, iteration=0)` / `.metadata` / `.statistics()` | Pin exact target/action generations and apply a native action. | No Python hot-path callback; `iteration` selects a variable-linear action. Reports build/accepted stamps, schedule update counters, generation publish/retire/release telemetry, and refresh operation/transfer/resource counters. Solver telemetry separately reports action selections and wraps. |
| `ti.linalg.experimental.SolvePlan(operator, method=..., preconditioner=..., execution_policy=..., check_interval=..., restart=..., submission_workspace_lanes=1, submission_workspace_saturation="wait")` | Build a persistent CG, PCG, MINRES, BiCGSTAB, restarted GMRES, or FGMRES plan. | CPU GMRES/FGMRES support compatible `f32/f64` host actions. CUDA/Vulkan `f32` support fixed stored or compiled providers; FGMRES consumes a finite variable-linear action table, stores `restart` preconditioned basis vectors, and uses direct native submission. Submission workspace options apply only to the cached f32 CG/PCG `submit()` Graph. Restart is 8, 16, or 32. See the detailed guide for the complete provider and policy matrix. |
| `plan.graph_action(rhs_arg, output_arg, initial_guess=..., name=...)` | Inline a complete f32 CG/PCG solve as a structured Graph action and expose device terminal resources. | Requires recordable A and optional fixed-linear M. `action.allocate_terminal()` supplies the explicit runtime packet; one compiled Graph instance owns one completion-fenced workspace lane. |
| `action.statistics()` / `plan.graph_action_statistics()` | Attribute enclosing Graph submissions, observed ticket completions, and explicit terminal snapshots to one action or all actions created by a plan. | Adds no device dispatch or implicit terminal readback. Iterations advance only when `terminal.snapshot()` materializes the packet; a nested packet describes its last action invocation. |
| `plan.submit(rhs, initial_guess=None, out=None, pacer=None, lane=None, on_saturation="wait", telemetry=False, workspace_lane=None)` | Submit one complete solve and return `SolvePlanSubmission`. GPU submissions accept the same `False`/`"summary"`/`"timestamps"` modes as `Graph.submit()`; `True` aliases timestamps. | CUDA/Vulkan require a recordable f32 CG/PCG device-convergent plan, reuse one cached Graph/ticket, and materialize terminal state only in `result()`. CPU executes exact native `solve()` synchronously and returns a completed lane-0 submission without Graph telemetry. |
| `plan.prepare_telemetry(mode, with_initial_guess=False)` | Build and prepare the selected cached GPU submission variant without submitting a solve or allocating caller output/terminal state. | CUDA/Vulkan device-convergent plans only; no-initial and explicit-initial variants are independent. |
| `plan.submission_statistics()` | Inspect cached submission variants, lanes, bytes, calls, failures, telemetry requests, and terminal materializations. | No-initial and explicit-initial variants are independent Graphs and therefore own independent lane pools. |
| `plan.solve(rhs, initial_guess=None, out=None)` | Return an immutable `SolveResult` with solution, true-residual terminal state, and structured `breakdown_reason`. | Scalar one-dimensional ndarray or supported dense field/view. Qualified recordable Graph Krylov binds compact/contiguous Field operands in its preamble/epilogue without separate pack/unpack submissions; other Field layouts use reusable device pack/gather and unpack/scatter. No conversion occurs inside an iteration. RHS/output aliasing is prohibited. |
| `plan.execution_capabilities()` | Return the backend/provider policy matrix, selected default, automatic replay primitive, and structured unsupported reason. | CUDA stored f32 CSR/BSR CG/PCG defaults to auto-upgrading `bounded_convergent`. CUDA compiled-kernel f32 CG/PCG and compiled-Graph CG report `device_convergent` as `explicit_only`; recordable compiled-Graph PCG selects it automatically. Vulkan recordable compiled-kernel CG/PCG and compiled-Graph PCG select it automatically, while compiled-Graph CG remains explicit-only. Direct requests fail without fallback when unavailable. |
| `ti.linalg.experimental.BatchedSolvePlan(operator, batch_size, independent_systems=True, ..., active_system_compaction=False)` | Build homogeneous independent f32 CG/PCG over contiguous flat partitions. | CPU/CUDA/Vulkan; per-system tolerance, status, and iteration count. Recordable A/M may explicitly select `device_convergent`; the default remains host-check execution. Active compaction is an explicit CUDA device-convergent recurrence-only capability; it does not compact A/M provider applies. |
| `batch_plan.solve(rhs_flat, initial_guess=None, out=None)` | Return a flat solution and immutable per-system `BatchedSolveResult` tuples. | Independent direct-sum systems only; not multi-RHS or block Krylov. |
| `batch_plan.submit(rhs_flat, initial_guess=None, out=None, pacer=None, lane=None, on_saturation='wait', telemetry=False)` | Submit a solve and return `SolveSubmission`; GPU device-convergent submissions accept `False`, `"summary"`, `"timestamps"`, or compatibility `True`. | CUDA/Vulkan with `fixed_budget_masked` or qualified recordable-A/M `device_convergent`; one plan-owned slot; optional shared `SubmissionPacer`; exact generations and arrays are retained through completion. Device-convergent execution uses one Graph ticket and exposes its exact stop counter at terminal materialization. |
| `batch_plan.prepare_telemetry(mode)` | Materialize and prepare telemetry for the cached batched Graph replay without solving. | CUDA/Vulkan `execution_policy="device_convergent"` only; fixed-budget host-orchestrated submission has no Graph telemetry arena. |
| `SolveSubmission.done()` / `.wait()` / `.result()` / `.telemetry()` / `.workspace_lane` | Observe completion, materialize one packed terminal state, and return result or opt-in telemetry. | `done()` does not release the slot; `wait()`/`result()` surface backend faults and release it. `telemetry()` returns `None` unless requested and never invents unavailable device counters. |
| `batch_plan.clone_workspace()` | Create an equivalent plan with independent Krylov state. | Required for concurrent submissions; each clone owns another full workspace. Inspect `clone_workspace_payload_bytes` before constructing a pool. |
| `batch_plan.workspace_pool(lanes, workspace_saturation='wait')` | Create lazy, round-robin independent workspace/Graph lanes. | Every materialized lane owns a full payload and Graph instance. Submit may pin `workspace_lane`; saturation waits or raises explicitly. Pool statistics report materialized/capacity bytes and do not promise physical GPU overlap. |
| `operator.statistics()` / `plan.statistics()` | Return provider/plan execution and workspace diagnostics. | Single-system GPU plans report exact A/M, dot-product, multi-dot, and vector-update work where available, logical/executed/masked iterations, workspace bytes, action counters, preconditioning side, and chunk activity. Recordable Graph Krylov additionally reports its two-stage reduction strategy, block size, items per thread, partial count, fixed scratch bytes, and encoded window sizes. Batched schema v5 separates host-check recurrence replay from whole-solve `device_convergent_replay`, reports terminal-packet and compaction resources, and includes the exact logical stop counter; unavailable backend encoded/masked counts are not inferred. A diagnostic snapshot is not part of the numerical result. |

Iterative convergence uses
`||b - A x||_2 <= max(atol, rtol * ||b||_2)`. Taichi does not infer
symmetry or positive definiteness. Unsupported format/backend operations fail
without a host fallback.

For batch size `B`, per-system size `N`, and f32 storage, one host-check CG
plan has a logical ndarray workspace payload of
`12 * B * N + 92 * B + 24` bytes. PCG uses
`16 * B * N + 92 * B + 24` bytes. Device-convergent execution adds 12 bytes;
CUDA active-system compaction adds another `4 * B + 8`. Every clone or
materialized pool lane adds the corresponding payload. These values include
the packed terminal packet but exclude allocator rounding and reservation,
backend driver objects, RHS/output/initial-guess vectors, and operator/preconditioner
resources; `statistics()["resources"]` also reports the exclusion list.

`SparseSolver.analyze_pattern(A)` may be reused across
`factorize(B)` calls only when the complete compressed index pattern
is identical. A value update after factorization makes the factorization stale
until `factorize()` runs again. All patterns, matrices, ndarrays, and
solvers are Program-generation objects and become invalid after
`ti.reset()`.

## Sparse Layout APIs

For workload-level layout selection and feature status, see
[Choosing a sparse layout](sparse_layout_selection.en.md).
For physics operator and solver selection, see
[Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md).
For construction, solve, lifecycle, and backend tables, see
[Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md).

### `SNode.hash(...)` and `FieldsBuilder.hash(...)`

Location: SNode and FieldsBuilder APIs.

Experimental fixed-capacity hash SNode layout.

```python
x = ti.field(dtype=ti.f32)
root = ti.root.hash(ti.i, dimensions=1024, expected_active=128)
root.place(x)
```

Signature:

```python
hash(axes, dimensions, *, max_active=None, expected_active=None,
     capacity=None, hash_load_factor=None)
```

Parameters:

| Parameter | Meaning |
| --- | --- |
| `axes` | Axis or axes covered by this SNode. |
| `dimensions` | Logical dimensions. |
| `expected_active` | Expected number of active entries; capacity is derived from the load factor. |
| `max_active` | Compatibility alias for `expected_active`; it is a sizing input, not a hard active-entry limit. |
| `capacity` | Explicit physical capacity. |
| `hash_load_factor` | Per-node load factor override. |

Limits:

- Exactly one of `expected_active`, `max_active`, or `capacity` must be given.
- `expected_active` and `max_active` derive table slots; only the resulting
  physical table capacity is a hard bound.
- Supported public backends are CPU, CUDA, and Vulkan.
- Capacity is fixed before JIT; there is no automatic grow/rehash path.
- `hash` is not supported under quantized layouts such as `quant_array` or
  `bit_struct`.
- Sparse or complex child layouts should be validated under the target backend
  before production use.

See also [Hash SNode](hash_snode.en.md).

## CLI

### `ti cache warmup script.py [-- script-args...]`

Location: Forge CLI.

Runs a Python script once with offline cache warmup enabled, so later runs on
the same arch, driver, compile options, and source hash can reuse compiled
artifacts.

Limits:

- Warmup does not make backend artifacts cross-arch compatible.
- Only safely reusable frontend/source cache state is shared; backend artifacts
  remain separated by backend and compile configuration.

See also [Compile and cache guide](cache_compile.en.md).
