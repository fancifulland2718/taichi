# LinearOperator and Experimental SolvePlan

> This page describes the published Taichi Forge `0.6.2` release contract; see
> [release notes](release_notes.en.md) for version attribution.

`ti.linalg.LinearOperator` provides a runtime-bound linear-map abstraction for
stored sparse matrices, compiled Taichi kernels, and compiled Graphs. It is a
general numerical API: applications may use it for discretized PDEs, implicit
systems, graph problems, optimization, or other linear algebra without
introducing physics-domain objects into the Taichi DSL.

`LinearOperator`, `OperatorTraits`, storage-view helpers, composition helpers,
and operator qualification are public `ti.linalg` APIs. Solver execution
objects such as `SolvePlan`, `PreconditionerPlan`, and `BatchedSolvePlan`
remain under `ti.linalg.experimental`; their backend and numerical support
boundaries are documented separately below.

## Core model

A `LinearOperator` represents `y = A x` with:

- a scalar dtype and `(rows, columns)` shape;
- one concrete provider bound to the current Taichi `Program`;
- explicit mathematical traits such as self-adjointness and positive
  definiteness;
- observable capabilities and resource-generation metadata; and
- one reusable native execution plan.

Public vector arguments retain a one-dimensional scalar-flat mathematical ABI.
`LinearOperator.apply()` accepts scalar one-dimensional Taichi ndarrays,
supported dense fields, explicit `DenseNdarrayView` objects, or `VectorView`
objects. Single-system `SolvePlan.solve()` accepts scalar one-dimensional
ndarrays, supported dense fields, or `VectorView` objects; explicit
`DenseNdarrayView` operands are not part of its contract. Vector payloads do
not pass through the host, matrix-free providers are not materialized, and
unsupported operations do not change backend.
All operators, plans, providers, ndarrays, and field views belong to one runtime
generation. They become invalid after `ti.reset()` and cannot be rebound to a
later `ti.init()` session.

### Precision boundary

Compiled-kernel and compiled-Graph matrix-free providers currently use an
exact `ti.f32` vector ABI on CPU, CUDA, and Vulkan. Consequently, GPU
matrix-free solver execution through these providers is qualified for
`ti.f32` only. `ti.f64` support for this ABI is outside the current release
boundary.

The runtime does not silently downcast `ti.f64` vectors, materialize a
matrix-free provider, or substitute a different provider or backend.
Unsupported dtype/provider/backend combinations fail explicitly. Stored
operators and dense fields may still use `ti.f64` where the corresponding
provider, solver, and backend rows in the support tables below allow it.

## Prepared apply (0.6.3)

For repeated application with fixed storage, use:

```python
prepared = operator.prepare_apply(input_array, output_array, adjoint=False)
prepared.run()
builder.append_native(prepared.record())
graph = builder.compile()
prepared.close()
graph.run({})  # The recorded snapshot has its own retained reference.
graph.close()
```

`ti.linalg.PreparedOperatorPlan` pins one provider topology/numeric/coefficient
generation and validates dense input/output extent, layout and disjointness when
preparing. It performs no mathematical application or field staging. Input
contents and explicitly declared live field state remain live. In contrast,
`update_numeric()` or parameter publication **does not update an existing prepared
plan**: prepare again to adopt that generation. The existing `graph_action()`
continues to rebind compatible generations dynamically.

This path reuses qualified f32 compiled-kernel/compiled-Graph actions and their
supported compositions; it does not add a stored/vendor sparse route, f64 ABI,
generalized alpha/beta apply, or indexed-view packing. Derived subviews of
unresolved composition workspace are rejected. `record()` expands only at Graph
root, with an opaque semantic boundary; it is not a new fusion claim. Provider
snapshots are retained without copies, while each compiled Graph owns its own
composition workspace. Preparation may compile helper kernels and allocate that
workspace; warm the Graph before timing. No provider/schema rediscovery is added
to prepared replay, and normal Graph capture/replay eligibility still applies.

`run()`, `report()`, `close()` and context management are available. Reports give
pinned generation revisions, dispatch/workspace facts and binding status, not a
performance guarantee or a complete driver VRAM measurement. Closing a prepared
plan retires its own Graph; separately compiled recordings retain their snapshot
and workspace until they close. Runtime reset or live source-tree retirement
invalidates execution. Plans are not executable cross-process serialization.

## Dense fields and VectorView

A supported dense field may be passed directly as `input`, `out`, or `addend`
to `LinearOperator.apply()`, and as `rhs`, `initial_guess`, or `out` to
`SolvePlan.solve()`:

```python
rhs = ti.field(ti.f32, shape=(nx, ny))
solution = ti.field(ti.f32, shape=(nx, ny))

operator.apply(rhs, out=solution)
result = plan.solve(rhs, initial_guess=solution, out=solution)
assert result.solution is solution
```

The supported field contract is:

- `ti.f32` or `ti.f64`, subject to matching dtype support in the selected
  operator, provider, and backend;
- a 1D, 2D, or 3D `root -> dense -> place` scalar field;
- a canonically packed `ti.Vector.field` or `ti.Matrix.field`; and
- fixed shape, the current `Program`, and a live `SNodeTree`.

A scalar field whose records contain fixed sibling padding can still be a
supported staged full view, but it is not a direct compact candidate. Packed
Vector and Matrix fields must retain their canonical component layout.

A field maps to operator space in scalar-flat order: canonical index-shape
order first, followed by Vector lanes or row-major Matrix components. The
scalar extent is therefore `prod(index_shape) * prod(element_shape)` and must
match the operator domain or range exactly. Sparse SNodes such as `pointer`,
`bitmasked`, `dynamic`, and `hash`, quantized storage, arbitrary nested dense
trees, and noncanonical packed-component placement fail before submission.

Use `vector_view()` to declare an explicit scalar subset or permutation of a
dense field:

```python
indices = ti.ndarray(ti.i32, shape=active_size)
indices.from_numpy(active_scalar_indices)

rhs_view = ti.linalg.vector_view(rhs, indices=indices)
solution_view = ti.linalg.vector_view(solution, indices=indices)
result = active_plan.solve(rhs_view, out=solution_view)
```

For a continuous participant-owned interval, use an immutable scalar-flat
range instead of an index map:

```python
rhs_view = ti.linalg.vector_view(rhs, offset=offset, length=count)
solution_view = ti.linalg.vector_view(solution, offset=offset, length=count)
```

`offset`, `length`, and `stride` are host-known integers interpreted after the
field has been flattened in canonical scalar order. `offset >= 0`,
`length > 0`, and `stride > 0`; the final selected scalar must remain in the
source extent. Range arguments and `indices` are mutually exclusive. A
`stride=1` compact range preserves a direct dense-storage descriptor and can
bind the qualified CUDA/Vulkan Graph Krylov boundary without a gather or
scatter. `stride>1` remains a device-staged affine view and is never reported
as direct contiguous storage.

`indices` may be a one-dimensional `ti.i32` ndarray or root-dense scalar field.
View construction copies, validates, and freezes the index topology. Indices
must be nonempty, in the source scalar-extent range, and unique. Later mutation
of the original indices does not change an existing view. Construction performs
one explicit host validation; vector values remain device resident throughout
apply and solve. Indexed scatter overwrites selected scalar entries only and
leaves all other field entries unchanged.

Dense-field execution is provider-qualified. The overwrite form
`operator.apply(input, out=output, alpha=1, beta=0)` directly binds canonical
compact full fields when the selected provider reports
`dense_storage_operands=True`:

```text
dense field descriptor -> resolved range + submission lease -> provider operands
```

Compiled-kernel and direct-dispatch compiled-Graph providers accept direct
field operands on CPU, CUDA, and Vulkan. This includes ordered multi-dispatch
Graph providers. Fixed native CSR/BSR providers accept them on CPU and CUDA;
the Vulkan native sparse provider remains staged. Direct input and output must
be non-aliasing, have the exact dtype and scalar extent, and both qualify for a
compact scalar-flat mapping.

All other supported cases retain an explicit device-staging path:

```text
dense field/view -> device pack or gather -> scalar ndarray provider ABI
scalar ndarray result -> device unpack or scatter -> dense field/view
```

This includes indexed views, padded or non-compact fields, generalized apply,
`out=None`, and `SolvePlan.solve()` outside the qualified Graph Krylov scope.
Each operator or plan owns and reuses compatible staging ndarrays. A warm solve
does not allocate staging, and conversions occur only at apply/solve boundaries,
never inside a Krylov iteration. A field `out` is unpacked or scattered before
the synchronous API returns.

CUDA/Vulkan recordable f32 CG/PCG has a narrower direct-binding path for
canonical compact full-field or `stride=1` scalar-range RHS, output, and
initial-guess operands. The Field
is a runtime argument of the solver Graph; its preamble and epilogue copy the
boundary values to and from one plan-owned ndarray used by the recurrence. This
removes separate pack/unpack submissions, one completion synchronization, and
one of the two boundary staging vectors. It is not provider-native zero-copy:
the iterative solution deliberately remains ndarray-backed so Field/SNode
addressing is not repeated in every Krylov update. Indexed, strided, padded,
masked, or scatter views retain the reusable staging path.

RHS/input may not overlap output. An `initial_guess` or `addend` may be the exact
same view as output; nonexact overlap fails. Stable raw-field bindings are
qualified once per operator or plan and then reuse the same implicit view and
transfer plan or Graph runtime binding. Native bulk transfer is used where the
backend supports it; other staged layouts use compiled conversion Graph replay.

Capabilities and actual conversion costs are observable:

```python
capabilities = ti.linalg.vector_io_capabilities()
view_metadata = rhs_view.metadata
stats = plan.statistics()["vector_io"]
```

`VectorView.metadata["zero_copy_candidate"]` reports only whether the physical
full-field layout can be flattened without a copy; the provider capability and
the requested operation still decide execution. Statistics report direct
dense-field submissions, staging builds/reuses/reserved bytes, implicit-view
and transfer-plan builds/reuses/evictions, native/Graph transfer submissions,
pack/unpack and indexed gather/scatter calls, logical bytes, direct ndarray
bindings, direct Graph-solve boundary submissions and bindings, completion
synchronizations, and coalesced operator synchronizations.
`execution_capabilities()["direct_dense_field_solve"]` distinguishes support,
enablement, and whether the latest solve selected a complete direct Field
boundary. `execution_mode="device_staged"` means the field API is supported
without moving vector values through the host; it does not mean provider-native
zero-copy.

## Stored operator and CG

Use `aslinearoperator()` or `LinearOperator.from_sparse_matrix()` with a fixed
CSR/BSR matrix. The operator strongly retains the matrix and uses its existing
pattern and numeric storage without copying it.

```python
import taichi_forge as ti

ti.init(arch=ti.cuda)

operator = ti.linalg.aslinearoperator(
    A,
    traits=ti.linalg.OperatorTraits.spd(),
)

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
result = plan.solve(rhs)

if not result.converged:
    raise RuntimeError(result.termination_reason)
x = result.solution
```

Mutable Eigen CSR/CSC matrices remain supported by the established
`SparseCG`, `SparseMINRES`, `SparseBiCGSTAB`, and `SparseSolver` APIs. The
stored provider accepts fixed CSR/BSR only so that topology,
numeric generations, and provider ownership have one stable contract.

## Compiled kernel provider

`LinearOperator.from_kernel()` accepts an exact f32 ndarray ABI. With separate
topology and numeric data, the signature is:

```python
@ti.kernel
def apply_diagonal(
    active_size: ti.i32,
    topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
    x: ti.types.ndarray(dtype=ti.f32, ndim=1),
    y: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    for i in range(active_size):
        y[i] = numeric[i] * x[topology[i]]

operator = ti.linalg.LinearOperator.from_kernel(
    apply_diagonal,
    size,
    topology,
    numeric=numeric,
    traits=ti.linalg.OperatorTraits.spd(),
)
```

`size` may also be `(range_extent, domain_extent)`. A rectangular provider
must register an independent adjoint kernel through `adjoint=` before
`operator.adjoint()` is available:

```python
operator = ti.linalg.LinearOperator.from_kernel(
    forward_kernel,
    (rows, columns),
    topology,
    adjoint=adjoint_kernel,
    numeric=values,
)
```

The forward `active_size` is the range extent and the adjoint `active_size` is
the domain extent. Put dimensions required by both actions in an explicit
topology resource. A missing adjoint fails explicitly; the implementation does
not use autodiff, host materialization, or a self-adjoint trait as a fallback.

Without `numeric=`, the exact signature is `(active_size, operator_data, x,
y)`. All data arguments are scalar ndarrays; topology and numeric inputs may
have their own scalar dtypes but vectors are f32. The kernel must overwrite
every output entry and must not depend on an SNode tree. Construction compiles
one specialization and copies topology/numeric inputs into operator-owned
snapshots.

## Recordable Graph action

A compiled-kernel provider or a qualified compiled-Graph provider can expose
its apply operation as a recordable Graph action:

```python
input_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
)
output_arg = ti.graph.Arg(
    ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
)

builder = ti.graph.GraphBuilder()
builder.append_native(operator.graph_action(input_arg, output_arg))
graph = builder.compile()
graph.run({"input": x, "output": y})
```

`operator.graph_action()` is an f32, zero-copy recording boundary. It may be
appended at the Graph root or to a `Sequential` body used by structured
`while`, `if`, or `switch` control. Provider-owned topology, numeric, and
workspace snapshots become fixed bindings of the compiled Graph; they are not
copied a second time and do not appear in `Graph.run()` arguments. Explicit
root-dense Field state remains bound to its original storage. The surrounding
Graph can record every provider dispatch in order together with adjacent
kernels, so an iterative body does not call Python once per operator
application.

The symbolic input/output arguments use the one-dimensional scalar vector ABI.
At runtime they accept a matching scalar ndarray or a dense storage object that
the Graph runtime can scalar-linearize without copying. Input and output must
be proven disjoint. `adjoint=True` requires an explicitly registered adjoint
action. A compatible values-only update is rebound at launch without rebuilding
the Graph. The submission pins the exact immutable numeric generation it uses,
so an in-flight launch remains valid while a newer generation is published.
Topology, schema, state-tree, or runtime-generation changes still fail closed
and require a new Graph.

This recordable contract applies to compiled-kernel providers,
direct-dispatch compiled-Graph providers, fixed-layout `block_diagonal()`
containers, and f32 scale/shift/sum/compose/adjoint trees whose leaves are
recordable. Composition is lowered recursively while preserving child dispatch
order. A block container derives consecutive domain/range subviews from its
two public vector bindings at submission; these private views do not become
caller arguments, do not stage or allocate storage, and remain valid when the
container is nested inside another recordable composition. Ordered scale/sum
subtrees are normalized to weighted leaves: a two-leaf weighted sum uses the
two provider actions plus one in-place `axpby`, and larger sums reuse one
scratch vector. `compose` keeps its operator boundary and ordered intermediate.
Paths that need scratch allocate typed f32 temporary vectors from the
Graph-owned bounded arena, so scratch does not become a public runtime argument
and concurrent submissions receive independent arena lanes. The memory report
exposes the planned and persistent temporary bytes.
Both the generic rectangular or explicit-adjoint form and the legacy square
forward-only form preserve the compiled Graph's ordered multi-dispatch
sequence. `adjoint=True` records the explicit adjoint Graph; the legacy square
form does not infer one. Compiled-Graph providers containing indirect
dispatch, stored sparse providers, and other unsupported providers fail
explicitly instead of materializing an operator or inserting a hidden apply
fallback. Block-diagonal recording is limited to
fixed, consecutive layouts with recordable leaves; permutation, overlap, and
non-affine views remain unsupported. The provider recording protocol itself is
not a public custom-native callback API.

Recording one action does not by itself guarantee a speedup. The main use is
composing multiple operator actions and adjacent kernels into a larger Graph
region so that fixed submission costs can be amortized. Applications should
measure the complete composed workload rather than assume that wrapping one
apply is faster.

## Compiled Graph provider

`LinearOperator.from_graph()` binds a compiled Graph whose dynamic vector
arguments are named `input` and `output`. Every other Graph argument must be
assigned exactly one role:

```python
operator = ti.linalg.LinearOperator.from_graph(
    graph,
    size,
    fixed_i32={"active_size": size},
    topology={"row_offsets": row_offsets, "columns": columns},
    numeric={"values": values},
    workspace={"temporary": temporary},
    # Supply one live root-dense representative for every dependent
    # SNodeTree. Keys are labels, not per-Field access declarations.
    state={"coefficients": coefficients},
    traits=ti.linalg.OperatorTraits.spd(),
)
```

The provider is f32. `size` may be an integer square shorthand or `(range,
domain)`, and `adjoint=adjoint_graph` may register an independent adjoint Graph
with the same resource-role schema. It requires at least one topology ndarray.
Topology, numeric data, and workspace are copied into operator-owned
resources.

`state=` is the explicit exception to that snapshot policy. For every distinct
SNodeTree in the forward Graph's dependency set, the mapping must provide at
least one representative live root-dense scalar, Vector, or Matrix Field from
that tree. If an adjoint Graph is supplied, it must expose the same SNodeTree
dependency set and is validated against the same declaration. Forge retains
the tree's storage identity and binds it in place without a device copy, so
its contents may be updated between applications while the layout and
SNodeTree generation remain fixed. The mapping keys are nonempty diagnostic
labels; Field anchors and components are not matched individually. Dependency
comparison includes tree id, generation, and layout fingerprint, while
lifetime ownership and outer resource effects remain tree-granular. Multiple
representatives from the same tree do not add semantics.

Every dependent tree must be purely dense. A tree containing any sparse or
dynamic descendant is rejected conservatively even when the provider Graph
only accesses a dense sibling. Noncanonical Field storage or a missing or
extra dependency tree also fails during construction.

CPU lowers the Graph to an explicit sequence; CUDA and Vulkan use the
compiled-Graph execution contract. Backend capture/replay may use an ordinary
Graph fallback when the documented Graph runtime rules require it, without
changing the mathematical provider. Vulkan Graph replay requires at least two
dispatches. A one-dispatch Graph still executes correctly, while
`operator.statistics()` reports `ordinary_graph_fallback`.

Both generic and legacy compiled-Graph providers can export their ordered
forward dispatches through `graph_action()`. The generic form also exports an
explicitly registered adjoint Graph. A compatible numeric generation is
rebound into the already compiled outer Graph at launch. The complete leaf set
is validated before any binding is changed, and the resulting submission pins
one coherent generation snapshot. Destroying or replacing a declared state
SNodeTree, changing topology/schema, using the action after `ti.reset()`, or
otherwise changing the owning runtime still fails closed; reusing a numeric
tree id does not revive the old action.

## Mathematical traits

`OperatorTraits` uses `None` for unknown and `bool` for an explicit caller
claim:

```python
traits = ti.linalg.OperatorTraits(
    self_adjoint=True,
    positive_definite=True,
    positive_semidefinite=True,
    singular=False,
)
```

`OperatorTraits.spd()` is the equivalent convenience constructor. CG and PCG
require trusted `self_adjoint=True` and `positive_definite=True` claims and
reject an operator declared singular. Shape alone, a positive diagonal sample,
or an empirical product check does not establish these properties.

Traits are contracts for every numeric generation used through the operator.
When `update_numeric()` changes coefficients, the caller must ensure that the
declared properties remain valid. Structurally safe compositions derive the
traits they can prove; unsupported inferences remain unknown.

## Apply and composition

```python
y = operator.apply(x)
operator.apply(x, out=y)
y = operator.apply(x, alpha=2.0, beta=-0.5, addend=z)
y = operator @ x

B = 2.0 * operator
C = operator + B
D = operator.compose(B)       # operator(B(x))
E = operator.adjoint()
F = ti.linalg.block_diagonal((operator, B))
I = ti.linalg.identity(size, dtype=ti.f32)
S = operator.shifted(0.01)    # operator(x) + 0.01 * x
P = operator.parameterized_affine(
    alpha=1.0,
    beta=0.01,
    alpha_range=(1.0, 1.0),
    beta_range=(0.0, 0.1),
)
P.update_parameters(alpha=1.0, beta=0.02, expected_version=1)

# Flat row-major inverse 1x1, 2x2, 3x3, or 4x4 blocks.
M = ti.linalg.inverse_block_diagonal(
    inverse_blocks, block_size=3, assume_spd=True
)

inverse = ti.linalg.SmallBlockInverseBuilder(3, block_count).build(blocks)
M = ti.linalg.inverse_block_diagonal(
    inverse.inverse_blocks, block_size=3, assume_spd=True
)
```

The generalized form is `out = alpha * A(x) + beta * addend`. Input/output
aliasing is always rejected. `addend` may alias `out` for in-place
accumulation. When `beta == 0`, `addend` is neither validated nor read.
Generalized coefficient lowering is available on CPU and for f32 ndarray
operands on CUDA/Vulkan. GPU lowering uses device transform/scaled-add
primitives and performs no host readback. The non-aliasing path allocates no
N-sized scratch. If `addend is out`, one persistent scratch preserves the old
addend before `A(x)` is evaluated. GPU f64 and unsupported storage layouts fail
explicitly without a host fallback.

`adjoint()` is available only when the
provider exposes explicit adjoint application; no self-adjointness assumption
is used as an implementation fallback.

Scale, shift, sum, and compose execute on CPU and, for f32 Program-bound
operands, on CUDA and Vulkan. A recordable shifted operator lowers to the base
provider action followed by one in-place `axpby`; it does not launch a second
identity provider or allocate an identity-sized temporary. Standalone
sum/compose retain a private persistent ndarray workspace; their recordable
form instead uses Graph-owned typed temporaries. Equal-extent pure compose
chains are flattened in their exact leaf/adjoint order and alternate between
the destination and one scratch vector. Their Graph temporary therefore stays
at one vector from depth 2 onward instead of growing with syntax-tree depth;
rectangular or mixed-extent chains retain the conservative nested lowering.
Compact Field operands bind directly when the composition is embedded with
`graph_action()`; standalone composition retains the documented reusable
boundary-staging behavior.
`parameterized_affine()` publishes alpha and beta as one immutable generation.
The mandatory closed coefficient ranges are also the trait-proof boundary:
SPD/PSD properties are retained only when they hold over the complete ranges.
For an f32 operator, coefficients and range endpoints are first canonicalized
to the values that the f32 kernels actually execute. An out-of-range finite
Python value fails instead of becoming infinity, and a positive bound that
rounds to zero cannot incorrectly preserve an SPD claim.
An update uses optimistic `expected_version` validation, and cached Graph
actions rebind the new two-scalar snapshot without rebuilding; in-flight
submissions keep the exact old snapshot pinned. The snapshot is owned by the
native affine operator: updating it does not allocate or upload a dummy device
parameter array.

Public `identity()` remains CPU-only. General `block_diagonal()` additionally
supports f32 CUDA/Vulkan standalone apply when every leaf exposes qualified
direct affine dense storage. Leaves bind consecutive domain/range subviews and
run in order without gather, scatter, full-vector staging, or an N-sized
temporary. When every leaf is recordable, the same fixed-layout container is a
public Graph action and may be used directly as a SolvePlan operator or
fixed-linear preconditioner. Its private subviews are derived once per outer
submission, so a multi-iteration SolvePlan amortizes that binding work while
keeping one ticket and zero block-sized scratch. This is dispatch composition,
not kernel fusion. Permuted, overlapping, non-affine, or non-recordable layouts
fail explicitly. GPU f64 composition also remains unsupported.

`inverse_block_diagonal()` is the recordable fixed-linear preconditioner
helper for common physics layouts. The caller supplies already inverted,
row-major f32 blocks and must state `assume_spd=True`; Forge deliberately does
not read them back, invert them, regularize them, or infer SPD. Sizes 1 through
4 are supported on CPU, CUDA, and Vulkan. Each size has a statically specialized
kernel and derives row/block offsets from the scalar index; only one constant-
size topology word is snapshotted instead of an offset table proportional to
the vector length. Compatible values-only updates copy only inverse values and
use the ordinary immutable-generation rebind contract, which lets the same
single-system or batched PCG Graph consume a refreshed preconditioner safely.

`SmallBlockInverseBuilder` constructs those f32 row-major inverse blocks on
the device for fixed block sizes 1 through 4. It provides direct `build()` and
one-dispatch `graph_action()` forms. Partial-pivot Gauss-Jordan
elimination applies the requested nonnegative diagonal regularization and
writes a scale-relative pivot decision: `pivot_tolerance` is multiplied by the
largest absolute regularized coefficient in the block. This keeps the status
decision invariant when a block is uniformly rescaled. Controls that cannot be
represented by the f32 kernel fail at construction. The builder writes one
device-resident status per block: 0 for success, 1 for non-finite
input, and 2 for a singular or ill-conditioned pivot. A failed block is zeroed.
Forge does not read back status or infer SPD; callers must inspect status when
their policy requires it and make the independent `assume_spd` assertion.

### Fixed-topology implicit reference

[`implicit_linear_operator.py`](../../python/taichi_forge/examples/simulation/implicit_linear_operator.py)
is a headless CPU/CUDA/Vulkan reference for a time-varying implicit physics
solve. It keeps spring connectivity fixed, publishes new A/M numeric
generations, assembles the changing 2x2 system blocks and their inverses on
device, explicitly approves the rebuilt inverse-block action through
`PreconditionerPlan.update()`, and reuses one PCG `SolvePlan`:

```bash
python -m taichi_forge.examples.simulation.implicit_linear_operator \
  --arch cuda --nodes 2304 --steps 120 --telemetry
```

The example deliberately uses only public primitives. CUDA/Vulkan submit the
complete solve through one cached Graph ticket; compact Vector fields are
handled inside that recorded action without separate pack/unpack submissions.
CPU uses the documented synchronous completed-ticket path. `result()` is the
single terminal materialization point, and the reported iteration count is the
logical early-stop position. Vulkan may additionally encode a bounded masked
tail, visible through opt-in telemetry; it is not reported as useful solver
work. Per-block inverse status remains device-resident throughout stepping and
is read only once by the example's final qualification report.

## SolvePlan and SolveResult

`SolvePlan` retains its operator, solver state, and persistent workspace across
calls. Supported methods are:

- `method="cg"`: identity-preconditioned conjugate gradient for SPD systems;
- `method="pcg"`: PCG with `"jacobi"` on fixed CSR,
  `"block_jacobi"` on fixed BSR, or a trusted SPD `LinearOperator` or
  `PreconditionerPlan`
  that applies a fixed-linear approximate inverse; and
- `method="minres"`: identity- or SPD-preconditioned MINRES for square,
  self-adjoint systems that may be indefinite; and
- `method="bicgstab"`: identity- or fixed-linear right-preconditioned
  BiCGSTAB for general square systems; and
- `method="gmres"`: restarted, identity- or fixed-linear
  right-preconditioned GMRES for general square systems; and
- `method="fgmres"`: restarted flexible GMRES with a finite
  variable-linear right-preconditioner action table.

```python
result = plan.solve(rhs, initial_guess=x0, out=x)
print(result.iterations, result.residual_norm, result.termination_reason)
stats = plan.statistics()
```

On CUDA/Vulkan, an f32 CG/PCG plan with recordable A/M providers and the
qualified `device_convergent` policy can own a cached one-action Graph and
submit the complete solve asynchronously:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="pcg",
    preconditioner=preconditioner,
    submission_workspace_lanes=2,
    submission_workspace_saturation="raise",
)
submission = plan.submit(rhs, out=x, telemetry=True)
# no terminal packet readback has occurred
submission.wait()                 # backend completion only
result = submission.result()      # one explicit terminal materialization
print(submission.workspace_lane, result.iterations)
```

On CUDA/Vulkan, `SolvePlanSubmission` retains the plan, runtime operands,
output, terminal packet, and cached Graph through completion. `done()` and
`wait()` do not read the terminal packet; `result()` reads it once and caches
the immutable `SolveResult`. `telemetry()` exposes the underlying single Graph
submission.
The no-initial-guess and explicit-initial-guess forms are cached separately;
each materialized variant owns its own lane pool. `submission_statistics()`
reports those variants, lane use, persistent/transient bytes, submissions,
failures, telemetry requests, and terminal materializations.

The GPU convenience path adds no solver or backend primitive: it is exactly
`graph_action()` inside a cached Graph followed by `Graph.submit()`. Therefore
its qualification and failure boundary is the same recordable f32 CG/PCG
device-convergent contract. `submission_workspace_lanes=N` gives each in-flight
GPU lane independent Krylov storage; extra lanes increase persistent memory
linearly and do not promise physical kernel overlap. CPU instead executes the
existing exact native `solve()` synchronously and returns an already-completed
`SolvePlanSubmission`: it uses lane 0, has no device terminal packet or Graph
telemetry, and avoids replaying a GPU-style masked capacity.

### Complete SolvePlan Graph action

An f32 CG/PCG plan whose operator and optional fixed-linear preconditioner are
recordable can inline the complete solve into an enclosing Graph. This is a
structured action, not a nested `Graph.run()` call:

```python
rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
x_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)

solve = plan.graph_action(rhs_arg, x_arg, name="inner_pcg")
builder = ti.graph.GraphBuilder()
builder.append_native(solve)
graph = builder.compile(workspace_lanes=2, workspace_saturation="raise")

terminal = solve.allocate_terminal()
ticket = graph.submit({"rhs": rhs, "x": x, **terminal.arguments})
ticket.wait()
result = terminal.snapshot()
print(ticket.workspace_lane)
```

`solve.terminal.state` is an i32[4] symbolic resource containing status,
logical iteration count, breakdown code, and a completion marker.
`solve.terminal.metrics` is an f32[4] resource containing initial residual
squared, final residual squared, reference norm squared, and effective
tolerance squared. An enclosing structured region may consume these symbolic
resources on device. Forge performs no implicit terminal readback;
`terminal.snapshot()` is the explicit host boundary and should be called only
after the enclosing `SubmissionTicket` completes.

`solve.statistics()` attributes execution at these same boundaries without
instrumenting the Krylov loop. `enclosing_graph_submissions` advances once per
successful outer Graph execution containing this action. `observed_completions`
advances when `done()`/`wait()` observes its ticket (or immediately for an
already-complete submission), without reading terminal storage.
`terminal_snapshots` and `terminal_iteration_sum` advance exactly once per
packet when `terminal.snapshot()` performs the explicit readback. For an action
inside an outer loop, the packet describes the last invocation in that packet;
ticket region telemetry remains the source for outer-region invocation and
backend replay counts. `plan.graph_action_statistics()` aggregates all actions
created from the plan, while `solve.statistics()` returns one action record.

The action may be appended to an outer `Sequential` together with up to seven
other ordered inner `while` actions. The qualified depth-two shape is one
outer `while` whose body contains one to eight leaf inner `while` regions,
with ordinary dispatch/native actions between them. Inner control resources
must be disjoint and the complete encoded hierarchy is capped at 4,096
actions. CPU executes exact nested host control. Qualified CUDA Driver API
12.4+ runtimes use device-updatable kernel-node groups after an explicit setup
probe; older or unqualified runtimes use Forge's version-independent bounded
double-gate Graph. Vulkan uses bounded conditional replay. CUDA and Vulkan
keep the complete hierarchy under one ticket with no intermediate host
readback. An outer suffix kernel may read each solve's terminal state and
store its iteration count in a device trace before advancing the outer
counter. These GPU routes retain bounded static topology and do not claim
exact dynamic command termination.

All Krylov vectors and scalar recurrence state are private, address-stable
storage owned by a compiled Graph workspace lane. The default is one lane and
retains the fail-safe completion-fence serialization. `workspace_lanes=N`
enables lazy materialization of up to `N` independent storage sets; automatic
round-robin selection prefers a completed lane, while `workspace_lane=i` pins
a submission. `workspace_saturation="wait"` waits when every lane is busy and
`"raise"` fails immediately. Each extra materialized lane has a linear
persistent-memory cost, visible together with acquisitions, waits, saturation
errors, and busy/materialized counts in `Graph.execution_stats().memory`.

Workspace lanes remove storage-reuse waits between queued solves; they do not
create backend streams or promise overlapping GPU execution. Compile
independent Graphs when separate submission streams or concurrent host setup
are required. Runtime operands accept qualified scalar ndarrays, full dense
Fields, and compact scalar-flat
`vector_view(..., offset=..., length=..., stride=1)` ranges. RHS and output
must be proven disjoint; a separate initial guess may either be disjoint from
the output or be its exact view.

For the coefficient-invariant compatibility path, a fixed-linear
preconditioner may be passed as an operator rather than as an application
callback:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="pcg",
    preconditioner=inverse_operator,
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
)
```

`inverse_operator` maps the range of `operator` back to its domain and must
have the same dtype. It must carry trusted self-adjoint, positive-definite,
and nonsingular traits. CPU accepts any provider combination supported by
the operator execution plan. CUDA and Vulkan accept the recordable
compiled-kernel, compiled-Graph, or qualified composition combinations listed
in the support matrix below. Their topology
and numeric generations are pinned together for each solve; there is no
host callback or backend fallback.

### MINRES

`method="minres"` consumes the same `LinearOperator` and lifecycle contracts
as CG/PCG, but requires a trusted `self_adjoint=True` trait rather than
positive definiteness:

```python
operator = ti.linalg.LinearOperator.from_sparse_matrix(
    A,
    traits=ti.linalg.OperatorTraits(
        self_adjoint=True,
        singular=False,
    ),
)
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="minres",
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

CPU supports identity-preconditioned `f32/f64` MINRES for every compatible
CPU operator provider. CUDA and Vulkan support `f32` identity-preconditioned
MINRES for fixed CSR/BSR and compiled providers. On CUDA and Vulkan, a fixed
CSR may select `"jacobi"`, a fixed BSR with block size 2, 3, 6, or 12 may
select `"block_jacobi"`, and a trusted device-native fixed-linear
`LinearOperator` or `PreconditionerPlan` may be supplied directly. A MINRES
preconditioner must be self-adjoint, positive-definite, and nonsingular;
applications remain responsible for ensuring that a selected scalar Jacobi
inverse satisfies that mathematical contract.

MINRES rejects an operator declared `singular=True`. It does not implement
MINRES-QLP or minimum-length semantics for compatible singular systems. Both
halves of an explicitly stored symmetric matrix must be present and
consistent. The terminal status is qualified with the true residual of the
original system, including when preconditioning is active.

A CUDA/Vulkan MINRES plan owns nine persistent length-`n` `f32` vectors and
144 bytes of persistent scalar state. These figures exclude caller-owned
operator values, preconditioner resources, RHS/output arrays, backend handles,
and native replay objects. Inspect `statistics()` for the exact plan/provider
telemetry of a concrete configuration.

### BiCGSTAB

`method="bicgstab"` is the fixed-memory Krylov route for square nonsymmetric
operators. It accepts identity preconditioning or a fixed-linear
`LinearOperator`/`PreconditionerPlan` applied on the right:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="bicgstab",
    preconditioner=inverse_operator,
    max_iterations=300,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs)
```

The preconditioner maps the operator range back to its domain, has the same
dtype, is fixed-linear, and must not be declared singular. Unlike PCG and
MINRES preconditioners, it is not required to be self-adjoint or positive
definite. Right preconditioning keeps terminal qualification in the original
system: convergence is accepted only after evaluating the true residual
`b - A x`.

CPU supports `f32/f64` host-action providers. CUDA and Vulkan support `f32`
fixed CSR/BSR and compiled kernel/Graph providers. Compiled A/M providers use
direct native submissions. Identity-preconditioned fixed stored A providers
can reuse CUDA Graph or Vulkan command-sequence iteration chunks. No provider
is copied to the host to satisfy another backend path.

A device identity plan owns six persistent length-`n` vectors; right
preconditioning adds two vectors for the preconditioned directions. Both
configurations own 112 bytes of scalar state. `statistics()` reports exact
A/M applications, dot products, vector updates, logical/executed/wasted
iterations, host observations, replay activity, workspace bytes, and
`preconditioning_side`. `SolveResult.breakdown_reason` distinguishes
`nonfinite`, `rho`, `alpha_denominator`, `omega_denominator`, and `omega`
failures from ordinary maximum-iteration termination.

BiCGSTAB can stagnate or break down on a nonsingular problem. It is a
low-storage general-system option, not a stability substitute for a qualified
GMRES-family method.

### Restarted GMRES

`method="gmres"` provides a bounded-memory Krylov route for general square
operators when BiCGSTAB's short recurrence is not sufficiently robust.
`restart` is a plan-build parameter and must be `8`, `16`, or `32`; the
default is `16`.

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="gmres",
    preconditioner=inverse_operator,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

The implementation uses two-pass classical Gram-Schmidt (CGS2) on every
Arnoldi step. Orthogonalization uses a multi-dot reduction followed by a
fused projection, rather than issuing one host-observed dot product for every
basis vector. A cycle applies Givens rotations on device and qualifies its
terminal result with the original-system true residual `b - A x`.
`happy_breakdown` is counted separately from
`arnoldi_breakdown`, `orthogonalization_failure`,
`hessenberg_singular`, and `nonfinite` failures.

CPU supports compatible `f32/f64` host-action providers. CUDA and Vulkan
support `f32` fixed CSR/BSR and compiled kernel/Graph providers. An optional
preconditioner must be a nonsingular fixed-linear map from the operator range
back to its domain and is applied on the right. It need not carry the SPD
traits required by PCG or MINRES. `method="gmres"` intentionally accepts only
identity or fixed-linear preconditioning; use FGMRES for the supported
variable-linear schedule.

A plan preallocates a contiguous `(restart + 1) * n` basis. Device identity
plans own `restart + 5` persistent length-`n` vectors; right preconditioning
adds one vector. Hessenberg, Givens, least-squares, multi-dot partial, and
terminal state storage is also persistent, and warm solves allocate no
transient solver workspace. `statistics()` reports
`basis_vector_count`, `basis_reserved_bytes`,
`persistent_vector_reserved_bytes`, `persistent_scalar_reserved_bytes`,
exact A/M, dot, multi-dot, and vector-pass counts, restart cycles,
logical/executed/wasted iterations, and replay activity. Applications should
inspect these values before selecting a larger restart.

For fixed stored identity-preconditioned operators, CUDA Graph and Vulkan
command replay cover a complete restart cycle. Compiled providers and
right-preconditioned plans retain the same numerical implementation but use
direct native submission. CUDA supports `host_check_every_k` with
`check_interval == restart`; Vulkan supports that policy and
`fixed_budget_masked`. The complete cycle is submitted before the host
observes its terminal state, so up to `restart - 1` inactive tail steps can be
executed. Larger restarts may improve a difficult system's convergence but
also increase basis memory, cycle work, and its worst-case inactive tail.

### FGMRES with a variable-linear action table

`method="fgmres"` accepts a `PreconditionerPlan` whose
`behavior="variable_linear"`. The plan contains a finite table of 1 to 32
linear actions; this bounded table makes allocation, generation pinning, and
backend execution explicit without introducing a Python callback into an
Arnoldi step:

```python
schedule = ti.linalg.experimental.PreconditionerPlan(
    operator,
    (inverse0, inverse1, inverse2),
    method="external_multilevel_cycle",
    behavior="variable_linear",
    selection="cyclic",
).setup()

plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="fgmres",
    preconditioner=schedule,
    restart=16,
    max_iterations=160,
    atol=1e-8,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=16,
)
result = plan.solve(rhs)
```

The selected action for solve-global scheduled inner slot `k` is
`actions[k % len(actions)]`. A restart does not reset the schedule. CPU
scheduled and logical slots coincide; masked GPU execution may schedule
inactive tail slots after an earlier logical termination, so telemetry reports
selection and executed-iteration counts separately from logical iterations.
All target and action generations are pinned at solve entry. Every action must
belong to the same Program, have the solver dtype, map the operator range back
to its domain, and not be declared singular.

CPU supports compatible `f32/f64` host-action providers. CUDA and Vulkan
support `f32` compatible fixed stored and compiled kernel/Graph providers.
FGMRES uses the same CGS2 Arnoldi, true-residual termination, restart values,
and GPU observation policies as GMRES, but stores every preconditioned basis
vector in a persistent `Z` basis. The additional reservation is
`restart * n * sizeof(dtype)` bytes and is exposed as
`preconditioned_basis_vector_count` and
`preconditioned_basis_reserved_bytes`. For `restart=8`, the qualified CPU and
device plans own 20 and 22 persistent solver vectors respectively, excluding
operator, action, RHS/output, backend, and replay resources.

Variable-action FGMRES currently uses direct native submission. It does not
claim CUDA Graph or Vulkan command replay until every action in the table has
a qualified capture and binding contract; `solver_replay_unavailable_reason`
reports `variable_action_capture_contract_unavailable`. The API does not
support nonlinear preconditioners, Python iteration callbacks, automatic
restart selection, block GMRES, or domain-specific outer-solver policy.
Passing a variable-linear plan to CG, PCG, MINRES, BiCGSTAB, or ordinary GMRES
fails during plan construction.

### Current unsupported boundary

The `0.6.2` numerical-tooling contract intentionally does not provide:

- nonlinear, residual-dependent, adaptive, or Python-callback preconditioners;
- automatic restart selection, block or multi-RHS Krylov methods, recycling,
  deflation, pipelining, or communication-avoiding GMRES variants;
- MINRES-QLP, singular minimum-norm/minimum-length guarantees, or automatic
  nullspace handling;
- GPU `f64` GMRES-family execution, GPU f64/block-diagonal composition, or
  generalized GPU `alpha/beta/addend` apply;
- variable-action CUDA Graph/Vulkan command replay or single-system
  asynchronous solve submission; device-convergent execution outside the
  qualified CUDA and Vulkan scopes documented below;
- dynamic-topology solve plans, ragged batches, or transparent host fallback;
  and
- built-in IC/ILU/AMG, multigrid hierarchy construction, Schur/field splitting,
  domain decomposition, discretization, contact/KKT policy, or nonlinear outer
  iteration.

Independent fixed-size batched CG/PCG remains distinct from block Krylov or
multi-RHS solving. Unsupported backend/provider/policy combinations fail at
construction or capability validation; they are not approximated by changing
the provider, execution policy, or mathematical problem.

## PreconditionerPlan lifecycle

Use `PreconditionerPlan` when coefficients change or when provenance, explicit
reuse, and generation telemetry are required:

```python
preconditioner = ti.linalg.experimental.PreconditionerPlan(
    operator,
    inverse_operator,
    method="external_block_inverse",
).setup()

z = preconditioner.apply(r)
pinned = preconditioner.pin()

# Each provider publishes its next numeric generation first.
operator.update_numeric(next_a, expected_topology_version=1,
                        expected_numeric_version=3)
inverse_operator.update_numeric(next_m, expected_topology_version=1,
                                expected_numeric_version=7)
preconditioner.update()  # next_m was rebuilt from the current operator

pcg = ti.linalg.experimental.SolvePlan(
    operator, method="pcg", preconditioner=preconditioner
)
```

`built_from_operator_stamp` records the operator generation from which the
action was actually built. `accepted_target_stamp` records the generation the
action is currently approved to serve. A target update makes the plan stale by
default. If lagged preconditioning is valid, call
`preconditioner.update(accept_reuse=True)` while the action is unchanged. This
updates only compatibility and preserves provenance. A changed action requires
an ordinary `update()` rebuild attestation.

`pin()` retains the exact target and action generations together. The returned
`PreconditionerSession` can continue applying that old generation after later
generations are published. A fixed-linear session uses `apply(r, out=None)`;
a variable-linear session accepts `iteration=k` and selects the pinned cyclic
action for that scheduled slot; the default `iteration=0` selects the first
action. `metadata` exposes provenance and compatibility stamps.
`PreconditionerPlan.statistics()` reports setup, rebuild, reuse, stale
rejections, schedule update success/failure, and approved-generation
publish/retire/release counters. `SolvePlan.statistics()` additionally reports
action selections and schedule wraps. Setup and update execute at host
boundaries; session apply and solver iterations invoke native `OperatorAction`
objects without Python callbacks.

A setup fixed-linear plan whose approved action is recordable can be consumed
directly by recordable PCG and batched PCG. Graph recording does not bypass the
plan lifecycle: each launch atomically validates and pins the approved target
and action generations, and the asynchronous ticket owns that exact pair
through completion. A stale target, unapproved action generation, or
nonrecordable action fails explicitly. Variable-linear action tables remain a
direct native FGMRES facility and are not promoted to Graph replay.

For a variable-linear table, `update(accept_reuse=...)` accepts either one
boolean for all actions or one boolean per action. The complete next table is
validated before any generation is published: one stale or incompatible
action rejects the whole update. A solve that already pinned the previous
table retains that immutable snapshot. `fixed_linear` is supported by the
documented PCG, MINRES, BiCGSTAB, and GMRES consumers; `variable_linear` is
supported only by FGMRES. `nonlinear` remains descriptive and returns a
structured unsupported reason. Built-in Jacobi and block-Jacobi use the same
native setup/update/pin lifecycle and remain selected with
`preconditioner="jacobi"` or `"block_jacobi"`.

`rhs`, `initial_guess`, and `out` must match the operator dtype and scalar
extent and belong to the current runtime. If `out` is omitted, the plan creates
a result ndarray. If `initial_guess` is omitted, the result is initialized to
zero. RHS/output aliasing is rejected.

`SolveResult` contains the solution and a terminal snapshot: status code,
termination reason, convergence/breakdown/max-iteration flags, iteration
count, initial and final residual norms, both tolerances, relative reference
norm, and effective tolerance. CG, PCG, MINRES, BiCGSTAB, GMRES, and FGMRES use:

The result record is frozen; its `solution` ndarray remains caller-writable.

```text
||b - A x||_2 <= max(atol, rtol * ||b||_2)
```

`statistics()` exposes backend-neutral plan/provider/workspace counters; it is
diagnostic data, not part of the numerical result.

### GPU execution policy

`execution_policy` controls when a GPU solve observes convergence on the
host:

```python
plan = ti.linalg.experimental.SolvePlan(
    operator,
    method="cg",
    max_iterations=200,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=8,
)
```

- CPU defaults to `"host_each_iteration"`. Stored f32 CG/PCG also accepts
  explicit `"bounded_convergent"`, which preserves the same native loop.
- CUDA fixed stored f32 CSR/BSR CG/PCG defaults to
  `"bounded_convergent"`. In `bounded_mode="auto"`, a qualified runtime uses
  an exact device-side CUDA conditional Graph; an unavailable native path
  falls back to reusable Graph chunks with host checks. Explicit
  `"host_each_iteration"` remains available as an opt-out.
- Replay-qualified CUDA fixed stored f32 MINRES/BiCGSTAB defaults to
  `"host_check_every_k"` with `check_interval=4`.
- Qualified square matrix-free Kernel/Graph plans on CUDA also select
  `"host_check_every_k"` automatically: CG, Kernel-PCG, MINRES, and BiCGSTAB
  use K=4, while GMRES/FGMRES use `check_interval == restart`. Chunked
  execution keeps recurrence scalars on the device and reads one terminal
  snapshot per chunk. K=8 remains available for the non-GMRES methods, and
  explicit `"host_each_iteration"` remains available where supported. CUDA
  BiCGSTAB with a compiled Graph provider retains `"host_each_iteration"` by
  default because short converging solves did not show a stable K=4 benefit;
  callers can still opt in.
- CUDA compiled-kernel f32 CG/PCG may explicitly request
  `execution_policy="device_convergent"` when conditional Graph support is
  available. The generic structured Graph records the A action and, for PCG,
  the fixed-linear compiled-kernel M action, keeps recurrence control on the
  device, and reads one terminal packet per solve. Vector updates remain
  parallel range kernels; dot products use persistent two-stage shared-block
  reductions whose fixed partial buffers are owned by the plan. This
  combination is correctness-qualified but marked
  `qualification_scope="explicit_only"`; its automatic default remains the
  K=4 `"host_check_every_k"` path because Graph construction/capture and first
  execution must be amortized by repeated warm solves.
- A recordable compiled-Graph f32 PCG plan automatically selects
  `"device_convergent"` when both A and its fixed-linear M action qualify.
  The same provider keeps symbolic Krylov input/output bindings through every
  ordered dispatch, so a canonical compact Field or contiguous Field range
  does not add SolvePlan pack/unpack submissions. Recordable compiled-Graph CG
  also accepts an explicit device-convergent request, but retains its existing
  host-check default until that policy has a general latency advantage.
- A recordable f32 scale/sum/compose operator used by CUDA or Vulkan CG/PCG
  automatically selects `"device_convergent"`. This is the only qualified GPU
  solve policy for composed providers: no host-check path substitutes
  standalone applies or moves vectors between Field and ndarray storage inside
  Krylov. CUDA stops at the exact logical iteration. Vulkan reports the exact
  logical stop but may execute an inactive encoded tail. Each backend reads one
  terminal packet per solve; an unavailable structured-Graph path fails rather
  than selecting another policy.
- CUDA GMRES/FGMRES defaults to `"host_check_every_k"` and requires
  `check_interval == restart`. Stored identity-preconditioned GMRES records a
  reusable restart-cycle Graph; FGMRES and other non-recordable provider
  combinations preserve direct submission.
- Vulkan recordable compiled-kernel f32 CG/PCG and recordable compiled-Graph
  PCG default to `"device_convergent"`. A generic structured Graph records the
  A action and, for PCG, a fixed-linear recordable M action. Its compact device-control
  plan keeps recurrence state on the device and performs no per-iteration
  host observation. Budgets that exceed one encoded command plan are executed
  as bounded chunks of at most 64 iterations, with one terminal observation
  per chunk. Vulkan kernel-profiler or dispatch-cache modes disable command
  replay; automatic policy selection then uses `"host_check_every_k"`, while
  an explicit `"device_convergent"` request fails with a capability reason.
- Other replay-qualified Vulkan fixed stored or compiled f32
  CG/PCG/MINRES/BiCGSTAB plans default to `"host_check_every_k"` with K=4;
  stored identity-preconditioned GMRES uses the same default with
  `check_interval == restart`. Qualified matrix-free methods outside the
  recordable provider CG/PCG scope use K=4 or restart-sized
  host checks.
  Explicit `"fixed_budget_masked"` remains available for workloads that
  intentionally consume the full iteration budget. These policies support
  `atol`, `rtol`, and their combined effective tolerance.

For fixed stored f32 CSR/BSR, CUDA `host_check_every_k` and Vulkan
`host_check_every_k`/`fixed_budget_masked` record supported CG/PCG/MINRES and
identity-preconditioned BiCGSTAB iteration chunks and
identity-preconditioned GMRES restart cycles as reusable native execution
sequences. The recordable CG/PCG/MINRES combinations include identity, stored
Jacobi, and stored block-Jacobi preconditioners. The first compatible execution
builds a CUDA Graph or Vulkan command sequence; later executions with the same
topology, workspace, and output binding replay it. A values-only matrix update
does not re-record an identity-preconditioned GMRES sequence; supported
preconditioner refreshes retain the existing CG/PCG/MINRES sequence.
Replacing the output ndarray, changing topology or schema, or recreating the
runtime invalidates and safely rebuilds it.

For qualified CUDA stored CG/PCG, the conditional Graph owns the complete
recurrence loop and evaluates convergence on the device after every logical
iteration. A solve retains one initial and one terminal state observation but
does not perform an iteration-by-iteration host scalar reduction or implicit
synchronization. The Graph terminates at the exact logical iteration, so this
path has no masked tail work, and a persistent plan replays the cached
executable while its topology, workspace, and output binding remain stable.

For qualified CUDA/Vulkan recordable compiled-kernel, compiled-Graph, or
composition CG/PCG, the
structured Graph owns the complete recurrence region. Vulkan uses compact
device masking rather than dynamic command-stream termination, so logical
convergence is exact while an encoded block may retain an inactive tail. The
provider action, dense ndarray workspaces, predicate, counter, status, and
reduction buffers have fixed runtime bindings. Canonical compact full Fields
bind to the Graph preamble/epilogue described above; other supported Fields use
separate staging. Neither route moves Field/ndarray values inside a Krylov
iteration.

FGMRES action tables use direct native submission on both GPU backends. No
identity-GMRES replay path is silently reused for a variable action schedule.

Outside the recordable-provider CG/PCG device-convergent scope,
compiled-kernel and compiled Graph A/M providers use direct outer solver-chunk
submission. That outer recurrence boundary is independent of provider
execution: each compiled Graph apply uses its provider-owned compiled Graph
plan, while a compiled-kernel apply uses an ordinary compiled kernel launch.
Replay-qualified multi-dispatch Graphs record and replay on CUDA/Vulkan;
ineligible plans preserve ordinary execution and report the backend path. In
particular, single-dispatch Vulkan Graphs intentionally use the ordinary path
because recording them adds no useful consolidation. The solver does not nest
a compiled Graph provider inside another captured Graph. Providers are not
staged through the host or replaced to obtain replay.

`statistics()` exposes the outer boundary through `solver_chunk_builds`,
`solver_chunk_replays`,
`solver_chunk_direct_submissions`, `solver_chunk_rebinds`,
`solver_chunk_invalidations`, `solver_graph_enabled`, and
`solver_replay_unavailable_reason`. It independently reports provider execution
through `operator_execution_kind`, `operator_compiled_graph_submissions`,
`operator_backend_captures`, and `operator_backend_replays`. Build cost belongs
to cold execution, so qualification should report first-solve and warm-solve
timing separately.

A chunk or GMRES-family restart cycle always completes before its terminal state is
inspected. The reported
`SolveResult.iterations` is the logical convergence or breakdown iteration;
`statistics()["operations"]` separately reports `executed_iterations`,
`wasted_iterations`, host synchronization counts, and direct chunk
submissions. A chunked solve can therefore execute up to `K - 1` masked
or otherwise inactive tail iterations. Vulkan fixed-budget execution may
execute the full `max_iterations` while preserving an earlier logical
termination result.

For CG/PCG/MINRES/BiCGSTAB, use `K=4` when earlier termination is more
important than synchronization frequency and `K=8` when amortizing host
checks is more important. GMRES and FGMRES use their selected restart as the
observation interval. The faster choice depends on vector size, operator cost, iteration
count, driver, and backend. Unsupported policies and intervals fail during
plan construction; they do not silently fall back.

Use `benchmarks/linear_operator_graph_krylov_bench.py` to qualify the complete
f32 compiled-kernel CG boundary. It reports plan construction, first solve,
warm synchronous solve completion, terminal observation, true residual,
maximum solution error, logical/executed iterations, host observations, and
kernel-profiler visibility. Select one or more policies with `--policies`;
CUDA or Vulkan runs should be performed on an otherwise idle device. Kernel
time is reported as unavailable rather than inferred when a backend profiler
cannot see inside a captured Graph.

`plan.execution_capabilities()` reports the policy matrix and a structured
reason for unavailable conditional execution, together with the selected
`default_execution_policy`.
The `automatic_solver_batching` object reports matrix-free Kernel/Graph
host-check selection, the default interval, the direct-chunk backend primitive,
and whether provider execution uses a compiled Graph plan or compiled-kernel
launch. Batching does not require or claim outer solver replay.
The separate `automatic_solver_replay` object reports whether outer recurrence
replay was selected, whether the operator and preconditioner combination is
qualified, and the backend primitive (`cuda_conditional_graph_or_chunk_replay`,
`cuda_graph_chunk_replay`, `vulkan_structured_graph`, or
`vulkan_command_replay`). Post-solve statistics remain authoritative for the
actual replay or direct-submission path.
Direct `"device_convergent"` execution covers stored and recordable-provider
scopes. A single-system stored f32 CSR/BSR CG/PCG plan is
eligible for automatic
selection through `"bounded_convergent"` when the driver, conditional-Graph
entry points, device setter, provider capture, and cuBLAS workspace
requirements are satisfied; otherwise the documented chunk fallback is used.
A single-system compiled-kernel f32 CG/PCG plan may request the generic
structured-Graph path explicitly when its A/M actions are recordable, but is
not selected automatically on CUDA. A recordable compiled-Graph f32 PCG plan
and a recordable f32 composition are selected automatically on both GPU
backends; compiled-Graph CG remains explicit-only. On Vulkan, all qualified
recordable-provider CG/PCG scopes report
`primitive="vulkan_dispatch_indirect"`; unsupported providers fail without
changing policy or backend. The stored CUDA path requires its solver-specific
conditional setter and cuBLAS user-workspace support. The compiled-kernel CUDA
path instead requires the general Graph conditional setter and does not
require the cuBLAS workspace symbol. The Vulkan path requires the qualified
structured-Graph runtime and recordable fixed f32 bindings.
`device_convergent.qualification_scope` and
`automatic_selection_qualified` expose this distinction. Explicit requests
fail without fallback. Independent batched f32 CG/PCG additionally qualifies
an explicit device-convergent policy when every A/M action is recordable; CPU
and non-recordable providers do not claim this execution mode.

## Independent batched CG and PCG

`BatchedSolvePlan` solves a homogeneous batch of independent SPD systems with
one persistent plan. It uses one flat direct-sum operator with shape
`(B * N, B * N)` and partitions every vector into `B` contiguous systems of
length `N`:

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    method="cg",
    max_iterations=100,
    atol=1e-7,
    rtol=1e-5,
    execution_policy="host_check_every_k",
    check_interval=4,
)
result = plan.solve(rhs_flat, out=x_flat)

for env, reason in enumerate(result.termination_reasons):
    if not result.converged[env]:
        raise RuntimeError(f"system {env}: {reason}")
```

`independent_systems=True` is a required caller assertion. The system
operator and, for PCG, the fixed-linear preconditioner must preserve every
partition: output entries for environment `e` may depend only on input entries
from environment `e`. Global SPD traits do not prove this partition property.
Violating it gives an invalid independent-batch problem rather than a coupled
solve.

The first public layout is deliberately homogeneous and flat:

- the operator extent must be divisible by `batch_size`;
- all systems have the same scalar extent and f32 dtype;
- `rhs`, `initial_guess`, and `out` have shape `(B * N,)`;
- `atol` and `rtol` may be scalars or length-`B` sequences; and
- variable offsets and ragged systems are not part of this contract. Optional
  active-system compaction does not change the homogeneous flat layout.

`method="cg"` uses identity preconditioning. `method="pcg"` requires a
trusted SPD `LinearOperator` that applies a fixed-linear approximate inverse
over the same flat partitions. Fixed stored and compiled-kernel A/M providers
are qualified on CPU, CUDA, and Vulkan. Other provider kinds remain subject to
their ordinary backend capability and qualification boundaries; the batched
plan never stages through the host or changes provider.

Each environment owns its recurrence scalars, effective tolerance, status,
logical iteration count, and residual norms. One environment may converge,
break down, or reach `max_iterations` without changing another environment's
terminal result. `BatchedSolveResult` exposes these values as immutable tuples
and returns the flat solution ndarray separately. A batch of one follows the
same numerical contract as a single-system CG/PCG solve.

CPU uses `"host_each_iteration"`. CUDA and Vulkan default to
`"host_check_every_k"` with `K=4`; they also accept `K=8`, explicit
`"host_each_iteration"`, `"fixed_budget_masked"`, or explicit
`"device_convergent"`. The latter requires recordable f32 A/M actions and is
not selected automatically. A host-check chunk may issue inactive tail
iterations before observing that all environments have terminated. Recurrence
and vector-update kernels mask inactive environments, but the monolithic A/M
provider still applies to the full flat batch. Provider apply compaction is
therefore not implied by convergence masking.

For host-check execution, CUDA and Vulkan compile the stable iteration
recurrence into plan-owned Taichi Graphs and reuse those graphs across solves.
CG submits one recurrence Graph per iteration; PCG submits one segment after A
and another after M. A Vulkan fixed-budget transaction uses safe direct
recurrence dispatches because synchronizing a nested replay inside an active
submission batch is not a qualified operation.

The explicit device-convergent route instead records initialization, every A/M
provider action, reductions, per-system recurrence/status updates, and the
global active predicate in one structured Graph. `submit()` produces one Graph
ticket. CUDA uses its qualified structured control path; Vulkan uses bounded
conditional replay and may retain a documented encoded/masked tail. Neither
backend reads convergence through the host while the loop is active. The
plan-owned device counter records the actual stopping iteration and is read
only when terminal state is materialized. This route stops subsequent full
batch A/M payload once all systems are terminal, but it does not compact A/M
within a partially active batch.

CUDA `device_convergent` plans may explicitly set
`active_system_compaction=True`. This publishes a stable device-side list of
active systems and compacts only recurrence reductions and vector updates.
The A/M providers still apply to the complete flat batch, so the capability
reports `provider_apply_compacted=False` and scope
`recurrence_vector_payloads`. The compact route uses a capacity-grid masked
prefix rather than exact indirect launch, performs no host count readback, and
is neither available nor silently emulated on CPU or Vulkan. It remains off by
default because its benefit depends on convergence heterogeneity, provider
cost, and batch size; changing the parallel reduction order may also move a
borderline floating-point stop by one logical iteration without changing the
convergence contract. Inspect `statistics()["active_system_compaction"]`
instead of inferring support from the backend name.

All paths pin the exact operator and preconditioner generations they submit.
Compatible values-only updates are rebound into a cached device-convergent
Graph without reconstructing it; in-flight submissions keep the older
generation alive through completion. Replacing `out` patches the Graph binding
after the previous solve completes. Each workspace clone owns an independent
replay plan and therefore does not serialize through another clone's Graph
lock.

`statistics()` makes this distinction observable through executed system
iterations, provider system iterations, masked provider system iterations,
active efficiency, host checks, transfers, and persistent resource sizes. A CG
plan owns three length-`B * N` workspace vectors; PCG owns four. It also owns
per-environment recurrence, tolerance, and status state. Caller-owned RHS,
solution, initial guess, and provider resources are excluded from these plan
workspace counts. Batched-plan statistics use schema version 5 and report
`recurrence_replay_builds`, `recurrence_replay_graph_builds`,
`recurrence_replay_submissions`, `recurrence_replay_logical_kernels`, output
`recurrence_replay_rebinds`, and direct recurrence-kernel submissions. The
`recurrence_replay` record explicitly states that A/M provider applies are not
part of that host-check replay scope. `device_convergent_replay` separately
reports its whole-solve scope, Graph build/submission counts, logical stop
counter, and available structured-control telemetry. Encoded or masked counts
remain unavailable rather than inferred when a backend cannot expose them
without adding synchronization.

Every completed solve publishes one packed terminal packet containing the
global logical counter and all per-system status, iteration, residual, and
tolerance values. `result()` materializes that packet once. With
`submit(..., telemetry=True)`, `submission.telemetry()` additionally returns
immutable per-ticket logical/executed/provider work, active efficiency,
available encoded/masked counts, backend Graph launches, physical queue
submissions, and non-inferred timing fields. Telemetry is opt-in; unavailable
backend counters remain `None`.

Use `benchmarks/batched_graph_pcg_bench.py` to compare policies and
preconditioner quality with interleaved samples; it also checks steady runtime,
host-pool, and device-pool state after warmup. Use
`linear_operator_graph_rebind_bench.py` for generation churn and
`linear_operator_shifted_bench.py` for shifted-versus-explicit-identity
lowering. These are local qualification tools, not fixed performance promises.

### Asynchronous fixed-budget or device-convergent submission

CUDA and Vulkan batch plans using `execution_policy="fixed_budget_masked"` or
the qualified `"device_convergent"` policy can submit without materializing
terminal state at the call boundary:

```python
plan = ti.linalg.experimental.BatchedSolvePlan(
    operator,
    batch_size=B,
    independent_systems=True,
    max_iterations=8,
    atol=1e-6,
    execution_policy="device_convergent",
)

submission = plan.submit(rhs_flat, out=x_flat)
# Submit independent application work here.
submission.wait()
result = submission.result()
```

When multiple asynchronous solve producers share a GPU, create a lazy,
memory-accounted workspace pool and use one `SubmissionPacer` to bound
aggregate backlog and provide fair admission:

~~~python
pool = plan.workspace_pool(2, workspace_saturation="wait")
pacer = ti.graph.SubmissionPacer(
    2,
    max_in_flight_per_lane=1,
    max_queued=8,
)
primary_ticket = pool.submit(
    rhs_a, out=x_a, pacer=pacer, lane='primary_physics'
)
secondary_ticket = pool.submit(
    rhs_b, out=x_b, pacer=pacer, lane='secondary_physics'
)
primary_result = primary_ticket.result()
secondary_result = secondary_ticket.result()
~~~

The complete host launch sequence for one solve is submitted in one admission
turn. After launch, both tickets may remain incomplete within the
`max_in_flight` bound. Asynchrony here defines a host-completion boundary; it
does not guarantee concurrent GPU kernels, independent streams or queues, or
device preemption. `max_in_flight_per_lane` prevents a high-rate producer
from occupying every slot, while cross-lane round robin gives an existing
waiter bounded admission delay. A blocked caller polls all in-flight
completions with bounded adaptive backoff, so a later fast solve may release
capacity before an older slow solve completes. Only calls sharing the same
pacer are covered; engine frame deadlines, task dependencies, and cancellation
remain application policies.

`SolveSubmission.done()` observes backend completion without releasing the
workspace slot. `wait()` waits when necessary, materializes the complete
per-system terminal snapshot, and releases the slot. `result()` performs the
same operation when needed and returns the immutable `BatchedSolveResult`.
The submission retains the exact A/M generations, input/output ndarrays,
workspace, and backend completion until this boundary. Backend failures are
re-raised by `wait()` and `result()`; `ti.reset()` waits for retained backend
work and makes an outstanding ticket explicitly stale.

Publishing a new operator or preconditioner numeric generation before an old
submission completes retains the old generation through its completion. A high
update rate can therefore keep multiple complete value buffers resident. Include
update cadence in the memory budget, or complete the relevant tickets before
publishing. A pacer bounds participating invocation counts, not generation
bytes.

One `BatchedSolvePlan` owns one submission slot. Submitting again before the
pending ticket is completed and materialized fails instead of sharing Krylov
vectors. Use `clone = plan.clone_workspace()` when independent submissions
must be managed manually, or `plan.workspace_pool(lanes)` for lazy round-robin
selection. Each materialized pool lane owns another complete set of CG/PCG
workspace vectors, terminal state, and an independent Graph instance. A pool
may pin a submission with `workspace_lane=i`; `workspace_saturation="raise"`
fails immediately when all lanes are busy, while `"wait"` waits for and
materializes a completed lane. `submission.workspace_lane` reports the chosen
lane, and `pool.statistics()` reports capacity/materialized/pending lanes plus
per-lane, materialized, and capacity payload bytes. Workspace lanes remove
mutable-state and completion-fence aliasing; they do not promise distinct GPU
streams, queues, or physical overlap. Use one pool per root plan. Chunked
host-check policies and CPU plans are not qualified for `submit()`; a call
fails explicitly rather than moving a synchronous loop to a worker thread.

For batch size `B` and per-system size `N`, each f32 host-check CG plan or
clone has a logical workspace payload of `12 * B * N + 92 * B + 24` bytes;
PCG uses `16 * B * N + 92 * B + 24`. A device-convergent plan adds 12 bytes.
Enabling CUDA active-system compaction adds another `4 * B + 8` bytes. These
figures include the packed terminal packet. Inspect
`statistics()["resources"]["clone_workspace_payload_bytes"]` before creating
a pool; `pool.statistics()` is authoritative after lazy lane materialization.
Allocator and driver overhead, caller vectors, and operator or preconditioner
resources are excluded. Use `max_in_flight=1` by default for a
large solve. Increase it to 2 only when profiling demonstrates useful host
overlap with acceptable memory and tail latency. For small systems, increase
batching instead of creating one plan clone per application entity.

This API is independent batching. It is not global-scalar CG over an
implicitly coupled block matrix, multi-RHS CG, block CG, or another block
Krylov method. Coupled systems require an operator and solver whose mathematics
model that coupling explicitly.

## Support matrix

### Providers and apply

| Provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| Fixed stored CSR/BSR | `f32`, `f64` | `f32` | `f32` |
| Compiled kernel | `f32` | `f32` | `f32` |
| Compiled Graph | `f32` | `f32` | `f32` |
| Identity | `f32`, `f64` | Unsupported | Unsupported |
| Fixed-layout block diagonal | `f32`, `f64` | Qualified direct-affine leaves, `f32` | Qualified direct-affine leaves, `f32` |
| Caller-supplied inverse block diagonal, sizes 1-4 | `f32` | `f32` | `f32` |
| Scale/shift/sum/compose | `f32`, `f64` | `f32` | `f32` |

Kernel and Graph providers support rectangular shapes and explicit adjoints.
Generalized `alpha/beta` apply is available on CPU and for f32 ndarray
operands on CUDA/Vulkan. GPU f64 and non-ndarray generalized storage remain
unsupported.

### Solvers

| Method/provider | CPU | CUDA | Vulkan |
| --- | --- | --- | --- |
| CG, fixed stored | CSR/BSR, `f32/f64` | CSR, `f32` | CSR/BSR, `f32` |
| CG, compiled kernel/Graph | `f32` | `f32` | `f32` |
| CG, recordable composition | `f32/f64` | `f32`, device-convergent | `f32`, device-convergent |
| PCG + Jacobi | Fixed CSR, `f32/f64` | Fixed CSR, `f32` | Fixed CSR, `f32` |
| PCG + block-Jacobi | Fixed BSR, `f32/f64` | Fixed BSR, `f32` | Fixed BSR, `f32` |
| PCG + fixed-linear operator/plan | Supported providers, `f32/f64` | Recordable compiled-kernel/Graph/composition A/M, `f32`; Graph/composition device-convergent automatically | Recordable compiled-kernel/Graph/composition A/M, `f32`; Graph/composition device-convergent automatically |
| MINRES + identity | Supported providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| MINRES + Jacobi/block-Jacobi | Unsupported | Fixed CSR/BSR respectively, `f32` | Fixed CSR/BSR respectively, `f32` |
| MINRES + fixed-linear operator/plan | Unsupported | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| Independent batched CG/PCG | Fixed stored or compiled-kernel A/M, `f32` | Fixed stored or recordable A/M, `f32` | Fixed stored or recordable A/M, `f32` |
| Batched asynchronous submission | Unsupported | Fixed-budget stored/compiled-kernel or device-convergent recordable A/M, `f32` | Fixed-budget stored/compiled-kernel or device-convergent recordable A/M, `f32` |
| Device-convergent conditional execution | Unsupported | Single-system scopes listed above; independent batched recordable A/M explicit-only | Single-system scopes listed above; independent batched recordable A/M explicit-only |
| BiCGSTAB + identity | Supported host-action providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| BiCGSTAB + fixed-linear right preconditioner | Supported host-action providers, `f32/f64` | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| GMRES + identity | Supported host-action providers, `f32/f64` | Fixed CSR/BSR or compiled provider, `f32` | Fixed CSR/BSR or compiled provider, `f32` |
| GMRES + fixed-linear right preconditioner | Supported host-action providers, `f32/f64` | Compatible device-native A/M, `f32` | Compatible device-native A/M, `f32` |
| FGMRES + variable-linear action table | Supported host-action providers, `f32/f64` | Compatible device-native A/actions, `f32` | Compatible device-native A/actions, `f32` |

Direct factorization remains a stored-matrix API.

## Numeric updates and ownership

Stored operators use `operator.update_numeric(values)`. Compiled providers use
optimistic version checks:

```python
operator.update_numeric(
    next_values,
    expected_topology_version=1,
    expected_numeric_version=3,
)
```

Graph updates pass a complete mapping of numeric roles. A successful compiled
update publishes the next immutable numeric generation. In-flight work keeps
its pinned generation alive; later apply/solve calls observe the new
generation. A stored Jacobi/block-Jacobi PCG plan refreshes its numeric
preconditioner before the next solve while retaining its pattern and Krylov
workspace. Scalar CSR Jacobi stores diagonal reciprocals. BSR block-Jacobi
stores a lower Cholesky factor for every diagonal block; supported block sizes
are 2, 3, 6, and 12. Every block must be finite, symmetric, and positive
definite. Invalid blocks fail explicitly without symmetrization,
regularization, or fallback, and a failed refresh does not publish a new
preconditioner generation. Topology changes require constructing a new
operator.

CUDA and Vulkan perform value-only Jacobi and block-Jacobi refreshes on the
device. A successful refresh preserves the committed resource address, so a
replayable solve plan may rebind the numeric generation without rebuilding its
recurrence program. Warm refreshes do not copy the complete values or factors
through the host and do not allocate device memory; validation reads back only
a fixed-size status. `statistics()` exposes the refresh contract, operation
counters, transfer bytes, and device-allocation count.

The public API has no borrowed-resource mode. Stored operators strongly retain
their matrix; compiled providers own copied topology/numeric/workspace
resources; compositions and solve plans strongly retain their operands.

## Relationship to the legacy matrix-free API

The callback-based `ti.linalg.FieldLinearOperator`, `MatrixFreeCG`, and
`MatrixFreeBICGSTAB` retain their field-shaped vector ABI. They use a `(x, y)`
kernel callback and are not implicitly adapted because that ABI does not carry
explicit topology, numeric-resource, runtime-generation, or capability
information.

Use `FieldLinearOperator` when an application intentionally needs the legacy
field callback contract. Use `LinearOperator` when provider capabilities,
resource generations, runtime storage views, composition, or solver-plan
integration are required.

## Qualification boundary

`qualify_operator()` generates versioned, JSON-serializable protocol evidence
for any public `LinearOperator`:

```python
report = ti.linalg.qualify_operator(
    operator,
    reference=dense_reference,
    samples=4,
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
report.to_json()
matrix = ti.linalg.summarize_operator_qualifications([report])
```

The report contains backend/build identity, provider, shape, capabilities,
resource stamps, forward/adjoint oracle errors, linearity and dot-product
identity, generalized-apply `beta=0` no-read status, synchronous boundary
timing, and native counters. `summarize_operator_qualifications()` builds a
deterministic support matrix from detached backend/provider reports. An
unsupported GPU generalized coefficient is recorded as `unsupported`; it is
neither reported as a pass nor executed by a host fallback. Timing describes
the current machine and run and is not a cross-machine performance gate.

`qualify_solve_plan()` produces the corresponding execution evidence for a
public `SolvePlan` or `BatchedSolvePlan`:

```python
report = ti.linalg.experimental.qualify_solve_plan(
    lambda: ti.linalg.experimental.SolvePlan(
        operator,
        method="pcg",
        preconditioner=preconditioner,
        execution_policy="host_check_every_k",
        check_interval=4,
    ),
    rhs,
    reference=expected_solution,
    expected_termination="converged",
    warmup=2,
    repetitions=10,
    metadata={"case": "poisson_level_3"},
)
matrix = ti.linalg.experimental.summarize_solve_qualifications([report])
```

The example uses the CUDA/Vulkan chunked policy; a CPU factory uses
`host_each_iteration` instead.

Passing a zero-argument factory records plan construction separately. An
already constructed plan is also accepted, with build timing explicitly
unavailable. `reference` is either a flat expected solution or a callable
receiving a detached NumPy copy of the RHS. Independent batches use one flat
reference and may provide one expected termination reason per system.

The report separates first and warm synchronous wall time. A qualified
fixed-budget GPU batch additionally separates host submission from completion
wait and may record a supplied `SubmissionPacer`. It records A/M providers,
policy and check interval, terminal state, an independent `b - A(x)` residual,
logical/executed/provider iterations, inactive-work efficiency, chunk
direct/replay counters, transfers, plan resources, and process-global device
pool deltas. Device timestamp spans, device identity, and driver versions stay
explicitly unavailable when the runtime has no safe query; wall time is never
relabeled as device time. Nsight or another profiler can be retained as a
sidecar using the report metadata.

Qualification performs one first solve, the requested warmups and repetitions,
and one untimed operator apply for the independent true residual. It mutates
the plan counters and output supplied to it. Use a dedicated plan/workspace for
performance evidence, especially when measuring an asynchronous batch or a
shared pacer. The function returns detached evidence and never writes files.

The API is covered by backend correctness, lifecycle, trait, composition, 10k
approved-generation churn, and solver regression tests. Application production
qualification remains workload-specific: validate operator semantics,
conditioning, tolerances, preconditioner suitability, failure handling, memory
budgets, and backend driver behavior on representative physical and
non-physical systems.

See also [Sparse runtime and linear algebra](sparse_runtime_and_linear_algebra.en.md)
and [Choosing sparse operators and solvers](physics_sparse_solver_selection.en.md).
