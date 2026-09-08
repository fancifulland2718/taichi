"""Complete shared-A structured-sparse matmul regions, owned by Forge."""

import ctypes
import json
import math
import time
import weakref

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.graph._recipes.deferred import FrozenNativeRecipeSource
from taichi_forge.hardware._admission import _current_cuda_device_scope
from taichi_forge.hardware._bundled_runtime_provider import _binary_sha256
from taichi_forge.hardware._cusparselt import CusparseLtProvider
from taichi_forge.hardware._cusparselt_config import _encode, configured_plan
from taichi_forge.hardware._cusparselt_capture import CusparseLtCaptureRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_runtime_generation,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f16


_SCHEMA = "taichi_forge.sparse_matmul_preparation.v1"


def _names(semantics):
    return (*tuple(x for row in semantics["products"] for x in row), semantics["a"])


def _component(provider):
    runtime = provider._runtime
    adapter = _binary_sha256(runtime.loaded.path)
    vendor = _binary_sha256(runtime.runtime_info["library_path"])
    if adapter is None or vendor is None:
        raise TaichiRuntimeError("Sparse matmul component identity is unavailable")
    return dict(
        provider="cusparselt",
        abi=provider.identity["provider_abi"],
        version=provider.identity["provider_version"],
        configured_execution_abi=2,
        adapter_sha256=adapter,
        library_sha256=vendor,
        plan_identity_scope="frozen_algorithm_attributes_not_serialized_vendor_kernel",
    )


def _plan(provider, semantics, config, *, description=False):
    return configured_plan(
        provider,
        *(semantics[x] for x in ("m", "n", "k")),
        configuration=config["algorithm"],
        preparation_only=description,
        expected_configuration=None if description else config["algorithm"],
        expected_resources=None if description else config["resources"],
    )


class _SparseMatmulCatalog:
    def __init__(self, facts, provider):
        self._facts_json = _canonical_json(facts)
        self._provider_ref = weakref.ref(provider)
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_generation = int(impl.runtime_generation())

    @property
    def facts(self):
        return json.loads(self._facts_json)

    @property
    def semantics(self):
        return self.facts["semantics"]

    @property
    def semantic_fingerprint(self):
        return _digest(self.semantics)

    @property
    def baseline(self):
        return self.facts["baseline"]

    def physical_config(self, key):
        return self.facts["choices"][key]

    def physical_id(self, key):
        return "sparse-matmul-physical:" + _digest(
            dict(
                semantics=self.semantic_fingerprint,
                configuration=self.physical_config(key),
                component=self.facts["component"],
            )
        )

    def _provider(self):
        validate_runtime_generation(
            self, "Sparse matmul catalog belongs to a retired runtime"
        )
        provider = self._provider_ref()
        if provider is None or provider.closed:
            provider = CusparseLtProvider()
            try:
                if _component(provider) != self.facts["component"]:
                    raise TaichiRuntimeError("Sparse matmul provider component drifted")
            except BaseException:
                provider.close()
                raise
            self._provider_ref = weakref.ref(provider)
        return provider

    def append(self, builder, key, admission):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.linalg._sparse_matmul_kernels import relu_kernel

        semantics, config = self.semantics, self.physical_config(key)
        plan = _plan(self._provider(), semantics, config)
        recording = _SparseMatmulRecording(self, key, plan)
        builder.append_native(recording, admission=admission)
        if config["epilogue"] == "separate_relu":
            for _, _, output in semantics["products"]:
                builder.dispatch(
                    relu_kernel(semantics["m"], semantics["n"]),
                    Arg(ArgKind.NDARRAY, output, f16, ndim=2),
                )


class _SparseMatmulRecording(CusparseLtCaptureRecording):
    def __init__(self, source, key, plan):
        semantics = source.semantics
        super().__init__(
            plan,
            a=semantics["a"],
            b=None,
            c=None,
            d=None,
            _products=semantics["products"],
            alpha=semantics["alpha"],
            beta=semantics["beta"],
        )
        object.__setattr__(self, "_graph_sparse_matmul_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(self, "_graph_physical_plan_id", source.physical_id(key))


class _FrozenSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = (
        CusparseLtCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        self._graph_sparse_matmul_source = source

    @property
    def _recording(self):
        return self

    def append_to_graph(self, builder, *, admission):
        source = self._graph_sparse_matmul_source
        source.append(builder, source.baseline, admission)


class _DescriptionRecipe(_CudaGraphCaptureRecipe):
    kind = "semantic_shared_a_sparse_matmul_f16"

    def __init__(self, source):
        self.source = source

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        builder._dispatch_cuda_capture_description(
            program,
            self.kind,
            [
                Arg(ArgKind.NDARRAY, name, f16, ndim=2)
                for name in _names(self.source.semantics)
            ],
        )


class _DescriptionRecording(BackendCommandRecording):
    _graph_binding_frame_capture_safe = (
        CusparseLtCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        super().__init__(
            backend="cuda",
            binding_names=_names(source.semantics),
            command_count=1,
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_graph_sparse_matmul_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(
            self, "_graph_physical_plan_id", source.physical_id(source.baseline)
        )
        object.__setattr__(self, "_cuda_capture_recipe", _DescriptionRecipe(source))

    @property
    def resource_effects(self):
        outputs = {
            row[2] for row in self._graph_sparse_matmul_source.semantics["products"]
        }
        return tuple(
            ResourceEffect(
                name, GraphAccess.WRITE if name in outputs else GraphAccess.READ
            )
            for name in self.binding_names
        )

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "Sparse matmul descriptions must be frozen and materialized"
        )

    def _freeze_graph_recipe_source(self):
        return _FrozenSource(self._graph_sparse_matmul_source)


class SparseMatmulOperation(NativeGraphNode):
    """A group of D_i := activation(alpha * A @ B_i.T + beta * C_i).

    All products have the same M/N/K. A is already-valid, row-major FP16 2:4;
    Forge does not prune or read it back. A/B/C may change every replay. Only
    each product's own C/D may alias; no output may alias any other operand.
    One retained plan compresses current A once per invocation, then executes
    the products in order, using FP32 accumulation and FP16 outputs.
    """

    def __init__(
        self,
        m,
        n,
        k,
        *,
        products=(("b", "c", "output"),),
        a="a",
        alpha=1.0,
        beta=0.0,
        activation="identity",
        absolute_tolerance,
        relative_tolerance,
        preparation=None
    ):
        if any(
            isinstance(x, bool)
            or not isinstance(x, int)
            or x <= 0
            or x % 16
            or x > 2**31 - 1
            for x in (m, n, k)
        ):
            raise ValueError(
                "Sparse matmul requires positive int32 M/N/K multiples of 16"
            )
        if max(m * k, n * k, m * n) > 2**31 - 1:
            raise ValueError(
                "Sparse matmul supports at most 2**31-1 elements per matrix"
            )
        products = tuple(tuple(row) for row in products)
        if not products or any(len(row) != 3 for row in products):
            raise ValueError(
                "Sparse matmul requires nonempty B/C/output binding triples"
            )
        names = (*tuple(x for row in products for x in row), a)
        if any(not isinstance(x, str) or not x for x in names) or len(
            set(names)
        ) != len(names):
            raise ValueError(
                "Sparse matmul binding names must be distinct nonempty strings"
            )
        scalars = (alpha, beta, absolute_tolerance, relative_tolerance)
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in scalars
        ):
            raise ValueError(
                "Sparse matmul coefficients and tolerances must be finite numbers"
            )
        alpha, beta = (ctypes.c_float(x).value for x in (alpha, beta))
        if (
            not all(math.isfinite(x) for x in (alpha, beta))
            or min(absolute_tolerance, relative_tolerance) < 0
            or max(absolute_tolerance, relative_tolerance) == 0
            or activation not in ("identity", "relu")
        ):
            raise ValueError(
                "Sparse matmul requires finite f32 coefficients, nonzero tolerances and identity/relu"
            )
        self._semantics_json = _canonical_json(
            dict(
                operation="shared_a_structured_sparse_matmul",
                m=m,
                n=n,
                k=k,
                products=products,
                a=a,
                alpha=alpha,
                beta=beta,
                activation=activation,
                input_contract="already_valid_fp16_row_2of4",
                storage="compact_row_major",
                weight_lifetime="current_A_each_graph_invocation",
                compute="f32_accumulate_f16_output",
                output_alias="own_C_or_disjoint_from_all_operands",
                numerical_contract=dict(
                    finite_inputs_only=True,
                    bitwise_reproducible=False,
                    absolute_tolerance=float(absolute_tolerance),
                    relative_tolerance=float(relative_tolerance),
                ),
            )
        )
        self._catalog, self._provider_owner, self._closed = None, None, False
        if preparation is not None:
            self._restore(preparation)

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    def prepare(self, *, max_algorithms=8):
        if self._closed:
            raise TaichiRuntimeError("Sparse matmul operation has been closed")
        if self._catalog is not None:
            return self.preparation_artifact()
        if (
            isinstance(max_algorithms, bool)
            or not isinstance(max_algorithms, int)
            or max_algorithms <= 0
        ):
            raise ValueError(
                "Sparse matmul algorithm preparation budget must be a positive integer"
            )
        if not hasattr(
            getattr(core, "_CudaCusparseLtCapturePlan", None), "matmul_count"
        ):
            raise TaichiRuntimeError(
                "Sparse matmul requires native shared-A group capture support"
            )
        provider = CusparseLtProvider()
        start = time.perf_counter()
        choices, unavailable, baseline = {}, [], None
        semantics = self.semantics
        try:
            # Vendor default with legal compression reuse is the baseline, not
            # a deliberately repeated-compression path. No vendor search runs.
            default = {"activation": semantics["activation"]}
            with _plan(
                provider, semantics, dict(algorithm=default), description=True
            ) as plan:
                default_id, count = (
                    plan._configuration["algorithm_id"],
                    plan._algorithm_count,
                )
            ids = tuple(
                dict.fromkeys((default_id, *range(min(count, max_algorithms))))
            )[:max_algorithms]
            for algorithm_id in ids:
                for epilogue in (
                    ("vendor", "separate_relu")
                    if semantics["activation"] == "relu"
                    else ("vendor",)
                ):
                    requested = dict(
                        algorithm_id=algorithm_id,
                        activation=(
                            semantics["activation"]
                            if epilogue == "vendor"
                            else "identity"
                        ),
                    )
                    try:
                        plan = _plan(
                            provider,
                            semantics,
                            dict(algorithm=requested),
                            description=True,
                        )
                    except TaichiRuntimeError as error:
                        if baseline is None:
                            raise
                        unavailable.append(
                            dict(
                                algorithm=requested,
                                epilogue=epilogue,
                                reason=str(error),
                            )
                        )
                        continue
                    try:
                        config = dict(
                            algorithm=dict(plan._configuration),
                            epilogue=epilogue,
                            resources=(
                                plan.compressed_bytes,
                                plan.compression_buffer_bytes,
                                plan.workspace_bytes,
                            ),
                            compression="once_per_graph_invocation",
                            submission="ordered_shared_plan",
                        )
                        key = "sparse-matmul:" + _digest(config)
                        choices[key] = config
                        baseline = baseline or key
                    finally:
                        plan.close()
            facts = dict(
                schema=_SCHEMA,
                semantics=semantics,
                component=_component(provider),
                device=_current_cuda_device_scope(),
                choices=choices,
                baseline=baseline,
                preparation=dict(
                    origin="current_process_descriptors_only",
                    host_seconds=time.perf_counter() - start,
                    algorithm_count=count,
                    described_algorithm_ids=ids,
                    enumeration_complete=len(ids) == count,
                    vendor_library_path=provider.identity["vendor_library"],
                ),
                unavailable=unavailable,
            )
            self._catalog, self._provider_owner = (
                _SparseMatmulCatalog(facts, provider),
                provider,
            )
        except BaseException:
            provider.close()
            raise
        return self.preparation_artifact()

    def _restore(self, preparation):
        facts = json.loads(_canonical_json(preparation))
        if (
            not isinstance(facts, dict)
            or facts.get("schema") != _SCHEMA
            or facts.get("semantics") != self.semantics
            or not isinstance(facts.get("choices"), dict)
            or facts.get("baseline") not in facts["choices"]
        ):
            raise ValueError("Sparse matmul preparation semantic contract drifted")
        for key, config in facts["choices"].items():
            if set(config) != {
                "algorithm",
                "epilogue",
                "resources",
                "compression",
                "submission",
            }:
                raise ValueError("Sparse matmul physical configuration fields drifted")
            _encode(config["algorithm"])
            resources = config["resources"]
            if (
                not isinstance(resources, list)
                or len(resources) != 3
                or any(
                    isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**63
                    for x in resources
                )
                or not resources[0]
            ):
                raise ValueError("Sparse matmul resource contract drifted")
            epilogue = config["epilogue"]
            activation = self.semantics["activation"]
            if (
                epilogue not in ("vendor", "separate_relu")
                or (epilogue == "separate_relu" and activation != "relu")
                or config["algorithm"].get("activation")
                != (activation if epilogue == "vendor" else "identity")
                or config["compression"] != "once_per_graph_invocation"
                or config["submission"] != "ordered_shared_plan"
                or key != "sparse-matmul:" + _digest(config)
            ):
                raise ValueError("Sparse matmul dataflow/choice identity drifted")
        if facts["choices"][facts["baseline"]]["epilogue"] != "vendor":
            raise ValueError("Sparse matmul baseline dataflow drifted")
        if facts.get("device") != _current_cuda_device_scope():
            raise ValueError("Sparse matmul preparation device contract drifted")
        provider = CusparseLtProvider()
        try:
            if facts.get("component") != _component(provider):
                raise ValueError("Sparse matmul preparation component drifted")
        except BaseException:
            provider.close()
            raise
        facts["preparation"] = {
            **facts.get("preparation", {}),
            "origin": "imported_preparation_not_current_measurement",
        }
        self._catalog, self._provider_owner = (
            _SparseMatmulCatalog(facts, provider),
            provider,
        )

    def preparation_artifact(self):
        if self._catalog is None:
            raise TaichiRuntimeError(
                "Call sparse_matmul.prepare() before constructing its Graph"
            )
        return self._catalog.facts

    def _graph_recipe_description(self):
        if self._closed:
            raise TaichiRuntimeError("Sparse matmul operation has been closed")
        self.preparation_artifact()
        return native_recording_node(
            _DescriptionRecording(self._catalog),
            lifetime_leases=(self._catalog,),
            debug_info={"kind": "shared_a_sparse_matmul_region"},
        ).compile()

    def compile(self):
        return self._graph_recipe_description()

    def close(self):
        self._closed, self._provider_owner = True, None


def record_sparse_matmul(m, n, k, **kwargs):
    """Describe shared-A FP16 2:4 matmuls; see SparseMatmulOperation.

    prepare() freezes bounded algorithm/epilogue dataflows without reading
    caller data or allocating candidate GPU scratch. Search complete recipes
    with hardware.tensor.SparseMatmulRecipeProvider, not raw algorithm axes.
    """
    return SparseMatmulOperation(m, n, k, **kwargs)


__all__ = ["record_sparse_matmul"]
