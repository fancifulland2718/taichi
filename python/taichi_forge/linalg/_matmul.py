"""Semantic matmul regions with frozen, provider-owned physical recipes."""

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
from taichi_forge.hardware._cublaslt import (
    CUBLASLT_PROVIDER_ABI,
    CublasLtMatmulPlan,
    CublasLtProvider,
    _positive_dimension,
    _workspace_limit,
    _version_string,
)
from taichi_forge.hardware._cublaslt_algorithms import (
    _AlgorithmChoice,
    _MatmulRecipePlan,
)
from taichi_forge.hardware._cublaslt_capture import (
    CublasLtCaptureRecording,
    _BINDING_FRAMES_SUPPORTED,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_runtime_generation,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._packing_kernels import TILED_PACKING_IMPLEMENTATION
from taichi_forge.types.primitive_types import f32


_SCHEMA = "taichi_forge.matmul_preparation.v1"


def _component(provider):
    return {
        "provider": "cublaslt",
        "abi": CUBLASLT_PROVIDER_ABI,
        "version": _version_string(provider.version),
        "library": provider._library.candidate,
        "configuration_schema": "documented-algo-config-v1",
        "forge_packing_implementation": TILED_PACKING_IMPLEMENTATION,
    }


def _plan_semantics(semantics, packed_inputs=()):
    result = {
        key: semantics[key]
        for key in (
            "m",
            "n",
            "k",
            "batch_count",
            "transpose_a",
            "transpose_b",
            "alpha",
            "beta",
            "a",
            "b",
            "output",
        )
    }
    for name in packed_inputs:
        result[f"transpose_{name}"] = False
    return result


def _shapes(semantics):
    m, n, k = (semantics[key] for key in ("m", "n", "k"))
    shapes = (
        (k, m) if semantics["transpose_a"] else (m, k),
        (n, k) if semantics["transpose_b"] else (k, n),
        (m, n),
    )
    return tuple(
        (semantics["batch_count"], *shape) if semantics["batch_count"] > 1 else shape
        for shape in shapes
    )


def _packing(semantics):
    return tuple(
        name
        for name, dimensions in (("a", ("m", "k")), ("b", ("k", "n")))
        if semantics[f"transpose_{name}"]
        and min(semantics[key] for key in dimensions) > 1
    )


def _storage_bytes(semantics, configuration):
    packed = configuration["packed_inputs"]
    batch = semantics["batch_count"]
    return (batch * semantics["m"] * semantics["k"] * 4 if "a" in packed else 0) + (
        batch * semantics["k"] * semantics["n"] * 4 if "b" in packed else 0
    )


class _MatmulCatalog:
    """Immutable facts plus a weak handle lookup; it owns no trial plan."""

    def __init__(self, artifact, provider):
        self._facts_json = _canonical_json(artifact)
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_generation = int(impl.runtime_generation())
        self._provider_ref = weakref.ref(provider)

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

    def physical_config(self, choice_id):
        return self.facts["choices"][choice_id]

    def physical_id(self, choice_id):
        return "matmul-physical:" + _digest(
            {
                "semantics": self.semantic_fingerprint,
                "configuration": self.physical_config(choice_id),
                "component": self.facts["component"],
            }
        )

    def _provider(self):
        validate_runtime_generation(self, "Matmul catalog belongs to a retired runtime")
        provider = self._provider_ref()
        if provider is None or provider.closed:
            provider = CublasLtProvider()
            if _component(provider) != self.facts["component"]:
                provider.close()
                raise TaichiRuntimeError("Matmul provider component drifted")
            self._provider_ref = weakref.ref(provider)
        return provider

    def append(self, builder, choice_id, admission):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.linalg._matmul_kernels import (
            packing_kernel,
            relu_kernel,
        )

        config = self.physical_config(choice_id)
        semantics = self.semantics
        physical = _plan_semantics(semantics, config["packed_inputs"])
        rank = 3 if semantics["batch_count"] > 1 else 2
        prefix = (
            f"__forge_matmul_{builder._dispatch_count}_{self.semantic_fingerprint[:12]}"
        )
        public_names = {semantics[key] for key in ("a", "b", "output")}
        while any(
            f"{prefix}_{suffix}" in public_names
            or f"{prefix}_{suffix}" in builder._runtime_graph_arg_names
            for suffix in ("a", "b", "workspace")
        ):
            prefix += "_"
        for name in config["packed_inputs"]:
            shape = _shapes(physical)[0 if name == "a" else 1]
            packed = builder.private_ndarray(f"{prefix}_{name}", f32, shape)
            builder.dispatch(
                packing_kernel(
                    shape,
                    tiled=config["packing_lowering"] == TILED_PACKING_IMPLEMENTATION,
                ),
                Arg(ArgKind.NDARRAY, semantics[name], f32, ndim=rank),
                packed,
            )
            physical[name] = packed.name
        plan = _MatmulRecipePlan(
            self._provider(),
            physical,
            layout="row_major",
            epilogue="relu" if config["epilogue"] == "fused" else "identity",
            workspace_limit_bytes=config["algorithm"]["workspace_bytes"],
            choice=_AlgorithmChoice.from_dict(config["algorithm"]),
        )
        recording = _MatmulRecording(
            self, choice_id, plan, workspace=f"{prefix}_workspace"
        )
        builder.append_native(recording, admission=admission)
        if config["epilogue"] == "separate":
            builder.dispatch(
                relu_kernel(_shapes(semantics)[2]),
                Arg(ArgKind.NDARRAY, semantics["output"], f32, ndim=rank),
            )


class _FrozenMatmulSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = _BINDING_FRAMES_SUPPORTED

    def __init__(self, source):
        self._graph_matmul_source = source

    @property
    def _recording(self):
        return self

    def append_to_graph(self, builder, *, admission):
        self._graph_matmul_source.append(
            builder, self._graph_matmul_source.baseline, admission
        )


class _MatmulDescriptionRecipe(_CudaGraphCaptureRecipe):
    kind = "semantic_matmul_f32"

    def __init__(self, source):
        self.source = source

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        semantics = self.source.semantics
        rank = 3 if semantics["batch_count"] > 1 else 2
        builder._dispatch_cuda_capture_description(
            program,
            self.kind,
            [
                Arg(ArgKind.NDARRAY, semantics[name], f32, ndim=rank)
                for name in ("a", "b", "output")
            ],
        )


class _MatmulDescriptionRecording(BackendCommandRecording):
    _graph_binding_frame_capture_safe = _BINDING_FRAMES_SUPPORTED

    def __init__(self, source):
        semantics = source.semantics
        super().__init__(
            backend="cuda",
            binding_names=tuple(semantics[key] for key in ("a", "b", "output")),
            command_count=1,
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_graph_matmul_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(
            self, "_graph_physical_plan_id", source.physical_id(source.baseline)
        )
        object.__setattr__(
            self, "_cuda_capture_recipe", _MatmulDescriptionRecipe(source)
        )

    @property
    def resource_effects(self):
        semantics = self._graph_matmul_source.semantics
        return (
            ResourceEffect(semantics["a"], GraphAccess.READ),
            ResourceEffect(semantics["b"], GraphAccess.READ),
            ResourceEffect(
                semantics["output"],
                GraphAccess.READ_WRITE if semantics["beta"] else GraphAccess.WRITE,
            ),
        )

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "Matmul descriptions must be frozen and materialized before execution"
        )

    def _freeze_graph_recipe_source(self):
        return _FrozenMatmulSource(self._graph_matmul_source)


class _MatmulRecording(CublasLtCaptureRecording):
    graph_publish_time_binding_validation_stable = True

    def __init__(self, source, choice_id, plan, *, workspace):
        super().__init__(plan, workspace=workspace)
        object.__setattr__(self, "_graph_matmul_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(
            self, "_graph_physical_plan_id", source.physical_id(choice_id)
        )
        object.__setattr__(self, "_public_semantics", source.semantics)

    def validate_graph_bindings(self, bindings):
        # Original public operands matter even when the physical plan reads
        # packed private arrays. Immutable Graph.bind versions discharge this.
        names = tuple(self._public_semantics[key] for key in ("a", "b", "output"))
        values = tuple(bindings[name] for name in names)
        for name, value, shape in zip(names, values, _shapes(self._public_semantics)):
            CublasLtMatmulPlan._validate_array(value, name, shape)
        if values[2] is values[0] or values[2] is values[1]:
            raise TaichiRuntimeError("Matmul output must not alias either public input")

    def _graph_provider_memory_report(self):
        return make_memory_report(
            "matmul_region",
            "cuda",
            (
                HardwareMemoryComponent(
                    "algorithm_workspace",
                    self.plan.workspace_bytes,
                    True,
                    "provider_generation",
                    "provider",
                    resident=not self.plan.closed,
                ),
                HardwareMemoryComponent(
                    "vendor_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=not self.plan.closed,
                ),
            ),
            lifecycle_state="closed" if self.plan.closed else "ready",
            ownership_scope="plan_generation",
        )


class MatmulOperation(NativeGraphNode):
    """Fixed-shape, compact f32 matmul semantics, currently materialized on CUDA.

    D := activation(alpha * op(A) @ op(B) + beta * D). Input/output bindings
    are distinct; input values may change on every replay. Finite values and
    application-qualified tolerances are caller obligations, not replay scans.
    Call prepare(), append_native(), freeze(), then compile/materialize/search.
    """

    def __init__(
        self,
        m,
        n,
        k,
        *,
        batch_count=1,
        transpose_a=False,
        transpose_b=False,
        alpha=1.0,
        beta=0.0,
        activation="identity",
        a="a",
        b="b",
        output="output",
        absolute_tolerance,
        relative_tolerance,
        preparation=None,
    ):
        dimensions = {
            key: _positive_dimension(value, key)
            for key, value in (
                ("m", m),
                ("n", n),
                ("k", k),
                ("batch_count", batch_count),
            )
        }
        if not isinstance(transpose_a, bool) or not isinstance(transpose_b, bool):
            raise TypeError("Matmul transpose flags must be bool")
        if activation not in ("identity", "relu"):
            raise ValueError("Matmul activation must be identity or relu")
        if (
            any(not isinstance(name, str) or not name for name in (a, b, output))
            or len({a, b, output}) != 3
        ):
            raise ValueError("Matmul requires distinct nonempty binding names")
        scalars = []
        for value in (alpha, beta, absolute_tolerance, relative_tolerance):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    "Matmul coefficients and tolerances must be finite numbers"
                )
            scalars.append(float(value))
        for index in (0, 1):
            scalars[index] = ctypes.c_float(scalars[index]).value
            if not math.isfinite(scalars[index]):
                raise ValueError("Matmul alpha and beta must be representable in f32")
        if min(scalars[2:]) < 0 or not any(scalars[2:]):
            raise ValueError(
                "Matmul requires nonnegative tolerances and at least one positive tolerance"
            )
        self._semantics_json = _canonical_json(
            {
                "operation": "matmul",
                **dimensions,
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "alpha": scalars[0],
                "beta": scalars[1],
                "activation": activation,
                "a": a,
                "b": b,
                "output": output,
                "numerical_contract": {
                    "dtype": "f32",
                    "accumulation": "f32",
                    "finite_inputs_only": True,
                    "bitwise_reproducible": False,
                    "absolute_tolerance": scalars[2],
                    "relative_tolerance": scalars[3],
                },
                "storage": "compact_row_major",
                "output_alias": "disjoint_from_inputs",
            }
        )
        self._catalog, self._provider_owner = None, None
        self._closed = False
        if preparation is not None:
            self._restore(preparation)

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    def prepare(self, *, workspace_limit_bytes=32 << 20, heuristic_limit=4):
        if self._closed:
            raise TaichiRuntimeError("Matmul operation has been closed")
        if self._catalog is not None:
            return self.preparation_artifact()
        workspace_limit_bytes = _workspace_limit(workspace_limit_bytes)
        if (
            isinstance(heuristic_limit, bool)
            or not isinstance(heuristic_limit, int)
            or not 1 <= heuristic_limit <= 32
        ):
            raise ValueError("Matmul heuristic_limit must be in [1, 32]")
        provider = CublasLtProvider()
        if not hasattr(core, "_CudaCublasLtCapturePlan"):
            provider.close()
            raise TaichiRuntimeError(
                "Matmul recipes require typed native capture support"
            )
        semantics, choices, failures = self.semantics, {}, []
        begin = time.perf_counter()
        layouts = ((), _packing(semantics)) if _packing(semantics) else ((),)
        epilogues = (
            ("separate", "fused")
            if semantics["activation"] == "relu"
            else ("identity",)
        )
        baseline = None
        try:
            for packed in layouts:
                for epilogue in epilogues:
                    for limit in dict.fromkeys((workspace_limit_bytes, 0)):
                        try:
                            description = _MatmulRecipePlan(
                                provider,
                                _plan_semantics(semantics, packed),
                                layout="row_major",
                                epilogue="relu" if epilogue == "fused" else "identity",
                                workspace_limit_bytes=limit,
                                shortlist_size=heuristic_limit,
                                preparation_only=True,
                            )
                        except TaichiRuntimeError as error:
                            if getattr(error, "vendor_status", None) not in (
                                8,
                                15,
                            ) and "no compatible physical matmul strategy" not in str(
                                error
                            ):
                                raise
                            failures.append(
                                {
                                    "packed_inputs": packed,
                                    "epilogue": epilogue,
                                    "workspace_limit_bytes": limit,
                                    "status": "unavailable",
                                    "reason": str(error),
                                }
                            )
                            continue
                        try:
                            for choice in description.choices:
                                base_config = {
                                    "packed_inputs": packed,
                                    "epilogue": epilogue,
                                    "algorithm": choice.to_dict(),
                                    "submission": "enclosing_graph",
                                    "workspace_lifetime": "retained_plan",
                                }
                                lowerings = (
                                    ("direct-f32-v1", TILED_PACKING_IMPLEMENTATION)
                                    if packed
                                    else (None,)
                                )
                                for lowering in lowerings:
                                    config = {
                                        **base_config,
                                        "packing_lowering": lowering,
                                    }
                                    key = "matmul:" + _digest(config)
                                    choices[key] = config
                                    if (
                                        baseline is None
                                        and not packed
                                        and epilogue == epilogues[0]
                                    ):
                                        baseline = key
                        finally:
                            description.close()
            if baseline is None:
                raise TaichiRuntimeError(
                    "No baseline matmul plan is available for the declared semantics"
                )
            artifact = {
                "schema": _SCHEMA,
                "semantics": semantics,
                "component": _component(provider),
                "device": _current_cuda_device_scope(),
                "baseline": baseline,
                "choices": choices,
                "preparation": {
                    "origin": "current_process_descriptors_only",
                    "host_seconds": time.perf_counter() - begin,
                    "workspace_limit_bytes": workspace_limit_bytes,
                    "heuristic_limit": heuristic_limit,
                },
                "unavailable": failures,
            }
            self._catalog = _MatmulCatalog(artifact, provider)
            self._provider_owner = provider
        except BaseException:
            provider.close()
            raise
        return self.preparation_artifact()

    def _restore(self, preparation):
        artifact = json.loads(_canonical_json(preparation))
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema") != _SCHEMA
            or artifact.get("semantics") != self.semantics
        ):
            raise ValueError("Matmul preparation schema or semantic contract drifted")
        choices = artifact.get("choices")
        if not isinstance(choices, dict) or artifact.get("baseline") not in choices:
            raise ValueError("Matmul preparation requires a frozen baseline")
        for key, config in choices.items():
            if not isinstance(config, dict) or set(config) != {
                "packed_inputs",
                "packing_lowering",
                "epilogue",
                "algorithm",
                "submission",
                "workspace_lifetime",
            }:
                raise ValueError("Invalid frozen matmul physical configuration")
            if tuple(config["packed_inputs"]) not in ((), _packing(self.semantics)):
                raise ValueError("Invalid frozen operand packing")
            allowed_lowerings = (
                ("direct-f32-v1", TILED_PACKING_IMPLEMENTATION)
                if config["packed_inputs"]
                else (None,)
            )
            if config["packing_lowering"] not in allowed_lowerings:
                raise ValueError("Matmul packing lowering drifted")
            allowed = (
                ("separate", "fused")
                if self.semantics["activation"] == "relu"
                else ("identity",)
            )
            if (
                config["epilogue"] not in allowed
                or config["submission"] != "enclosing_graph"
                or config["workspace_lifetime"] != "retained_plan"
            ):
                raise ValueError("Matmul preparation physical contract drifted")
            _AlgorithmChoice.from_dict(config["algorithm"])
            if key != "matmul:" + _digest(config):
                raise ValueError("Matmul preparation choice identity drifted")
        if artifact.get("device") != _current_cuda_device_scope():
            raise ValueError("Matmul preparation device contract drifted")
        provider = CublasLtProvider()
        if artifact.get("component") != _component(provider):
            provider.close()
            raise ValueError("Matmul preparation provider component drifted")
        artifact["preparation"] = {
            **artifact.get("preparation", {}),
            "origin": "imported_preparation_not_current_measurement",
        }
        self._catalog = _MatmulCatalog(artifact, provider)
        self._provider_owner = provider

    def preparation_artifact(self):
        if self._catalog is None:
            raise TaichiRuntimeError(
                "Call matmul.prepare() before building or searching its Graph"
            )
        return self._catalog.facts

    def _graph_recipe_description(self):
        if self._closed:
            raise TaichiRuntimeError("Matmul operation has been closed")
        self.preparation_artifact()
        recording = _MatmulDescriptionRecording(self._catalog)
        return native_recording_node(
            recording,
            lifetime_leases=(self._catalog,),
            debug_info={"kind": "matmul_region"},
        ).compile()

    def compile(self):
        return self._graph_recipe_description()

    def close(self):
        self._closed = True
        self._provider_owner = None  # Existing Graph plan leases remain owners.


def record_matmul(m, n, k, **kwargs):
    """Describe a complete matmul optimization region; see MatmulOperation.

    Call prepare(), append to a GraphBuilder, and freeze it before compiling or
    using MatmulRecipeProvider. Imported preparation facts create no trial plan.
    """
    return MatmulOperation(m, n, k, **kwargs)


__all__ = ["record_matmul"]
