"""Semantic tensor contractions with complete, reconstructible Graph dataflows."""

import ctypes
import itertools
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
from taichi_forge.hardware._cutensor import (
    CutensorContractionPlan,
    CutensorProvider,
    _validate_array,
)
from taichi_forge.hardware._cutensor_capture import CutensorCaptureRecording
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_runtime_generation,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


_SCHEMA = "taichi_forge.contraction_preparation.v1"
_POLICY = "vendor_default_no_jit_or_autotune"


def _component(provider):
    runtime = provider._runtime
    adapter_hash = _binary_sha256(runtime.loaded.path)
    library_hash = _binary_sha256(runtime.runtime_info["library_path"])
    if adapter_hash is None or library_hash is None:
        raise TaichiRuntimeError(
            "Contraction preparation binary identity is unavailable"
        )
    return {
        "provider": "cutensor",
        "abi": provider.identity["provider_abi"],
        "version": provider.identity["provider_version"],
        "adapter_sha256": adapter_hash,
        "library_sha256": library_hash,
        "vendor_build_version": runtime.runtime_info["build_version"],
        "cuda_runtime_version": runtime.runtime_info["cuda_runtime_version"],
        "plan_identity_scope": "declared_dataflow_and_plan_request_not_vendor_kernel_binary",
    }


def _tensor(shape, modes, name):
    shape = tuple(shape)
    if (
        not shape
        or len(shape) > 12
        or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)
    ):
        raise ValueError(
            f"Contraction {name} needs 1-12 positive integer dimensions (Forge ndarray rank limit)"
        )
    if math.prod(shape) > 2**31 - 1:
        raise ValueError(
            "Contraction dataflow helpers require at most 2**31-1 elements per tensor"
        )
    modes = tuple(ord(x) for x in modes) if isinstance(modes, str) else tuple(modes)
    if (
        len(modes) != len(shape)
        or len(set(modes)) != len(modes)
        or any(
            isinstance(x, bool) or not isinstance(x, int) or not -(2**31) <= x < 2**31
            for x in modes
        )
    ):
        raise ValueError(
            f"Contraction {name} modes must be unique int32 labels matching its rank"
        )
    return {"shape": shape, "modes": modes}


def _permutations(semantics):
    output = semantics["out"]["modes"]
    reduction = tuple(x for x in semantics["a_tensor"]["modes"] if x not in output)
    options = []
    for name in ("a_tensor", "b_tensor"):
        tensor = semantics[name]
        modes = tensor["modes"]
        identity = tuple(range(len(modes)))
        order = tuple(modes.index(x) for x in (*output, *reduction) if x in modes)
        # Moving singleton axes alone does not change physical storage.
        effective = lambda p: tuple(modes[i] for i in p if tensor["shape"][i] > 1)
        options.append(
            (identity, order)
            if effective(identity) != effective(order)
            else (identity,)
        )
    return tuple(itertools.product(*options))


def _dataflows(semantics):
    epilogues = (
        ("vendor", "separate")
        if semantics["alpha"] != 1.0 or semantics["beta"] != 0.0
        else ("vendor",)
    )
    return tuple(
        {"permutations": {"a": a, "b": b}, "epilogue": e}
        for a, b in _permutations(semantics)
        for e in epilogues
    )


def _physical_tensors(semantics, config):
    tensors = []
    for name in ("a", "b"):
        tensor, permutation = semantics[f"{name}_tensor"], config["permutations"][name]
        tensors.append(
            {
                key: tuple(tensor[key][i] for i in permutation)
                for key in ("shape", "modes")
            }
        )
    return (*tensors, semantics["out"], semantics["out"])


def _packing(semantics, config):
    return tuple(
        name
        for name in ("a", "b")
        if tuple(config["permutations"][name])
        != tuple(range(len(semantics[f"{name}_tensor"]["shape"])))
    )


def _storage_bytes(semantics, config):
    return 4 * (
        sum(
            math.prod(semantics[f"{name}_tensor"]["shape"])
            for name in _packing(semantics, config)
        )
        + (
            math.prod(semantics["out"]["shape"])
            if config["epilogue"] == "separate"
            else 0
        )
    )


class _PlanDescription(CutensorContractionPlan):
    def execute(self, *args, **kwargs):
        raise TaichiRuntimeError("Contraction preparation descriptions cannot execute")


def _make_plan(provider, semantics, config, *, description=False):
    tensors = _physical_tensors(semantics, config)
    cls = _PlanDescription if description else CutensorContractionPlan
    with provider._lock:
        provider._validate_lifetime()
        return cls(
            provider,
            *(x for t in tensors for x in (t["shape"], t["modes"])),
            compute=semantics["compute"],
            alignment_bytes=128,
            workspace_preference="default",
            workspace_limit_bytes=config["workspace_limit_bytes"],
            _preparation_only=description,
            _expected_workspace_bytes=(
                None if description else config["workspace_bytes"]
            ),
        )


class _ContractionCatalog:
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

    def physical_config(self, key):
        return self.facts["choices"][key]

    def physical_id(self, key):
        return "contraction-physical:" + _digest(
            {
                "semantics": self.semantic_fingerprint,
                "configuration": self.physical_config(key),
                "component": self.facts["component"],
            }
        )

    def _provider(self):
        validate_runtime_generation(
            self, "Contraction catalog belongs to a retired runtime"
        )
        provider = self._provider_ref()
        if provider is None or provider.closed:
            provider = CutensorProvider()
            try:
                if _component(provider) != self.facts["component"]:
                    raise TaichiRuntimeError("Contraction provider component drifted")
            except BaseException:
                provider.close()
                raise
            self._provider_ref = weakref.ref(provider)
        return provider

    def append(self, builder, key, admission):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.linalg._contraction_kernels import (
            epilogue_kernel,
            permutation_kernel,
        )

        config, semantics = self.physical_config(key), self.semantics
        physical = _physical_tensors(semantics, config)
        names = {name: semantics[name] for name in ("a", "b", "c", "output")}
        prefix = f"__forge_contraction_{builder._dispatch_count}_{self.semantic_fingerprint[:12]}"
        while any(
            f"{prefix}_{suffix}" in set(names.values())
            or f"{prefix}_{suffix}" in builder._runtime_graph_arg_names
            for suffix in ("a", "b", "product", "workspace")
        ):
            prefix += "_"
        private_args = {}

        def arg(name, shape):
            if name in private_args:
                return private_args[name]
            return Arg(ArgKind.NDARRAY, name, f32, ndim=len(shape))

        for name in _packing(semantics, config):
            shape = tuple(physical[0 if name == "a" else 1]["shape"])
            buffer = builder.private_ndarray(f"{prefix}_{name}", f32, shape)
            private_args[buffer.name] = buffer
            builder.dispatch(
                permutation_kernel(shape, tuple(config["permutations"][name])),
                arg(names[name], shape),
                buffer,
            )
            names[name] = buffer.name
        shape = tuple(semantics["out"]["shape"])
        if config["epilogue"] == "separate":
            product = builder.private_ndarray(f"{prefix}_product", f32, shape)
            private_args[product.name] = product
            names["output"] = product.name
        plan = _make_plan(self._provider(), semantics, config)
        recording = _ContractionRecording(
            self, key, plan, names, workspace=f"{prefix}_workspace"
        )
        builder.append_native(recording, admission=admission)
        if config["epilogue"] == "separate" or semantics["activation"] == "relu":
            alpha, beta = (
                (semantics["alpha"], semantics["beta"])
                if config["epilogue"] == "separate"
                else (1.0, 0.0)
            )
            builder.dispatch(
                epilogue_kernel(shape, alpha, beta, semantics["activation"] == "relu"),
                arg(names["output"], shape),
                arg(semantics["c"], shape),
                arg(semantics["output"], shape),
            )


class _FrozenContractionSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = (
        CutensorCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        self._graph_contraction_source = source

    @property
    def _recording(self):
        return self

    def append_to_graph(self, builder, *, admission):
        source = self._graph_contraction_source
        source.append(builder, source.baseline, admission)


class _DescriptionRecipe(_CudaGraphCaptureRecipe):
    kind = "semantic_contraction_f32"

    def __init__(self, source):
        self.source = source

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        semantics = self.source.semantics
        shapes = (
            semantics["a_tensor"]["shape"],
            semantics["b_tensor"]["shape"],
            semantics["out"]["shape"],
            semantics["out"]["shape"],
        )
        builder._dispatch_cuda_capture_description(
            program,
            self.kind,
            [
                Arg(ArgKind.NDARRAY, semantics[name], f32, ndim=len(shape))
                for name, shape in zip(("a", "b", "c", "output"), shapes)
            ],
        )


class _DescriptionRecording(BackendCommandRecording):
    _graph_binding_frame_capture_safe = (
        CutensorCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        super().__init__(
            backend="cuda",
            binding_names=tuple(
                source.semantics[name] for name in ("a", "b", "c", "output")
            ),
            command_count=1,
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_graph_contraction_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(
            self, "_graph_physical_plan_id", source.physical_id(source.baseline)
        )
        object.__setattr__(self, "_cuda_capture_recipe", _DescriptionRecipe(source))

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(name, GraphAccess.WRITE if i == 3 else GraphAccess.READ)
            for i, name in enumerate(self.binding_names)
        )

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "Contraction descriptions must be frozen and materialized"
        )

    def _freeze_graph_recipe_source(self):
        return _FrozenContractionSource(self._graph_contraction_source)


class _ContractionRecording(CutensorCaptureRecording):
    graph_publish_time_binding_validation_stable = True

    def __init__(self, source, key, plan, names, *, workspace):
        semantics = source.semantics
        separate = source.physical_config(key)["epilogue"] == "separate"
        super().__init__(
            plan,
            a=names["a"],
            b=names["b"],
            c=names["c"],
            d=names["output"],
            workspace=workspace,
            alpha=1.0 if separate else semantics["alpha"],
            beta=0.0 if separate else semantics["beta"],
        )
        object.__setattr__(self, "_graph_contraction_source", source)
        object.__setattr__(self, "_public_semantics", semantics)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(self, "_graph_physical_plan_id", source.physical_id(key))

    def validate_graph_bindings(self, bindings):
        semantics = self._public_semantics
        values = [bindings[semantics[name]] for name in ("a", "b", "c", "output")]
        shapes = (
            semantics["a_tensor"]["shape"],
            semantics["b_tensor"]["shape"],
            semantics["out"]["shape"],
            semantics["out"]["shape"],
        )
        for value, shape, name in zip(values, shapes, ("a", "b", "c", "output")):
            _validate_array(value, tuple(shape), semantics[name])
        if values[3] is values[0] or values[3] is values[1]:
            raise TaichiRuntimeError("Contraction output must not alias A or B")


class ContractionOperation(NativeGraphNode):
    """D := activation(alpha * contract(A,B) + beta*C), with caller tolerance.

    Compact f32 arrays, explicit unique modes, matched extents (no broadcasting),
    at most 12 dimensions and 2**31-1 elements per tensor. Every reduced mode
    occurs in both inputs; shared output modes represent batches. C/D may alias.
    Input values may change every replay. prepare() freezes complete dataflows,
    not opaque vendor kernel binaries; rebuild a selected plan on restoration.
    """

    def __init__(
        self,
        a_shape,
        a_modes,
        b_shape,
        b_modes,
        output_modes,
        *,
        alpha=1.0,
        beta=0.0,
        activation="identity",
        compute="f32",
        a="a",
        b="b",
        c="c",
        output="output",
        absolute_tolerance,
        relative_tolerance,
        preparation=None,
    ):
        at, bt = _tensor(a_shape, a_modes, "A"), _tensor(b_shape, b_modes, "B")
        extents = dict(zip(at["modes"], at["shape"]))
        for label, extent in zip(bt["modes"], bt["shape"]):
            if label in extents and extents[label] != extent:
                raise ValueError(
                    "Contraction mode extents disagree; broadcasting is not implicit"
                )
            extents[label] = extent
        output_modes = (
            tuple(ord(x) for x in output_modes)
            if isinstance(output_modes, str)
            else tuple(output_modes)
        )
        if not set(output_modes) <= extents.keys():
            raise ValueError("Contraction output modes must occur in the inputs")
        out = _tensor(tuple(extents[x] for x in output_modes), output_modes, "output")
        reduced = extents.keys() - set(output_modes)
        if not reduced <= (set(at["modes"]) & set(bt["modes"])):
            raise ValueError("Each reduced contraction mode must occur in both inputs")
        names = (a, b, c, output)
        if any(not isinstance(x, str) or not x for x in names) or len(set(names)) != 4:
            raise ValueError("Contraction bindings must have distinct nonempty names")
        if activation not in ("identity", "relu") or compute not in ("f32", "tf32"):
            raise ValueError(
                "Contraction supports identity/relu activation and f32/tf32 compute"
            )
        scalars = (alpha, beta, absolute_tolerance, relative_tolerance)
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            for x in scalars
        ):
            raise ValueError(
                "Contraction coefficients and tolerances must be finite numbers"
            )
        alpha, beta = (ctypes.c_float(x).value for x in (alpha, beta))
        if (
            not math.isfinite(alpha)
            or not math.isfinite(beta)
            or min(absolute_tolerance, relative_tolerance) < 0
            or max(absolute_tolerance, relative_tolerance) == 0
        ):
            raise ValueError(
                "Contraction requires finite f32 coefficients and nonnegative, nonzero tolerances"
            )
        self._semantics_json = _canonical_json(
            dict(
                operation="contraction",
                a_tensor=at,
                b_tensor=bt,
                out=out,
                alpha=alpha,
                beta=beta,
                activation=activation,
                compute=compute,
                a=a,
                b=b,
                c=c,
                output=output,
                numerical_contract=dict(
                    dtype="f32",
                    finite_inputs_only=True,
                    bitwise_reproducible=False,
                    absolute_tolerance=float(absolute_tolerance),
                    relative_tolerance=float(relative_tolerance),
                ),
                storage="compact_row_major",
                output_alias="identical_C_or_disjoint",
            )
        )
        self._catalog, self._provider_owner, self._closed = None, None, False
        if preparation is not None:
            self._restore(preparation)

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    def prepare(self, *, workspace_limit_bytes=32 << 20):
        if self._closed:
            raise TaichiRuntimeError("Contraction operation has been closed")
        if self._catalog is not None:
            return self.preparation_artifact()
        if (
            isinstance(workspace_limit_bytes, bool)
            or not isinstance(workspace_limit_bytes, int)
            or not 0 < workspace_limit_bytes < 2**64
        ):
            raise ValueError(
                "Contraction workspace limit must be a positive uint64 (legacy adapter zero means estimated workspace)"
            )
        if not hasattr(core, "_CudaCutensorCapturePlan"):
            raise TaichiRuntimeError(
                "Contraction recipes require typed native capture support"
            )
        provider = CutensorProvider()
        choices, unavailable = {}, []
        begin = time.perf_counter()
        baseline = None
        try:
            for flow in _dataflows(self.semantics):
                config = {
                    **flow,
                    "workspace_limit_bytes": workspace_limit_bytes,
                    "vendor_plan_policy": _POLICY,
                    "submission": "enclosing_graph",
                }
                try:
                    plan = _make_plan(
                        provider, self.semantics, config, description=True
                    )
                except TaichiRuntimeError as error:
                    if baseline is None or not any(
                        status in str(error)
                        for status in (
                            "CUTENSOR_STATUS_NOT_SUPPORTED",
                            "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE",
                        )
                    ):
                        raise
                    unavailable.append({"dataflow": flow, "reason": str(error)})
                    continue
                try:
                    config["workspace_bytes"] = plan.workspace_required_bytes
                    key = "contraction:" + _digest(config)
                    choices[key] = config
                    baseline = baseline or key
                finally:
                    plan.close()
            artifact = dict(
                schema=_SCHEMA,
                semantics=self.semantics,
                component=_component(provider),
                device=_current_cuda_device_scope(),
                baseline=baseline,
                choices=choices,
                preparation=dict(
                    origin="current_process_descriptors_only",
                    host_seconds=time.perf_counter() - begin,
                    vendor_library_path=provider.identity["vendor_library"],
                ),
                unavailable=unavailable,
            )
            self._catalog, self._provider_owner = (
                _ContractionCatalog(artifact, provider),
                provider,
            )
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
            or artifact.get("baseline") not in artifact.get("choices", {})
        ):
            raise ValueError("Contraction preparation semantic contract drifted")
        flows = {_canonical_json(flow) for flow in _dataflows(self.semantics)}
        for key, config in artifact["choices"].items():
            if (
                set(config)
                != {
                    "permutations",
                    "epilogue",
                    "workspace_limit_bytes",
                    "workspace_bytes",
                    "vendor_plan_policy",
                    "submission",
                }
                or _canonical_json({k: config[k] for k in ("permutations", "epilogue")})
                not in flows
            ):
                raise ValueError("Contraction preparation dataflow drifted")
            limit, actual = config["workspace_limit_bytes"], config["workspace_bytes"]
            if (
                any(
                    isinstance(x, bool) or not isinstance(x, int)
                    for x in (limit, actual)
                )
                or not 0 <= actual <= limit < 2**64
                or limit == 0
                or config["vendor_plan_policy"] != _POLICY
                or config["submission"] != "enclosing_graph"
            ):
                raise ValueError(
                    "Contraction preparation resource/plan contract drifted"
                )
            if key != "contraction:" + _digest(config):
                raise ValueError("Contraction preparation choice identity drifted")
        baseline = artifact["choices"][artifact["baseline"]]
        if _packing(self.semantics, baseline) or baseline["epilogue"] != "vendor":
            raise ValueError("Contraction baseline dataflow drifted")
        if artifact.get("device") != _current_cuda_device_scope():
            raise ValueError("Contraction preparation device contract drifted")
        provider = CutensorProvider()
        try:
            if artifact.get("component") != _component(provider):
                raise ValueError("Contraction preparation provider component drifted")
        except BaseException:
            provider.close()
            raise
        artifact["preparation"] = {
            **artifact.get("preparation", {}),
            "origin": "imported_preparation_not_current_measurement",
        }
        self._catalog, self._provider_owner = (
            _ContractionCatalog(artifact, provider),
            provider,
        )

    def preparation_artifact(self):
        if self._catalog is None:
            raise TaichiRuntimeError(
                "Call contraction.prepare() before constructing its Graph"
            )
        return self._catalog.facts

    def _graph_recipe_description(self):
        if self._closed:
            raise TaichiRuntimeError("Contraction operation has been closed")
        self.preparation_artifact()
        return native_recording_node(
            _DescriptionRecording(self._catalog),
            lifetime_leases=(self._catalog,),
            debug_info={"kind": "contraction_region"},
        ).compile()

    def compile(self):
        return self._graph_recipe_description()

    def close(self):
        self._closed, self._provider_owner = True, None


def record_contraction(a_shape, a_modes, b_shape, b_modes, output_modes, **kwargs):
    """Describe a complete contraction region; see ContractionOperation.

    prepare() does not measure candidates or allocate their GPU scratch. Search
    complete dataflows with hardware.tensor.ContractionRecipeProvider, alongside
    graph.default_recipe_providers(). No vendor/library route is a search axis.
    """
    return ContractionOperation(
        a_shape, a_modes, b_shape, b_modes, output_modes, **kwargs
    )


__all__ = ["record_contraction"]
