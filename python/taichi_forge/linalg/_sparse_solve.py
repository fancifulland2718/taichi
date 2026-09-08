"""Frozen sparse-solve regions and provider-owned numerical lifecycles."""

import hashlib
import json
import math
import time

import numpy as np

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
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from taichi_forge.hardware._cudss_config import canonical_configuration, configured_plan
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_runtime_generation,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg.sparse_matrix import (
    SparsePattern,
    _require_current_scalar_ndarray,
)
from taichi_forge.types import f32

_SCHEMA = "taichi_forge.sparse_solve_preparation.v1"


def _names(semantics):
    return (() if semantics["values"] is None else (semantics["values"],)) + tuple(
        name for pair in semantics["rhs_pairs"] for name in pair
    )


def _component(library_path):
    from taichi_forge.hardware._cudss import resolve_cudss_provider

    resolved = resolve_cudss_provider(library_path)
    vendor_hash = _binary_sha256(resolved.runtime_library_path)
    if not resolved.adapter_binary_sha256 or not vendor_hash:
        raise TaichiRuntimeError("Sparse solve component identity is unavailable")
    return (
        dict(
            provider="cudss",
            execution_abi=1,
            configuration_abi=1,
            allocator_abi=1,
            version=resolved.provider_version,
            header_version=resolved.provider_header_version,
            adapter_sha256=resolved.adapter_binary_sha256,
            library_sha256=vendor_hash,
            restoration="rebuild_analysis_and_private_factors_not_vendor_binary_state",
        ),
        resolved.runtime_library_path,
    )


def _plan(matrix, semantics, configuration, library_path):
    return configured_plan(
        matrix,
        configuration,
        matrix_type=semantics["matrix_type"],
        matrix_view=semantics["matrix_view"],
        library_path=library_path,
        _graph_owned=True,
    )


def _resources(plan):
    facts = plan._configuration_report()["graph_allocator"]
    return dict(
        snapshot_bytes=facts["snapshot_bytes"],
        vendor_payload_bytes=facts["live_bytes"] - facts["snapshot_bytes"],
    )


class _SparseSolveCatalog:
    def __init__(self, facts, matrix, library_path):
        self._facts_json = _canonical_json(facts)
        self._matrix, self._library_path = matrix, library_path
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
        facts = self.facts
        return "sparse-solve-physical:" + _digest(
            dict(
                semantics=self.semantic_fingerprint,
                configuration=self.physical_config(key),
                component=facts["component"],
                preparation_input=facts["initial_values_sha256"],
            )
        )

    def append(self, builder, key, admission):
        validate_runtime_generation(
            self, "Sparse solve catalog belongs to a retired runtime"
        )
        component, path = _component(self._library_path)
        if component != self.facts["component"]:
            raise TaichiRuntimeError("Sparse solve provider component drifted")
        semantics, config = self.semantics, self.physical_config(key)
        plan = _plan(self._matrix, semantics, config["configuration"], path)
        try:
            if _resources(plan) != config["resources"]:
                raise TaichiRuntimeError(
                    "Sparse solve prepared resource contract drifted"
                )
            if (
                plan._configuration_report()["capture_parameter_storage"]
                != config["capture_parameter_storage"]
            ):
                raise TaichiRuntimeError(
                    "Sparse solve capture parameter storage drifted"
                )
            recording = _SolveRecording(self, key, plan)
            builder.append_native(recording, admission=admission)
        except BaseException:
            # If append retained a command, its owning transaction retires it.
            # Otherwise close the newly constructed owner immediately.
            if not plan._capture_leases:
                plan.close()
            raise


class _SolveRecording(CudssCaptureRecording):
    def __init__(self, source, key, plan):
        semantics, config = source.semantics, source.physical_config(key)
        super().__init__(
            plan,
            phase=config["numeric_phase"],
            values=semantics["values"],
            _rhs_pairs=semantics["rhs_pairs"],
        )
        object.__setattr__(self, "_graph_sparse_solve_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(self, "_graph_physical_plan_id", source.physical_id(key))

    def _graph_provider_memory_report(self):
        from taichi_forge.hardware._memory import (
            HardwareMemoryComponent,
            make_memory_report,
        )

        report = super()._graph_provider_memory_report()
        # Source-owned immutable numeric values are distinct from the private
        # vendor snapshot. Caller-owned pattern/bindings are not workspace.
        return make_memory_report(
            report.provider,
            "cuda",
            (
                *report.components,
                HardwareMemoryComponent(
                    "catalog_numeric_snapshot",
                    4 * self.plan._nnz,
                    True,
                    "provider_generation",
                    "provider",
                ),
            ),
            lifecycle_state="ready",
            ownership_scope="plan_generation",
        )


class _FrozenSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = (
        CudssCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        self._graph_sparse_solve_source = source

    @property
    def _recording(self):
        return self

    def append_to_graph(self, builder, *, admission):
        source = self._graph_sparse_solve_source
        source.append(builder, source.baseline, admission)


class _DescriptionRecipe(_CudaGraphCaptureRecipe):
    kind = "semantic_sparse_solve_f32"

    def __init__(self, source):
        self.source = source

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        builder._dispatch_cuda_capture_description(
            program,
            self.kind,
            [
                Arg(ArgKind.NDARRAY, name, f32, ndim=1)
                for name in _names(self.source.semantics)
            ],
        )


class _DescriptionRecording(BackendCommandRecording):
    _graph_binding_frame_capture_safe = (
        CudssCaptureRecording._graph_binding_frame_capture_safe
    )

    def __init__(self, source):
        super().__init__(
            backend="cuda",
            binding_names=_names(source.semantics),
            command_count=1,
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_graph_sparse_solve_source", source)
        object.__setattr__(
            self, "_graph_semantic_fingerprint", source.semantic_fingerprint
        )
        object.__setattr__(
            self, "_graph_physical_plan_id", source.physical_id(source.baseline)
        )
        object.__setattr__(self, "_cuda_capture_recipe", _DescriptionRecipe(source))

    @property
    def resource_effects(self):
        outputs = {p[1] for p in self._graph_sparse_solve_source.semantics["rhs_pairs"]}
        return tuple(
            ResourceEffect(n, GraphAccess.WRITE if n in outputs else GraphAccess.READ)
            for n in self.binding_names
        )

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "Sparse solve descriptions must be frozen and materialized"
        )

    def _freeze_graph_recipe_source(self):
        return _FrozenSource(self._graph_sparse_solve_source)


class SparseSolveOperation(NativeGraphNode):
    """Solve A x_i = b_i with a frozen CSR pattern and shared factors.

    ``initial_values`` is copied once into private preparation storage. With
    ``values=None`` it defines the fixed matrix; otherwise ``values`` names a
    current-values Graph input in the pattern's CSR order. All RHS/output
    pairs are compact f32 vectors and mutually disjoint. Outputs may feed a
    later Graph invocation, but are never modified by prepare or bind.

    Matrix structure/type and caller numerical tolerances are semantic inputs,
    not optimizer choices. Residual evaluation belongs to the caller, outside
    steady replay. The current implementation requires a CUDA cuDSS runtime.
    """

    def __init__(
        self,
        pattern,
        initial_values,
        *,
        values="matrix_values",
        rhs_pairs=(("rhs", "solution"),),
        matrix_type="general",
        matrix_view=None,
        absolute_tolerance,
        relative_tolerance,
        library_path=None,
        preparation=None
    ):
        if not isinstance(pattern, SparsePattern):
            raise TypeError("Sparse solve requires an immutable SparsePattern")
        pattern._ensure_valid()
        if pattern.storage_format != "csr" or pattern.shape[0] != pattern.shape[1]:
            raise ValueError("Sparse solve requires a square CSR pattern")
        initial_values = _require_current_scalar_ndarray(
            initial_values, "Sparse solve initial_values", f32, one_dimensional=True
        )
        if initial_values.shape != (pattern.num_nonzeros,):
            raise ValueError("Sparse solve initial values do not match the pattern")
        if matrix_type not in ("general", "symmetric", "spd"):
            raise ValueError(
                "Sparse solve matrix_type must be general, symmetric or spd"
            )
        if matrix_view is None:
            matrix_view = "full" if matrix_type == "general" else "lower"
        if matrix_view not in ("full", "lower", "upper") or (
            matrix_type == "general" and matrix_view != "full"
        ):
            raise ValueError("Sparse solve matrix view is incompatible with its type")
        rhs_pairs = tuple(tuple(p) for p in rhs_pairs)
        if not rhs_pairs or any(len(p) != 2 for p in rhs_pairs):
            raise ValueError("Sparse solve requires nonempty RHS/output binding pairs")
        names = (() if values is None else (values,)) + tuple(
            n for p in rhs_pairs for n in p
        )
        if any(not isinstance(n, str) or not n for n in names) or len(
            set(names)
        ) != len(names):
            raise ValueError(
                "Sparse solve binding names must be distinct nonempty strings"
            )
        tolerances = (absolute_tolerance, relative_tolerance)
        if (
            any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x < 0
                for x in tolerances
            )
            or max(tolerances) == 0
        ):
            raise ValueError(
                "Sparse solve requires finite nonnegative, nonzero residual tolerances"
            )
        # One cold read establishes content identity and catches non-finite
        # preparation data. No input scan is installed on Graph replay.
        host_values = initial_values.to_numpy()
        if not np.isfinite(host_values).all():
            raise ValueError("Sparse solve initial values must be finite")
        self._initial_values_sha256 = hashlib.sha256(host_values.tobytes()).hexdigest()
        self._matrix = pattern.matrix(initial_values)
        self._semantics_json = _canonical_json(
            dict(
                operation="shared_pattern_sparse_solve",
                shape=pattern.shape,
                nonzeros=pattern.num_nonzeros,
                topology_fingerprint=self._matrix._topology_fingerprint,
                storage="csr_i32_f32",
                matrix_type=matrix_type,
                matrix_view=matrix_view,
                values=values,
                rhs_pairs=rhs_pairs,
                matrix_lifetime=(
                    "fixed_snapshot"
                    if values is None
                    else "current_values_each_invocation"
                ),
                fixed_values_sha256=(
                    self._initial_values_sha256 if values is None else None
                ),
                numerical_contract=dict(
                    finite_inputs_only=True,
                    nonsingular=True,
                    bitwise_reproducible=False,
                    residual="each_rhs_inf_norm(Ax-b) <= absolute_tolerance + relative_tolerance * inf_norm(b)",
                    absolute_tolerance=float(absolute_tolerance),
                    relative_tolerance=float(relative_tolerance),
                    validation="caller_evaluator_not_replay",
                ),
                output_alias="disjoint_from_all_bindings",
            )
        )
        self._catalog, self._closed, self._library_path = None, False, library_path
        if preparation is not None:
            self._restore(preparation)

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    def prepare(self, *, max_plans=4):
        """Freeze bounded ordering/lifecycle plans; measurements remain separate."""
        if self._closed:
            raise TaichiRuntimeError("Sparse solve operation has been closed")
        if self._catalog is not None:
            return self.preparation_artifact()
        if (
            isinstance(max_plans, bool)
            or not isinstance(max_plans, int)
            or max_plans < 1
        ):
            raise ValueError(
                "Sparse solve plan preparation budget must be a positive integer"
            )
        if len(self.semantics["rhs_pairs"]) > 1 and not hasattr(
            core.GraphBuilder, "_dispatch_cuda_cudss_capture_group"
        ):
            raise TaichiRuntimeError(
                "Sparse solve requires native shared-factor group capture"
            )
        component, path = _component(self._library_path)
        start = time.perf_counter()
        semantics, choices, unavailable, baseline = self.semantics, {}, [], None
        orderings = ("default", "amd", "nested_dissection", "natural")[:max_plans]
        observations = []
        for reordering in orderings:
            configuration = dict(
                version=1,
                reordering=reordering,
                solve="default" if reordering == "default" else "general",
            )
            try:
                plan = _plan(self._matrix, semantics, configuration, path)
            except TaichiRuntimeError as error:
                if baseline is None:
                    raise
                unavailable.append(dict(configuration=configuration, reason=str(error)))
                continue
            try:
                observations.append(plan._configuration_report())
                phases = (
                    ("solve",)
                    if semantics["values"] is None
                    else ("factor_solve", "refactor_solve")
                )
                for phase in phases:
                    config = dict(
                        configuration=configuration,
                        numeric_phase=phase,
                        resources=_resources(plan),
                        analysis="once_per_materialization",
                        factor_scope="shared_by_all_rhs",
                        submission="ordered_capture_group",
                        capture_parameter_storage=plan._configuration_report()[
                            "capture_parameter_storage"
                        ],
                    )
                    key = "sparse-solve:" + _digest(config)
                    choices[key], baseline = config, baseline or key
            finally:
                plan.close()
        facts = dict(
            schema=_SCHEMA,
            semantics=semantics,
            component=component,
            device=_current_cuda_device_scope(),
            initial_values_sha256=self._initial_values_sha256,
            choices=choices,
            baseline=baseline,
            unavailable=unavailable,
            preparation=dict(
                origin="current_process_private_analysis_and_factor_warmup",
                host_seconds=time.perf_counter() - start,
                requested_orderings=orderings,
                enumeration_complete=len(orderings) == 4,
                observations=observations,
                vendor_library_path=path,
            ),
        )
        self._catalog = _SparseSolveCatalog(facts, self._matrix, path)
        return self.preparation_artifact()

    def _restore(self, preparation):
        facts = json.loads(_canonical_json(preparation))
        if (
            not isinstance(facts, dict)
            or facts.get("schema") != _SCHEMA
            or facts.get("semantics") != self.semantics
            or facts.get("initial_values_sha256") != self._initial_values_sha256
            or not isinstance(facts.get("choices"), dict)
            or facts.get("baseline") not in facts["choices"]
        ):
            raise ValueError("Sparse solve preparation semantic/input contract drifted")
        phases = (
            ("solve",)
            if self.semantics["values"] is None
            else ("factor_solve", "refactor_solve")
        )
        for key, config in facts["choices"].items():
            if set(config) != {
                "configuration",
                "numeric_phase",
                "resources",
                "analysis",
                "factor_scope",
                "submission",
                "capture_parameter_storage",
            }:
                raise ValueError("Sparse solve physical configuration fields drifted")
            canonical_configuration(config["configuration"])
            resources = config["resources"]
            if (
                not isinstance(resources, dict)
                or set(resources) != {"snapshot_bytes", "vendor_payload_bytes"}
                or any(
                    type(x) is not int or x < 0 or x >= 2**63
                    for x in resources.values()
                )
                or config["numeric_phase"] not in phases
                or config["analysis"] != "once_per_materialization"
                or config["factor_scope"] != "shared_by_all_rhs"
                or config["submission"] != "ordered_capture_group"
                or config["capture_parameter_storage"]
                not in ("device_resident_per_binding", "host_snapshot_per_binding")
                or key != "sparse-solve:" + _digest(config)
            ):
                raise ValueError(
                    "Sparse solve lifecycle/resource/choice contract drifted"
                )
        baseline = facts["choices"][facts["baseline"]]
        if (
            baseline["configuration"]
            != dict(version=1, reordering="default", solve="default")
            or baseline["numeric_phase"] != phases[0]
        ):
            raise ValueError("Sparse solve baseline contract drifted")
        if facts.get("device") != _current_cuda_device_scope():
            raise ValueError("Sparse solve preparation device contract drifted")
        component, path = _component(self._library_path)
        if facts.get("component") != component:
            raise ValueError("Sparse solve preparation component drifted")
        facts["preparation"] = {
            **facts.get("preparation", {}),
            "origin": "imported_preparation_not_current_measurement",
        }
        self._catalog = _SparseSolveCatalog(facts, self._matrix, path)

    def preparation_artifact(self):
        if self._catalog is None:
            raise TaichiRuntimeError(
                "Call sparse_solve.prepare() before constructing its Graph"
            )
        return self._catalog.facts

    def _graph_recipe_description(self):
        if self._closed:
            raise TaichiRuntimeError("Sparse solve operation has been closed")
        self.preparation_artifact()
        return native_recording_node(
            _DescriptionRecording(self._catalog),
            lifetime_leases=(self._catalog,),
            debug_info={"kind": "shared_pattern_sparse_solve_region"},
        ).compile()

    def compile(self):
        return self._graph_recipe_description()

    def close(self):
        self._closed, self._matrix, self._catalog = True, None, None


def record_sparse_solve(pattern, initial_values, **kwargs):
    """Describe shared-pattern A x_i = b_i; see SparseSolveOperation.

    prepare() freezes complete numerical lifecycles. Search with
    hardware.linalg.SparseSolveRecipeProvider; no library or raw policy axis.
    """
    return SparseSolveOperation(pattern, initial_values, **kwargs)


__all__ = ["record_sparse_solve"]
