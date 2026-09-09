"""Mathematical FFT regions, independent of physical plan decomposition."""

import json
import math
import struct
import threading
import time
import weakref

from taichi_forge.graph._native import NativeGraphNode
from taichi_forge.graph._native import BackendCommandRecording, _CudaGraphCaptureRecipe
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._recipes.definition import _canonical_json, _digest
from taichi_forge.graph._recipes.deferred import FrozenNativeRecipeSource
from taichi_forge.hardware._fft import (
    CufftPlanND,
    CufftRecording,
    _CufftPlanBase,
    _positive_int,
    _positive_int_tuple,
)
from taichi_forge.hardware._admission import _current_cuda_device_scope
from taichi_forge.hardware._native_adapter import native_recording_node, validate_runtime_generation
from taichi_forge.hardware._retained import passive_dynamic_provider_scope
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


class _SeparableFftPlan(_CufftPlanBase):
    def __init__(self, dimensions, batch_count, *, cross_batch=False):
        self._initialize(
            dimensions, batch_count=batch_count, transform="c2c", _separable=True, _cross_batch=cross_batch
        )


class _FftDescription:

    @property
    def output_scale(self):
        return self.semantics.get("output_scale", 1.0)

    @property
    def semantics(self):
        return json.loads(self._semantics_json)

    @property
    def semantic_fingerprint(self):
        return _digest(self.semantics)

    @property
    def component(self):
        return json.loads(self._component_json)

    @property
    def input(self):
        return self.semantics["input"]

    @property
    def output(self):
        return self.semantics["output"]

    @property
    def direction(self):
        return self.semantics["direction"]

    def physical_config(self, strategy):
        if strategy not in (
            "whole_transform",
            "row_batch_column_inplace",
            "row_batch_cross_batch_columns",
            "whole_transform_store_scale",
        ):
            raise ValueError("Unknown FFT physical strategy")
        config = {
            "strategy": strategy,
            "workspace_lifetime": "completion_retained_plan_generation",
            "submission": "enclosing_graph",
            "stream": "runtime_ordered",
            "intermediate_dense_bytes": 0,
            "phases": (
                ("whole_transform",)
                if strategy in ("whole_transform", "whole_transform_store_scale")
                else (
                    ("all_rows_out_of_place", "columns_in_place_across_batches")
                    if strategy == "row_batch_cross_batch_columns"
                    else ("all_rows_out_of_place", "columns_in_place_per_batch")
                )
            ),
        }
        if self.output_scale != 1.0:
            config["output_scale"] = self.output_scale
            config["postprocess"] = (
                "cufft_lto_store_scale_v1"
                if strategy == "whole_transform_store_scale"
                else "forge_f32_scale_kernel_v1"
            )
            if strategy != "whole_transform_store_scale":
                config["phases"] += ("separate_output_scale",)
        return config

    def preparation_report(self):
        return json.loads(json.dumps(self._preparation))

    def preparation_artifact(self):
        """JSON-safe expected plan facts for explicit, plan-free reconstruction.

        This is not a binary cache or a fresh measurement. The selected plan's
        component and workspace are verified when it is actually materialized.
        """
        return {
            "schema": "taichi_forge.fft_preparation.v1",
            "semantics": self.semantics,
            "component": self.component,
            "device": json.loads(self._device_json),
            "plans": self.preparation_report(),
        }


class _FftPlanCatalog(_FftDescription):
    """Preparation facts and weak plan lookup, never an owner of search plans.

    Recordings own their individual plans. Reacquisition is a materialization
    operation; neither this catalog nor its lock participates in Graph replay.
    """

    def __init__(self, operation, baseline=None):
        self._semantics_json = operation._semantics_json
        self._component_json = operation._component_json
        self._preparation = operation._preparation
        self._device_json = operation._device_json
        self._preparation_origin = operation._preparation_origin
        self._restoration = {}
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_generation = int(impl.runtime_generation())
        self._plans = weakref.WeakValueDictionary()
        if baseline is not None:
            self._plans["whole_transform"] = baseline
        self._lock = threading.RLock()

    def _recording(self, strategy):
        with self._lock:
            validate_runtime_generation(self, "FFT recipe catalog belongs to a retired runtime")
            if strategy not in self._preparation:
                raise TaichiRuntimeError("FFT physical plan has not been prepared")
            plan = self._plans.get(strategy)
            if plan is None or plan.closed:
                started = time.perf_counter()
                semantics = self.semantics
                if strategy == "whole_transform":
                    plan = CufftPlanND(semantics["dimensions"], batch_count=semantics["batch_count"])
                elif strategy in ("row_batch_column_inplace", "row_batch_cross_batch_columns"):
                    plan = _SeparableFftPlan(
                        semantics["dimensions"],
                        semantics["batch_count"],
                        cross_batch=strategy == "row_batch_cross_batch_columns",
                    )
                elif strategy == "whole_transform_store_scale":
                    from taichi_forge.hardware._fft_lto import _CufftStoreScalePlan

                    plan = _CufftStoreScalePlan.prepare(
                        semantics["dimensions"],
                        semantics["batch_count"],
                        self.output_scale,
                        expected=self._preparation[strategy]["callback"],
                    )
                else:
                    raise ValueError("Unknown FFT physical strategy")
                # Rebuild only the requested plan, preserving the frozen facts.
                # A changed provider/workspace must not silently reuse evidence.
                component = plan._retained_identity.to_dict()["provider_scope"]
                if (
                    component != self.component
                    or plan._workspace_bytes != self._preparation[strategy]["workspace_bytes"]
                ):
                    plan.close()
                    raise TaichiRuntimeError("Recreated FFT plan differs from its prepared component or workspace")
                self._plans[strategy] = plan
                self._restoration[strategy] = {
                    "measurement_scope": (
                        "host_elapsed_for_selected_fft_callback_compile_and_plan_recreation"
                        if strategy == "whole_transform_store_scale"
                        else "host_elapsed_for_selected_fft_plan_recreation"
                    ),
                    "host_setup_seconds": time.perf_counter() - started,
                    "workspace_bytes": plan._workspace_bytes,
                    "unselected_plans_created": 0,
                }
            plan._validate_lifetime()
            return _FftRecording(self, strategy, plan)

    def append(self, builder, strategy, admission):
        """Expand this one mathematical region using the existing provider seam."""
        builder._append_native(self._recording(strategy), admission=admission)
        if self.output_scale != 1.0 and strategy != "whole_transform_store_scale":
            from taichi_forge.graph._graph import Arg, ArgKind
            from taichi_forge.hardware._fft_kernels import output_scale_kernel
            from taichi_forge.types.primitive_types import f32

            semantics = self.semantics
            rank = 4 if semantics["batch_count"] > 1 else 3
            builder.dispatch(
                output_scale_kernel(
                    tuple(semantics["dimensions"]),
                    semantics["batch_count"],
                    self.output_scale,
                ),
                Arg(ArgKind.NDARRAY, self.output, f32, ndim=rank),
            )


class _FrozenFftSource(FrozenNativeRecipeSource):
    _graph_binding_frame_capture_safe = True

    def __init__(self, catalog, strategy):
        self._graph_fft_source = catalog
        self._strategy = strategy

    @property
    def _recording(self):
        # Provider discovery uses the same descriptive source as live records.
        return self

    def materialize(self):
        return self._graph_fft_source._recording(self._strategy)._as_graph_native_node().compile()

    def append_to_graph(self, builder, *, admission):
        self._graph_fft_source.append(builder, self._strategy, admission)


class _FftCaptureDescription(_CudaGraphCaptureRecipe):
    kind = "cufft_fixed_plan"

    def __init__(self, source):
        self._source = source

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.types.primitive_types import f32

        semantics = self._source.semantics
        rank = 4 if semantics["batch_count"] > 1 else 3
        arguments = [Arg(ArgKind.NDARRAY, name, f32, ndim=rank) for name in (self._source.input, self._source.output)]
        append = getattr(builder, "_dispatch_cuda_capture_description", None)
        if append is None:
            raise TaichiRuntimeError("Plan-free FFT restoration is unavailable in this native runtime")
        append(program, self.kind, arguments)


class _FftDescriptionRecording(BackendCommandRecording):
    """A non-executable, provider-owned baseline description for Graph freeze."""

    def __init__(self, source):
        super().__init__(
            backend="cuda",
            binding_names=(source.input, source.output),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_graph_fft_source", source)
        object.__setattr__(self, "_graph_semantic_fingerprint", source.semantic_fingerprint)
        object.__setattr__(self, "_graph_physical_plan_id", _fft_physical_id(source, "whole_transform"))
        object.__setattr__(self, "_cuda_capture_recipe", _FftCaptureDescription(source))

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self._graph_fft_source.input, GraphAccess.READ),
            ResourceEffect(self._graph_fft_source.output, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        raise TaichiRuntimeError("An FFT description must be materialized before execution")

    def _freeze_graph_recipe_source(self):
        return _FrozenFftSource(self._graph_fft_source, "whole_transform")


def _fft_physical_id(source, strategy):
    return "fft-physical:" + _digest(
        {
            "semantics": source.semantic_fingerprint,
            "config": source.physical_config(strategy),
            "component": source.component,
            **(
                {"callback": source.preparation_report()[strategy]["callback"]}
                if strategy == "whole_transform_store_scale"
                else {}
            ),
        }
    )


class _FftRecording(CufftRecording):
    _graph_binding_frame_capture_safe = True

    def __init__(self, source, strategy, plan):
        super().__init__(
            plan,
            direction=source.direction,
            input=source.input,
            output=source.output,
        )
        object.__setattr__(self, "_graph_fft_source", source)
        object.__setattr__(self, "_graph_fft_strategy", strategy)
        object.__setattr__(self, "_graph_semantic_fingerprint", source.semantic_fingerprint)
        object.__setattr__(
            self,
            "_graph_physical_plan_id",
            _fft_physical_id(source, strategy),
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.plan, item),
            debug_info={"kind": "fft_transform"},
        )

    def _freeze_graph_recipe_source(self):
        return _FrozenFftSource(self._graph_fft_source, self._graph_fft_strategy)


class FftOperation(_FftDescription, NativeGraphNode):
    """A compact batched 2D complex-f32 transform with explicit tolerances.

    Input and output are distinct scalar f32 arrays shaped (H, W, 2), or
    (batch, H, W, 2) when batch > 1. The last axis is [real, imaginary].
    By default both directions are unnormalized: inverse(forward(x)) = H * W * x.
    Explicit output_scale is rounded to finite f32 and applied after the FFT;
    output_scale=1/(H*W) therefore describes a normalized inverse transform.
    Only CUDA is currently implemented. Values/qualification are caller-owned.
    """

    def __init__(
        self,
        dimensions,
        *,
        batch_count=1,
        direction="forward",
        input="input",
        output="output",
        absolute_tolerance,
        relative_tolerance,
        preparation=None,
        output_scale=1.0,
    ):
        dimensions = _positive_int_tuple(dimensions, "FFT dimensions")
        batch_count = _positive_int(batch_count, "FFT batch_count")
        if len(dimensions) != 2:
            raise ValueError("Graph FFT currently requires exactly two transform dimensions")
        if direction not in ("forward", "inverse"):
            raise ValueError("Graph FFT direction must be forward or inverse")
        if any(not isinstance(name, str) or not name for name in (input, output)) or input == output:
            raise ValueError("Graph FFT needs two distinct nonempty binding names")
        tolerances = []
        for value in (absolute_tolerance, relative_tolerance):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("Graph FFT tolerances must be finite nonnegative numbers")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Graph FFT tolerances must be finite and nonnegative")
            tolerances.append(value)
        if not any(tolerances):
            raise ValueError("Graph FFT needs positive tolerance; bitwise reproducibility is not promised")
        if isinstance(output_scale, bool) or not isinstance(output_scale, (int, float)):
            raise TypeError("Graph FFT output_scale must be a finite f32 number")
        try:
            output_scale = struct.unpack("f", struct.pack("f", output_scale))[0]
        except (OverflowError, struct.error) as exc:
            raise ValueError(
                "Graph FFT output_scale must be representable as finite f32"
            ) from exc
        if not math.isfinite(output_scale):
            raise ValueError("Graph FFT output_scale must be finite")
        self._semantics_json = _canonical_json(
            {
                "operation": "fft_transform",
                "schema": 1,
                "dimensions": dimensions,
                "batch_count": batch_count,
                "input": input,
                "output": output,
                "direction": direction,
                "normalization": "none",
                **({"output_scale": output_scale} if output_scale != 1.0 else {}),
                "transform": "complex_to_complex",
                "layout": "compact_row_major_interleaved_f32",
                "numerical_contract": {
                    "input_dtype": "complex_f32",
                    "output_dtype": "complex_f32",
                    "accumulation": "f32",
                    "absolute_tolerance": tolerances[0],
                    "relative_tolerance": tolerances[1],
                    "special_values": "finite_inputs_only",
                    "bitwise_reproducibility": False,
                },
            }
        )
        self._plans, self._preparation = {}, {}
        self._closed = False
        self._device_json = _canonical_json(_current_cuda_device_scope())
        self._preparation_origin = "current_process_plan_creation"
        if preparation is not None:
            self._restore_preparation(preparation)
            self._catalog = _FftPlanCatalog(self)
            return
        started = time.perf_counter()
        plan = CufftPlanND(dimensions, batch_count=batch_count)
        self._plans["whole_transform"] = plan
        self._component_json = _canonical_json(plan._retained_identity.to_dict()["provider_scope"])
        self._preparation["whole_transform"] = {
            "workspace_bytes": plan._workspace_bytes,
            "host_setup_seconds": time.perf_counter() - started,
        }
        self._catalog = _FftPlanCatalog(self, plan)

    def _restore_preparation(self, preparation):
        # Copy and validate data only. No executable, Python object or vendor
        # plan is deserialized, and no missing baseline is silently prepared.
        data = json.loads(_canonical_json(preparation))
        if not isinstance(data, dict) or data.get("schema") != "taichi_forge.fft_preparation.v1":
            raise ValueError("Invalid FFT preparation schema")
        if _canonical_json(data.get("semantics")) != self._semantics_json:
            raise ValueError("FFT preparation semantic contract drifted")
        if _canonical_json(data.get("device")) != self._device_json:
            raise ValueError("FFT preparation device contract drifted")
        plans = data.get("plans")
        if not isinstance(plans, dict) or "whole_transform" not in plans:
            raise ValueError("FFT preparation requires baseline plan facts")
        for strategy, info in plans.items():
            self.physical_config(strategy)
            if not isinstance(info, dict):
                raise ValueError("Invalid FFT preparation plan facts")
            workspace = info.get("workspace_bytes")
            elapsed = info.get("host_setup_seconds")
            if isinstance(workspace, bool) or not isinstance(workspace, int) or workspace < 0:
                raise ValueError("FFT preparation workspace must be a nonnegative integer")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("FFT preparation elapsed time must be finite and nonnegative")
            if strategy == "whole_transform_store_scale":
                callback = info.get("callback")
                if (
                    self.output_scale == 1.0
                    or not isinstance(callback, dict)
                    or callback.get("schema") != "cufft-store-scale-lto-v1"
                ):
                    raise ValueError("Invalid FFT store-scaling preparation")
        component = passive_dynamic_provider_scope("cufft", "cufft-plan-many-dynamic-symbols-v3")
        if not component["library_candidate"]:
            from taichi_forge.hardware._external_providers import probe_external_provider

            status = probe_external_provider("cufft")
            if status.get("discovery") != "available":
                raise TaichiRuntimeError("FFT preparation requires an available CUDA cuFFT provider")
            component = {
                "provider_abi": status["provider_abi"],
                "provider_version": status["provider_version"],
                "library_candidate": status["native_facts"]["library_candidate"],
                "provider_binary_identity": None,
            }
        if component != data.get("component"):
            raise ValueError("FFT preparation provider component drifted")
        self._component_json = _canonical_json(component)
        self._preparation = plans
        self._preparation_origin = "imported_expected_facts_not_current_measurement"

    def _graph_recipe_description(self):
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        if (
            self._preparation_origin == "current_process_plan_creation"
            and self.output_scale == 1.0
        ):
            return None
        validate_runtime_generation(self._catalog, "FFT recipe catalog belongs to a retired runtime")
        return native_recording_node(
            _FftDescriptionRecording(self._catalog),
            lifetime_leases=lambda item: (item._graph_fft_source,),
            debug_info={"kind": "fft_transform"},
        ).compile()

    def prepare(
        self, *, lto_callbacks=False, nvrtc_library=None, nvjitlink_library=None
    ):
        """Prepare available decompositions before search, without executing FFT.

        The operation retains prepared plans until close. Graphs separately
        retain only plans they use; their lifetime does not end with this owner.
        No replay checks or per-invocation reconstruction are added.

        lto_callbacks=True additionally prepares an optional cuFFT store-scale
        fusion for non-identity output_scale. Supply absolute nvrtc_library and
        nvjitlink_library paths to compatible external runtimes; this does not
        select the candidate, change plain FFT support or install a compiler.
        A failed preparation raises without substituting unscaled FFT math.
        """
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        if not isinstance(lto_callbacks, bool):
            raise TypeError("lto_callbacks must be bool")
        if not lto_callbacks and (
            nvrtc_library is not None or nvjitlink_library is not None
        ):
            raise ValueError("JIT libraries require explicit lto_callbacks=True")
        if lto_callbacks and self.output_scale == 1.0:
            raise ValueError(
                "A store-scaling candidate requires non-identity output_scale"
            )
        if self._preparation_origin != "current_process_plan_creation":
            for strategy in self._preparation:
                self._plans[strategy] = self._catalog._recording(strategy).plan
            return self.preparation_report()
        baseline = self._plans["whole_transform"]
        baseline._validate_lifetime()
        info = baseline._runtime_prog._cuda_cufft_plan_memory_statistics(baseline._handle)
        if "separable" not in info:
            raise TaichiRuntimeError("Separable FFT plans are unavailable in this native runtime")
        strategies = ["row_batch_column_inplace"]
        if hasattr(baseline._runtime_prog, "_create_cuda_cufft_cross_batch_plan") and (
            self.semantics["batch_count"],
            self.semantics["dimensions"][1],
        ) != (1, 1):
            strategies.append("row_batch_cross_batch_columns")
        for strategy in strategies:
            if strategy not in self._plans:
                started = time.perf_counter()
                plan = _SeparableFftPlan(
                    tuple(self.semantics["dimensions"]),
                    self.semantics["batch_count"],
                    cross_batch=strategy == "row_batch_cross_batch_columns",
                )
                self._plans[strategy] = plan
                self._preparation[strategy] = {
                    "workspace_bytes": plan._workspace_bytes,
                    "host_setup_seconds": time.perf_counter() - started,
                }
                self._catalog._plans[strategy] = plan
        if lto_callbacks and "whole_transform_store_scale" not in self._plans:
            from taichi_forge.hardware._fft_lto import _CufftStoreScalePlan

            started = time.perf_counter()
            plan = _CufftStoreScalePlan.prepare(
                tuple(self.semantics["dimensions"]),
                self.semantics["batch_count"],
                self.output_scale,
                nvrtc_library=nvrtc_library,
                nvjitlink_library=nvjitlink_library,
            )
            strategy = "whole_transform_store_scale"
            self._plans[strategy] = plan
            self._preparation[strategy] = {
                "workspace_bytes": plan._workspace_bytes,
                "host_setup_seconds": time.perf_counter() - started,
                "callback": plan.callback_facts,
            }
            self._catalog._plans[strategy] = plan
        return self.preparation_report()

    def _recording(self, strategy):
        if self._closed:
            raise TaichiRuntimeError("FFT operation has been closed")
        return self._catalog._recording(strategy)

    def compile(self):
        if self.output_scale != 1.0:
            return self._graph_recipe_description()
        return self._recording("whole_transform")._as_graph_native_node().compile()

    def close(self):
        """Release this preparation owner; existing Graph plan leases stay live.

        Frozen definitions retain descriptions, not FFT execution plans. Live
        Graphs and in-flight native submissions keep their independent leases.
        """
        self._closed = True
        self._plans.clear()


def record_fft(
    dimensions,
    *,
    batch_count=1,
    direction="forward",
    input="input",
    output="output",
    absolute_tolerance,
    relative_tolerance,
    preparation=None,
    output_scale=1.0,
):
    """Describe a 2D C2C FFT with optional explicit output_scale (default 1).

    Call operation.prepare() before using the explicit FftRecipeProvider.
    Alternatively, pass a prior operation.preparation_artifact() as preparation
    to rebuild a Graph without creating FFT plans. Do not call prepare() on that
    path unless intentionally preparing all imported candidates again.
    Physical decomposition remains provider-owned, never a CompileIQ raw axis.
    """
    return FftOperation(
        dimensions,
        batch_count=batch_count,
        direction=direction,
        input=input,
        output=output,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        preparation=preparation,
        output_scale=output_scale,
    )


__all__ = ["record_fft"]
