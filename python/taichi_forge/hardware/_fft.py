"""Optional single-GPU cuFFT plan and execution provider."""

from dataclasses import dataclass

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording, _CudaGraphCaptureRecipe
from taichi_forge._hardware_telemetry import (
    hardware_provider_call,
    instrument_hardware_recording,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
    validate_exact_bindings,
    validate_runtime_generation,
)
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    RetainedExecutionContract,
    attach_retained_execution_contract,
    fixed_cost,
    make_retained_plan_identity,
    passive_dynamic_provider_scope,
    scale_cost,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


_DIRECTIONS = {"forward": -1, "inverse": 1}
_TRANSFORMS = {"c2c": 0, "r2c": 1, "c2r": 2}
_NATURAL_DIRECTIONS = {"c2c": "forward", "r2c": "forward", "c2r": "inverse"}


def _positive_int(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError(f"CUDA cuFFT {name} must be in [1, INT_MAX]")
    return value


def _positive_int_tuple(values, name):
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"CUDA cuFFT {name} must be an integer sequence") from exc
    if not result:
        raise ValueError(f"CUDA cuFFT {name} must not be empty")
    return tuple(_positive_int(value, name) for value in result)


def _product(values):
    result = 1
    for value in values:
        result *= value
    return result


def _layout_span(logical_dimensions, embed, stride):
    offset = 0
    for axis, logical in enumerate(logical_dimensions):
        if axis:
            offset *= embed[axis]
        offset += logical - 1
    return offset * stride + 1


@dataclass(frozen=True)
class CufftLayout:
    """Physical element layout for one cuFFT input or output batch."""

    embed: object = None
    stride: int = 1
    batch_distance: object = None

    def __post_init__(self):
        if self.embed is not None:
            object.__setattr__(
                self, "embed", _positive_int_tuple(self.embed, "layout embed")
            )
        object.__setattr__(self, "stride", _positive_int(self.stride, "layout stride"))
        if self.batch_distance is not None:
            object.__setattr__(
                self,
                "batch_distance",
                _positive_int(self.batch_distance, "layout batch_distance"),
            )


def _resolve_layout(layout, logical_dimensions, name):
    compact = layout is None
    if compact:
        layout = CufftLayout()
    if not isinstance(layout, CufftLayout):
        raise TypeError(f"CUDA cuFFT {name}_layout must be a CufftLayout")
    embed = logical_dimensions if layout.embed is None else layout.embed
    if len(embed) != len(logical_dimensions):
        raise ValueError(f"CUDA cuFFT {name} embed rank must match transform rank")
    if any(physical < logical for physical, logical in zip(embed, logical_dimensions)):
        raise ValueError(f"CUDA cuFFT {name} embed must cover every logical dimension")
    span = _layout_span(logical_dimensions, embed, layout.stride)
    distance = (
        _product(embed) * layout.stride
        if layout.batch_distance is None
        else layout.batch_distance
    )
    if distance < span:
        raise ValueError(
            f"CUDA cuFFT {name} batch_distance must not overlap transform storage"
        )
    normalized = CufftLayout(embed, layout.stride, distance)
    return normalized, span, compact


def _transform_value(transform):
    try:
        return _TRANSFORMS[transform]
    except (KeyError, TypeError) as exc:
        raise ValueError("CUDA cuFFT transform must be 'c2c', 'r2c', or 'c2r'") from exc


def _resolve_direction(transform, direction):
    if direction is None:
        direction = _NATURAL_DIRECTIONS[transform]
    try:
        direction_value = _DIRECTIONS[direction]
    except (KeyError, TypeError) as exc:
        raise ValueError("CUDA cuFFT direction must be 'forward' or 'inverse'") from exc
    expected = _NATURAL_DIRECTIONS[transform]
    if transform != "c2c" and direction != expected:
        raise ValueError(
            f"CUDA cuFFT {transform.upper()} requires direction={expected!r}"
        )
    return direction, direction_value


class _CufftCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cufft_fixed_plan"

    def __init__(self, plan, direction, input_name, output_name):
        self._plan = plan
        self._direction = direction
        self._input_name = input_name
        self._output_name = output_name

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind  # pylint: disable=C0415

        input_arg = Arg(
            ArgKind.NDARRAY,
            self._input_name,
            f32,
            ndim=len(self._plan.input_shape),
        )
        output_arg = Arg(
            ArgKind.NDARRAY,
            self._output_name,
            f32,
            ndim=len(self._plan.output_shape),
        )
        builder._dispatch_cuda_cufft_capture_recipe(
            self._plan._handle,  # pylint: disable=W0212
            program,
            input_arg,
            output_arg,
            self._direction,
            self._plan.input_storage_scalars,
            self._plan.output_storage_scalars,
        )


@instrument_hardware_recording("fft.transform.cufft")
class CufftRecording(BackendCommandRecording):
    """One out-of-place C2C, R2C, or C2R execution against a fixed plan."""

    def __init__(
        self,
        plan,
        *,
        direction=None,
        input="input",
        output="output",
    ):
        if not isinstance(plan, _CufftPlanBase):
            raise TypeError("CUDA cuFFT recording requires a cuFFT plan")
        direction, direction_value = _resolve_direction(plan.transform, direction)
        if any(not isinstance(name, str) or not name for name in (input, output)):
            raise ValueError("CUDA cuFFT binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuFFT input and output binding names must differ")
        super().__init__(
            backend="cuda",
            binding_names=(input, output),
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
            no_host_readback=True,
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "direction_value", direction_value)
        object.__setattr__(
            self,
            "output_scale",
            plan.inverse_scale if direction == "inverse" else 1.0,
        )
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "_cuda_capture_recipe",
            _CufftCaptureRecipe(plan, direction_value, input, output),
        )
        attach_retained_execution_contract(
            self,
            plan._retained_execution_contract,  # pylint: disable=W0212
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuFFT")
        self.validate_graph_lifetime()
        source = self.plan._validate_array(
            bindings[self.input], self.input, self.plan.input_shape
        )
        destination = self.plan._validate_array(
            bindings[self.output], self.output, self.plan.output_shape
        )
        if bindings[self.input] is bindings[self.output]:
            raise TaichiRuntimeError(
                "The first CUDA cuFFT slice requires distinct input and output arrays"
            )
        with hardware_provider_call("cufft"):
            self.plan._execute(source, destination, self.direction_value)

    def validate_graph_lifetime(self):
        self.plan._validate_lifetime()

    def memory_report(self):
        return self.plan.memory_report()

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.plan,),
            debug_info=lambda item: {
                "kind": f"cuda_cufft_{item.plan.transform}_{item.plan.rank}d",
                "dimensions": item.plan.dimensions,
                "batch_count": item.plan.batch_count,
                "transform": item.plan.transform,
                "direction": item.direction,
                "output_scale": item.output_scale,
                "input_layout": item.plan.input_layout,
                "output_layout": item.plan.output_layout,
            },
        )


class _CufftPlanBase:

    def _initialize(
        self,
        dimensions,
        *,
        batch_count,
        transform,
        input_layout=None,
        output_layout=None,
        _separable=False,
        _cross_batch=False,
        _store_callback=None,
    ):
        self.dimensions = _positive_int_tuple(dimensions, "dimensions")
        self.rank = len(self.dimensions)
        self.batch_count = _positive_int(batch_count, "batch_count")
        self.transform = transform
        self.transform_value = _transform_value(transform)
        self._separable = bool(_separable)
        self._cross_batch = bool(_cross_batch)
        if self._cross_batch and (not self._separable or self.rank != 2 or transform != "c2c"):
            raise ValueError("Cross-batch FFT requires a separable two-dimensional C2C plan")
        input_dimensions = list(self.dimensions)
        output_dimensions = list(self.dimensions)
        if transform == "c2r":
            input_dimensions[-1] = input_dimensions[-1] // 2 + 1
        if transform == "r2c":
            output_dimensions[-1] = output_dimensions[-1] // 2 + 1
        self.input_dimensions = tuple(input_dimensions)
        self.output_dimensions = tuple(output_dimensions)
        self.input_layout, input_span, input_compact = _resolve_layout(
            input_layout, self.input_dimensions, "input"
        )
        self.output_layout, output_span, output_compact = _resolve_layout(
            output_layout, self.output_dimensions, "output"
        )
        self._input_compact = input_compact
        self._output_compact = output_compact
        self.input_storage_scalars = (
            (self.batch_count - 1) * self.input_layout.batch_distance + input_span
        ) * (1 if transform == "r2c" else 2)
        self.output_storage_scalars = (
            (self.batch_count - 1) * self.output_layout.batch_distance + output_span
        ) * (1 if transform == "c2r" else 2)
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "CUDA cuFFT plans require an initialized Taichi runtime"
            )
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "A CUDA cuFFT plan requires the CUDA backend; the active backend is "
                f"{active_backend()}"
            )
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        with hardware_provider_call("cufft", failure_phase="provider_plan_failure"):
            if _store_callback is not None:
                if (
                    self.rank != 2
                    or transform != "c2c"
                    or not input_compact
                    or not output_compact
                    or self._separable
                ):
                    raise ValueError(
                        "LTO store callbacks require a compact whole 2D C2C plan"
                    )
                create = getattr(
                    program, "_create_cuda_cufft_store_callback_plan", None
                )
                if create is None:
                    raise TaichiRuntimeError(
                        "LTO store callbacks are unavailable in this native runtime"
                    )
                handle = create(
                    self.dimensions,
                    self.batch_count,
                    _store_callback.ir,
                    _store_callback.symbol,
                )
            elif self._cross_batch:
                if not input_compact or not output_compact:
                    raise ValueError("Cross-batch FFT requires compact input and output")
                create = getattr(program, "_create_cuda_cufft_cross_batch_plan", None)
                if create is None:
                    raise TaichiRuntimeError("Cross-batch FFT is unavailable in this native runtime")
                handle = create(self.dimensions, self.batch_count)
            elif self.rank == 1 and input_compact and output_compact:
                handle = program._create_cuda_cufft_plan_1d(
                    self.dimensions[0], self.batch_count, self.transform_value
                )
            else:
                arguments = (
                    self.dimensions,
                    self.input_layout.embed,
                    self.input_layout.stride,
                    self.input_layout.batch_distance,
                    self.output_layout.embed,
                    self.output_layout.stride,
                    self.output_layout.batch_distance,
                    self.batch_count,
                    self.transform_value,
                )
                handle = program._create_cuda_cufft_plan_many(
                    *arguments, *((True,) if self._separable else ())
                )
        self._handle = int(handle)
        self._workspace_bytes = int(
            program._cuda_cufft_plan_memory_statistics(self._handle)["workspace_bytes"]
        )
        provider_scope = passive_dynamic_provider_scope(
            "cufft", "cufft-plan-many-dynamic-symbols-v3"
        )
        self._retained_identity = make_retained_plan_identity(
            "fft.transform.cufft",
            "cufft",
            "cuda",
            provider_scope=provider_scope,
            problem_scope={
                "dimensions": self.dimensions,
                "batch_count": self.batch_count,
                "transform": self.transform,
                "resource_handle": self._handle,
                "input_layout": {
                    "embed": self.input_layout.embed,
                    "stride": self.input_layout.stride,
                    "batch_distance": self.input_layout.batch_distance,
                },
                "output_layout": {
                    "embed": self.output_layout.embed,
                    "stride": self.output_layout.stride,
                    "batch_distance": self.output_layout.batch_distance,
                },
            },
            execution_scope={
                "algorithm": (
                    "row_batch_cross_batch_columns"
                    if self._cross_batch
                    else (
                        "row_batch_column_inplace"
                        if self._separable
                        else "cufft_estimate"
                    )
                ),
                "workspace_limit_bytes": self._workspace_bytes,
                "stream_binding": "runtime_ordered",
                "capture_compatible": True,
                **(
                    {"store_callback": _store_callback.facts}
                    if _store_callback is not None
                    else {}
                ),
            },
        )
        self._retained_execution_contract = RetainedExecutionContract(
            identity=self._retained_identity,
            cost_model=HardwareExecutionCostModel(
                (
                    fixed_cost("provider_library_load", "process"),
                    fixed_cost("plan_creation", "provider_generation"),
                    fixed_cost("workspace_allocation", "provider_generation"),
                    fixed_cost("graph_capture", "graph_instance"),
                    scale_cost(
                        "fft_execution",
                        "transform_elements",
                        "batch_count",
                    ),
                )
            ),
            workspace_ownership="provider_generation",
            concurrency_policy="runtime_ordered",
            automatic_selection_policy="forbidden",
        )

    @property
    def closed(self):
        return self._handle is None

    @property
    def shape(self):
        """Legacy alias for :attr:`input_shape`; C2C shapes are unchanged."""

        return self.input_shape

    @property
    def input_shape(self):
        if not self._input_compact:
            return (self.input_storage_scalars,)
        components = 1 if self.transform == "r2c" else 2
        shape = (
            self.input_dimensions if components == 1 else (*self.input_dimensions, 2)
        )
        if self.batch_count == 1:
            return shape
        return (self.batch_count, *shape)

    @property
    def output_shape(self):
        if not self._output_compact:
            return (self.output_storage_scalars,)
        components = 1 if self.transform == "c2r" else 2
        shape = (
            self.output_dimensions if components == 1 else (*self.output_dimensions, 2)
        )
        if self.batch_count == 1:
            return shape
        return (self.batch_count, *shape)

    @property
    def inverse_scale(self):
        """Scale required after an unnormalized C2C inverse or C2R transform."""

        return 1.0 / _product(self.dimensions)

    def record(
        self,
        *,
        direction=None,
        input="input",
        output="output",
    ):
        self._validate_lifetime()
        return CufftRecording(
            self,
            direction=direction,
            input=input,
            output=output,
        )

    def execute(self, input, output, *, direction=None):
        recording = self.record(direction=direction)
        recording.execute({"input": input, "output": output})
        return output

    def _validate_array(self, value, name, expected_shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuFFT binding {name!r} must be a Taichi ndarray"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != expected_shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuFFT binding {name!r} must have scalar f32 shape "
                f"{expected_shape}"
            )
        return value.arr

    def _execute(self, input_array, output_array, direction):
        self._validate_lifetime()
        self._runtime_prog._cuda_cufft_execute(
            self._handle, input_array, output_array, direction
        )

    def _validate_lifetime(self):
        if self._handle is None:
            raise TaichiRuntimeError("CUDA cuFFT plan has been closed")
        validate_runtime_generation(
            self,
            "CUDA cuFFT plan belongs to a previous Taichi runtime generation",
        )

    def validate_graph_lifetime(self):
        self._validate_lifetime()

    def memory_report(self):
        """Report exact workspace bytes without inventing opaque plan bytes."""

        handle_present = self._handle is not None
        runtime_valid = handle_present and runtime_generation_matches(self)
        return make_memory_report(
            f"cufft_{self.transform}_{self.rank}d",
            "cuda",
            (
                HardwareMemoryComponent(
                    "plan_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=runtime_valid,
                ),
                HardwareMemoryComponent(
                    "automatic_workspace",
                    self._workspace_bytes,
                    True,
                    "provider_generation",
                    "provider",
                    resident=runtime_valid,
                ),
            ),
            lifecycle_state=(
                "ready"
                if runtime_valid
                else "closed" if not handle_present else "runtime_invalid"
            ),
            ownership_scope="plan_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def _graph_provider_memory_identity(self):
        identity = (
            "cufft_plan",
            self._runtime_generation,
            self.dimensions,
            self.batch_count,
            self.transform,
            self.input_layout,
            self.output_layout,
        )
        if self._cross_batch:
            return (*identity, "row_batch_cross_batch_columns")
        return (*identity, "row_batch_column_inplace") if self._separable else identity

    def close(self):
        if self._handle is None:
            return None
        handle = self._handle
        self._handle = None
        if runtime_generation_matches(self):
            self._runtime_prog._destroy_cuda_cufft_plan(handle)
        return None

    destroy = close

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CufftPlan1D(_CufftPlanBase):
    """One compact 1D C2C, R2C, or C2R plan.

    Complex values use a final scalar axis ``[real, imag]``. Native inverse
    transforms are unnormalized; multiply by :attr:`inverse_scale` when unit
    inverse normalization is required.
    """

    def __init__(self, length, *, batch_count=1, transform="c2c"):
        self.length = _positive_int(length, "length")
        self._initialize(
            (self.length,),
            batch_count=batch_count,
            transform=transform,
        )


class CufftPlanND(_CufftPlanBase):
    """One batched 2D/3D plan with optional explicit physical layouts."""

    def __init__(
        self,
        dimensions,
        *,
        batch_count=1,
        transform="c2c",
        input_layout=None,
        output_layout=None,
    ):
        dimensions = _positive_int_tuple(dimensions, "dimensions")
        if len(dimensions) not in (2, 3):
            raise ValueError("CufftPlanND dimensions must have rank 2 or 3")
        self._initialize(
            dimensions,
            batch_count=batch_count,
            transform=transform,
            input_layout=input_layout,
            output_layout=output_layout,
        )


@dataclass(frozen=True)
class CufftPlanCacheStatistics:
    create_requests: int
    cache_hits: int
    cache_misses: int
    live_handles: int
    live_plans: int
    workspace_bytes_live: int


def cache_statistics():
    """Return passive current-runtime plan/cache/workspace counters."""

    program = impl.get_runtime().prog
    if program is None or active_backend() != "cuda":
        values = {
            "create_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "live_handles": 0,
            "live_plans": 0,
            "workspace_bytes_live": 0,
        }
    else:
        values = dict(program._cuda_cufft_plan_cache_statistics())
    return CufftPlanCacheStatistics(
        **{name: int(value) for name, value in values.items()}
    )


def is_available():
    """Explicitly probe whether a compatible basic cuFFT provider is present."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cufft")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "fft.transform.cufft"
    )
    return operation.discovery == "available"


__all__ = [
    "CufftLayout",
    "CufftPlan1D",
    "CufftPlanND",
    "CufftPlanCacheStatistics",
    "CufftRecording",
    "cache_statistics",
    "is_available",
]
