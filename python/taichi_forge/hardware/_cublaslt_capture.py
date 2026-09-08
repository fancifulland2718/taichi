"""Private typed cuBLASLt capture; all provider work stays at cold boundaries."""

import ctypes
from dataclasses import replace

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._cublaslt import CublasLtMatmulPlan, _MatmulAlgo
from taichi_forge.hardware._native_adapter import (
    HardwareRecordingExecutable,
    validate_runtime_generation,
)
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    attach_retained_execution_contract,
    fixed_cost,
    retained_execution_contract,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, u8


_BINDING_FRAMES_SUPPORTED = bool(
    getattr(
        getattr(core, "_CudaCublasLtCapturePlan", None),
        "supports_binding_frames",
        lambda: False,
    )()
)


class _PlanLease:
    """Keep descriptors and exact workspace alive until every capture retires."""

    def __init__(self, plan):
        self.plan = None
        with plan.provider._lock, plan._lock:
            plan.provider._validate_lifetime()
            if plan.closed:
                raise TaichiRuntimeError("cuBLASLt capture requires a live plan")
            if getattr(plan, "_preparation_only", False):
                raise TaichiRuntimeError(
                    "cuBLASLt preparation descriptions must be materialized before capture"
                )
            plan._capture_leases += 1
            self.plan = plan

    def __del__(self):
        plan = self.plan
        if plan is not None:
            with plan._lock:
                plan._capture_leases -= 1
            self.plan = None


class _MatmulCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cublaslt_retained_matmul_f32"

    def __init__(self, recording):
        # Retain the lease rather than the recording, avoiding a reference cycle.
        self.lease = recording._lease
        self.names = recording.binding_names

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        plan = self.lease.plan
        validate_runtime_generation(
            plan, "cuBLASLt capture belongs to a previous Taichi runtime generation"
        )
        if plan.closed:
            raise TaichiRuntimeError("cuBLASLt capture requires a live plan")
        native = core._CudaCublasLtCapturePlan()
        native.matmul_address = ctypes.cast(
            plan.provider._library.matmul, ctypes.c_void_p
        ).value
        native.handle = plan.provider._handle.value
        native.descriptor = plan._matmul_desc.value
        native.layouts = [
            layout.value
            for layout in (
                plan._a_layout,
                plan._b_layout,
                plan._c_layout,
                plan._d_layout,
            )
        ]
        native.algorithm = ctypes.string_at(
            ctypes.byref(plan._heuristic.algo), ctypes.sizeof(_MatmulAlgo)
        )
        shapes = (plan.a_shape, plan.b_shape, plan.output_shape)
        order = plan._capture_operand_order
        native.shapes = [shapes[i] for i in order]
        native.workspace_bytes = plan.workspace_bytes
        native.alpha, native.beta = plan.alpha, plan.beta
        arguments = [
            Arg(ArgKind.NDARRAY, name, f32, ndim=len(shape))
            for name, shape in zip((self.names[i] for i in order), native.shapes)
        ]
        if plan.workspace_bytes:
            arguments.append(Arg(ArgKind.NDARRAY, self.names[3], u8, ndim=1))
        builder._dispatch_cuda_cublaslt_capture_recipe(program, native, arguments)


class _CaptureExecutable(HardwareRecordingExecutable):
    def __init__(self, recording):
        plan = recording.plan
        super().__init__(
            recording,
            runtime_bindings=tuple(
                (name, "ndarray") for name in recording.binding_names[:3]
            ),
            lifetime_leases=(recording,),
            debug_info={
                "kind": "cuda_cublaslt_captured_matmul_f32",
                "workspace_bytes": plan.workspace_bytes,
            },
            fixed_bindings=(
                {recording.binding_names[3]: plan.workspace}
                if plan.workspace_bytes
                else {}
            ),
            publish_time_binding_validation_stable=getattr(
                recording, "graph_publish_time_binding_validation_stable", False
            ),
        )
        if not callable(getattr(recording, "validate_graph_bindings", None)):
            # The typed native command checks these bindings at capture. Do
            # not install a no-op Python validator on every replay.
            self.validate_graph_bindings = None


class _CaptureNode(NativeGraphNode):
    def __init__(self, recording):
        self.recording = recording

    def compile(self):
        return _CaptureExecutable(self.recording)


class CublasLtCaptureRecording(BackendCommandRecording):
    """Fixed-plan Graph-only recording, not a raw algorithm search interface."""

    # Explicit close is prevented by the lease; reset invalidates the Graph.
    # No per-replay Python provider validation is necessary.
    _graph_binding_frame_capture_safe = _BINDING_FRAMES_SUPPORTED

    def __init__(self, plan, *, workspace):
        if not isinstance(plan, CublasLtMatmulPlan):
            raise TypeError("cuBLASLt capture requires a retained matmul plan")
        if not hasattr(core, "_CudaCublasLtCapturePlan"):
            raise TaichiRuntimeError(
                "cuBLASLt capture requires a runtime with typed matmul capture support"
            )
        if (
            not isinstance(workspace, str)
            or not workspace
            or workspace in plan.binding_names
        ):
            raise ValueError(
                "cuBLASLt capture workspace binding must be distinct and nonempty"
            )
        names = plan.binding_names + ((workspace,) if plan.workspace_bytes else ())
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_lease", _PlanLease(plan))
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_cuda_capture_recipe", _MatmulCaptureRecipe(self))
        previous = retained_execution_contract(plan)
        cost = HardwareExecutionCostModel(
            (
                fixed_cost("provider_library_load", "process"),
                fixed_cost("provider_handle", "runtime_generation"),
                fixed_cost(
                    "descriptors_heuristic_and_workspace", "provider_generation"
                ),
                fixed_cost("native_vendor_capture", "graph_instance"),
            )
            + previous.cost_model.scale_costs
        )
        attach_retained_execution_contract(self, replace(previous, cost_model=cost))

    @property
    def resource_effects(self):
        effects = self.plan.resource_effects[:3]
        if self.plan.workspace_bytes:
            effects += (ResourceEffect(self.binding_names[3], GraphAccess.READ_WRITE),)
        return effects

    def execute(self, bindings):
        raise TaichiRuntimeError("cuBLASLt captured matmul is a Graph-only recording")

    def _as_graph_native_node(self):
        return _CaptureNode(self)
