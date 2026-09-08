"""Fixed cuTENSOR plans captured through the existing bundled-provider C ABI."""

import ctypes
import math

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._cutensor import CutensorContractionPlan
from taichi_forge.hardware._native_adapter import HardwareRecordingExecutable
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    RetainedExecutionContract,
    attach_retained_execution_contract,
    fixed_cost,
    make_retained_plan_identity,
    scale_cost,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, u8


class _PlanLease:
    def __init__(self, plan):
        self.plan = None
        with plan.provider._lock, plan._lock:
            plan._validate_lifetime()
            if plan._preparation_only:
                raise TaichiRuntimeError("cuTENSOR preparation descriptions cannot be captured")
            plan._capture_leases += 1
            self.plan = plan

    def __del__(self):
        if self.plan is not None:
            with self.plan._lock:
                self.plan._capture_leases -= 1
            self.plan = None


class _ContractionCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cutensor_retained_contraction_f32"

    def __init__(self, recording):
        self.lease = recording._lease
        self.names = recording.binding_names
        self.alpha, self.beta = recording.alpha, recording.beta

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        plan = self.lease.plan
        plan._validate_lifetime()
        native = core._CudaCutensorCapturePlan()
        native.execute_address = ctypes.cast(
            plan.provider._execution_api.execute_contraction, ctypes.c_void_p
        ).value
        native.handle = plan._handle.value
        native.shapes = plan._shapes
        native.workspace_bytes = plan.workspace_required_bytes
        native.alignment_bytes = plan._alignment_bytes
        native.output_alias_compatible = plan._tensors[2] == plan._tensors[3]
        native.alpha, native.beta = self.alpha, self.beta
        args = [
            Arg(ArgKind.NDARRAY, name, f32, ndim=len(shape))
            for name, shape in zip(self.names, plan._shapes)
        ]
        if plan.workspace_required_bytes:
            args.append(Arg(ArgKind.NDARRAY, self.names[4], u8, ndim=1))
        builder._dispatch_cuda_cutensor_capture_recipe(program, native, args)


class _CaptureExecutable(HardwareRecordingExecutable):
    def __init__(self, recording):
        super().__init__(
            recording,
            runtime_bindings=tuple(
                (name, "ndarray") for name in recording.binding_names[:4]
            ),
            lifetime_leases=(recording,),
            debug_info={
                "kind": "cuda_cutensor_captured_contraction_f32",
                "workspace_bytes": recording.plan.workspace_required_bytes,
            },
            fixed_bindings=(
                {recording.binding_names[4]: recording.plan._workspace}
                if recording.plan.workspace_required_bytes
                else {}
            ),
            publish_time_binding_validation_stable=getattr(
                recording, "graph_publish_time_binding_validation_stable", False
            ),
        )
        if not callable(getattr(recording, "validate_graph_bindings", None)):
            self.validate_graph_bindings = None


class _CaptureNode(NativeGraphNode):
    def __init__(self, recording):
        self.recording = recording

    def compile(self):
        return _CaptureExecutable(self.recording)


class CutensorCaptureRecording(BackendCommandRecording):
    """Graph-only contraction. No provider callback is involved in replay."""

    _graph_binding_frame_capture_safe = bool(
        getattr(
            getattr(core, "_CudaCutensorCapturePlan", None),
            "supports_binding_frames",
            lambda: False,
        )()
    )

    def __init__(self, plan, *, a, b, c, d, workspace, alpha, beta):
        if not isinstance(plan, CutensorContractionPlan):
            raise TypeError("cuTENSOR capture requires a contraction plan")
        if not hasattr(core, "_CudaCutensorCapturePlan"):
            raise TaichiRuntimeError(
                "cuTENSOR capture requires a runtime with typed contraction capture support"
            )
        names = (a, b, c, d, workspace)
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 5
        ):
            raise ValueError(
                "cuTENSOR Graph binding names must be distinct nonempty strings"
            )
        alpha, beta = (
            ctypes.c_float(float(alpha)).value,
            ctypes.c_float(float(beta)).value,
        )
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ValueError("cuTENSOR Graph coefficients must be finite f32 values")
        super().__init__(
            backend="cuda",
            binding_names=names if plan.workspace_required_bytes else names[:4],
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_lease", _PlanLease(plan))
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(
            self, "_cuda_capture_recipe", _ContractionCaptureRecipe(self)
        )
        identity = make_retained_plan_identity(
            "tensor.contract.cutensor",
            "cutensor",
            "cuda",
            provider_scope=dict(plan.provider.identity),
            problem_scope={
                "tensors": plan._tensors,
                "compute": plan._compute,
                "alpha": alpha,
                "beta": beta,
            },
            execution_scope={
                "plan_handle": plan._handle.value,
                "workspace_bytes": plan.workspace_required_bytes,
                "binding_names": self.binding_names,
                "capture": "fixed_plan",
            },
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("provider_load", "process"),
                        fixed_cost("plan_and_workspace", "provider_generation"),
                        fixed_cost("native_capture", "graph_instance"),
                        scale_cost("contraction", "tensor_extents"),
                    )
                ),
                workspace_ownership="provider_generation",
                concurrency_policy="runtime_ordered",
                automatic_selection_policy="forbidden",
            ),
        )

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(
                name,
                (
                    GraphAccess.READ
                    if i < 3
                    else (GraphAccess.WRITE if i == 3 else GraphAccess.READ_WRITE)
                ),
            )
            for i, name in enumerate(self.binding_names)
        )

    def _graph_provider_memory_report(self):
        from taichi_forge.hardware._memory import (
            HardwareMemoryComponent,
            make_memory_report,
        )

        return make_memory_report(
            "cutensor_contraction",
            "cuda",
            (
                HardwareMemoryComponent(
                    "contraction_workspace",
                    self.plan.workspace_required_bytes,
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

    def execute(self, bindings):
        raise TaichiRuntimeError(
            "cuTENSOR captured contraction is a Graph-only recording"
        )

    def _as_graph_native_node(self):
        return _CaptureNode(self)
