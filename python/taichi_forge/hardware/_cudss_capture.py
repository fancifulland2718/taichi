"""Graph-owned cuDSS numerical phases; no Python/vendor call on replay."""

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._linalg import CudssPlan
from taichi_forge.hardware._native_adapter import (
    HardwareRecordingExecutable,
    static_resource_effect,
)
from taichi_forge.hardware._retained import (
    HardwareExecutionCostModel,
    RetainedExecutionContract,
    attach_retained_execution_contract,
    fixed_cost,
    make_retained_plan_identity,
    scale_cost,
)
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32

_PHASES = {"solve": 0, "factor_solve": 4, "refactor_solve": 8}


class _PlanLease:
    def __init__(self, plan):
        self.plan = None
        plan._ensure_open()
        if not plan._graph_owned:
            raise TaichiRuntimeError("cuDSS capture needs a Graph-owned snapshot")
        plan._capture_leases += 1
        self.plan = plan

    def __del__(self):
        if self.plan is not None:
            self.plan._capture_leases -= 1
            self.plan = None


class _CaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cudss_retained_solve_f32"

    def __init__(self, recording):
        self.lease = recording._lease
        self.names, self.phase = recording.binding_names, recording.phase

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        plan = self.lease.plan
        plan._ensure_open()
        dispatch = (
            builder._dispatch_cuda_cudss_capture_group
            if len(self.names) > (2 if self.phase == "solve" else 3)
            else builder._dispatch_cuda_cudss_capture_recipe
        )
        dispatch(
            program,
            plan._handle,
            _PHASES[self.phase],
            [Arg(ArgKind.NDARRAY, name, f32, ndim=1) for name in self.names],
        )


class _CaptureNode(NativeGraphNode):
    def __init__(self, recording):
        self.recording = recording

    def compile(self):
        recording = self.recording
        executable = HardwareRecordingExecutable(
            recording,
            runtime_bindings=tuple(
                (name, "ndarray") for name in recording.binding_names
            ),
            lifetime_leases=(recording,),
            debug_info={
                "kind": "cuda_cudss_captured_solve_f32",
                "numeric_phase": recording.phase,
            },
        )
        # Native capture owns validation; a no-op Python validator would make
        # the immutable binding unnecessarily volatile on every submission.
        executable.validate_graph_bindings = None
        return executable


class CudssCaptureRecording(BackendCommandRecording):
    """Internal complete-recipe materializer, not a public raw numerical axis."""

    _graph_binding_frame_capture_safe = hasattr(
        core.GraphBuilder, "_dispatch_cuda_cudss_capture_recipe"
    )

    def __init__(
        self,
        plan,
        *,
        phase,
        values="matrix_values",
        rhs="rhs",
        solution="solution",
        _rhs_pairs=None,
    ):
        if not isinstance(plan, CudssPlan) or phase not in _PHASES:
            raise TypeError("cuDSS capture needs a prepared plan and numerical phase")
        if not self._graph_binding_frame_capture_safe:
            raise TaichiRuntimeError(
                "cuDSS capture requires native Graph-owned plan support"
            )
        pairs = (
            ((rhs, solution),)
            if _rhs_pairs is None
            else tuple(tuple(p) for p in _rhs_pairs)
        )
        if not pairs or any(len(p) != 2 for p in pairs):
            raise ValueError("cuDSS capture requires RHS/solution binding pairs")
        if len(pairs) > 1 and not hasattr(
            core.GraphBuilder, "_dispatch_cuda_cudss_capture_group"
        ):
            raise TaichiRuntimeError(
                "cuDSS shared-factor groups require native group capture support"
            )
        names = (() if phase == "solve" else (values,)) + tuple(
            name for p in pairs for name in p
        )
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError("cuDSS capture names must be distinct nonempty strings")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=len(pairs) + (phase != "solve"),
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "_rhs_pairs", pairs)
        object.__setattr__(self, "_lease", _PlanLease(plan))
        object.__setattr__(self, "_cuda_capture_recipe", _CaptureRecipe(self))
        identity = make_retained_plan_identity(
            "linalg.solve.cudss",
            "cudss",
            "cuda",
            provider_scope=dict(plan.provider_identity),
            problem_scope={
                "rows": plan._rows,
                "nonzeros": plan._nnz,
                "matrix_type": plan.matrix_type,
                "matrix_view": plan.matrix_view,
            },
            execution_scope={
                "numeric_phase": phase,
                "rhs_count": len(pairs),
                "factor_reuse": "all_rhs_in_region",
                "plan_handle": plan._handle,
                "configuration": plan._configuration_report()["configuration"],
                "capture": "retained_private_snapshot",
            },
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost(
                            "analysis_and_factor_snapshot", "provider_generation"
                        ),
                        fixed_cost("capture", "graph_instance"),
                        scale_cost(phase, "rows", "nonzeros"),
                    )
                ),
                workspace_ownership="provider_generation",
                concurrency_policy="runtime_ordered",
                automatic_selection_policy="forbidden",
            ),
        )

    @property
    def resource_effects(self):
        outputs = {pair[1] for pair in self._rhs_pairs}
        return tuple(
            ResourceEffect(
                name,
                (GraphAccess.WRITE if name in outputs else GraphAccess.READ),
            )
            for name in self.binding_names
        ) + (static_resource_effect(self.plan._effect_name, GraphAccess.READ_WRITE),)

    def _graph_provider_memory_identity(self):
        return ("cudss_graph_plan", self.plan._runtime_generation, self.plan._handle)

    def _graph_provider_memory_report(self):
        from taichi_forge.hardware._memory import (
            HardwareMemoryComponent,
            make_memory_report,
        )

        allocation = self.plan._configuration_report()["graph_allocator"]
        return make_memory_report(
            "cudss_graph_plan",
            "cuda",
            (
                HardwareMemoryComponent(
                    "private_snapshot",
                    allocation["snapshot_bytes"],
                    True,
                    "provider_generation",
                    "provider",
                ),
                HardwareMemoryComponent(
                    "vendor_requested_payload",
                    allocation["live_bytes"] - allocation["snapshot_bytes"],
                    True,
                    "provider_generation",
                    "provider",
                ),
                HardwareMemoryComponent(
                    "driver_pool_backing_and_opaque_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                ),
            ),
            lifecycle_state="ready",
            ownership_scope="plan_generation",
        )

    def execute(self, bindings):
        raise TaichiRuntimeError("cuDSS captured solve is a Graph-only recording")

    def _as_graph_native_node(self):
        return _CaptureNode(self)
