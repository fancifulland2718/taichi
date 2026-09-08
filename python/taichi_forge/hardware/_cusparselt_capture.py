"""Retained cuSPARSELt snapshot or recompress/matmul Graph commands."""

import ctypes
import math

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._cusparselt import CusparseLtMatmulPlan
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
from taichi_forge.types.primitive_types import f16, u8


class _PlanLease:
    def __init__(self, plan, recompress):
        self.plan = None
        mode = "refresh" if recompress else "snapshot"
        with plan.provider._lock, plan._lock:
            plan._validate_lifetime()
            if plan._preparation_only:
                raise TaichiRuntimeError(
                    "cuSPARSELt preparation descriptions cannot be captured"
                )
            if plan._capture_leases and plan._capture_mode != mode:
                raise TaichiRuntimeError(
                    "cuSPARSELt cannot mix snapshot and refreshing recordings on one plan"
                )
            if not recompress and not plan._compressed_ready:
                raise TaichiRuntimeError(
                    "cuSPARSELt snapshot recording requires compress(A) first"
                )
            if recompress:
                # Capture itself does not produce data, and Python does not track
                # Graph executions. Explicit execute must not assume readiness.
                plan._compressed_ready = False
            plan._capture_mode = mode
            plan._capture_leases += 1
            self.plan = plan

    def __del__(self):
        if self.plan is not None:
            with self.plan._lock:
                self.plan._capture_leases -= 1
                if not self.plan._capture_leases:
                    self.plan._capture_mode = None
            self.plan = None


class _MatmulCaptureRecipe(_CudaGraphCaptureRecipe):
    def __init__(self, recording):
        self.lease = recording._lease
        self.names, self.arrays = recording.binding_names, recording._arrays
        self.recompress, self.alpha, self.beta = (
            recording.recompress,
            recording.alpha,
            recording.beta,
        )
        self.matmul_count = recording.matmul_count
        self.kind = (
            "cusparselt_compress_matmul_f16"
            if self.recompress
            else "cusparselt_snapshot_matmul_f16"
        )
        if self.matmul_count > 1:
            self.kind = "cusparselt_shared_a_matmul_f16"

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        plan = self.lease.plan
        plan._validate_lifetime()
        native = core._CudaCusparseLtCapturePlan()
        if hasattr(native, "matmul_count"):
            native.matmul_count = self.matmul_count
        native.compress_address = ctypes.cast(
            plan.provider._execution_api.compress_sparse_a, ctypes.c_void_p
        ).value
        native.execute_address = ctypes.cast(
            plan.provider._execution_api.execute_matmul, ctypes.c_void_p
        ).value
        native.handle = plan._handle.value
        for name in (
            "m",
            "n",
            "k",
            "compressed_bytes",
            "compression_buffer_bytes",
            "workspace_bytes",
        ):
            setattr(native, name, getattr(plan, name))
        native.alignment_bytes = plan._alignment_bytes
        native.recompress, native.alpha, native.beta = (
            self.recompress,
            self.alpha,
            self.beta,
        )
        arguments = [
            Arg(ArgKind.NDARRAY, name, dtype, ndim=rank)
            for name, (dtype, rank) in zip(self.names, self.arrays)
        ]
        builder._dispatch_cuda_cusparselt_capture_recipe(program, native, arguments)


class _CaptureExecutable(HardwareRecordingExecutable):
    def __init__(self, recording):
        super().__init__(
            recording,
            runtime_bindings=tuple(
                (name, "ndarray") for name in recording._public_names
            ),
            lifetime_leases=(recording,),
            debug_info={"kind": recording._cuda_capture_recipe.kind},
            fixed_bindings=recording._fixed_bindings,
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


class CusparseLtCaptureRecording(BackendCommandRecording):
    """Graph-only typed C-ABI capture; no Python callback during replay."""

    _graph_binding_frame_capture_safe = bool(
        getattr(
            getattr(core, "_CudaCusparseLtCapturePlan", None),
            "supports_binding_frames",
            lambda: False,
        )()
    )

    def __init__(self, plan, *, a, b, c, d, alpha, beta, _products=None):
        if not isinstance(plan, CusparseLtMatmulPlan):
            raise TypeError("cuSPARSELt capture requires a matmul plan")
        if not hasattr(core, "_CudaCusparseLtCapturePlan"):
            raise TaichiRuntimeError(
                "cuSPARSELt capture requires typed native capture support"
            )
        recompress = a is not None
        products = (
            ((b, c, d),) if _products is None else tuple(tuple(x) for x in _products)
        )
        if not products or any(len(x) != 3 for x in products):
            raise ValueError("cuSPARSELt products require nonempty B/C/D triples")
        if len(products) > 1 and not hasattr(
            core._CudaCusparseLtCapturePlan, "matmul_count"
        ):
            raise TaichiRuntimeError(
                "cuSPARSELt shared-A groups require native group capture support"
            )
        product_names = tuple(x for row in products for x in row)
        public = (*product_names, a) if recompress else product_names
        if any(not isinstance(x, str) or not x for x in public) or len(
            set(public)
        ) != len(public):
            raise ValueError(
                "cuSPARSELt Graph binding names must be distinct nonempty strings"
            )
        alpha, beta = (ctypes.c_float(float(x)).value for x in (alpha, beta))
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ValueError("cuSPARSELt Graph coefficients must be finite f32 values")
        prefix = f"__forge_cusparselt_{plan._handle.value if plan._handle else 0}"
        while any(x.startswith(prefix) for x in public):
            prefix += "_"
        fixed = {f"{prefix}_compressed": plan._compressed_a}
        names, arrays = [*product_names, *fixed], [(f16, 2)] * len(product_names) + [
            (u8, 1)
        ]
        if plan.workspace_bytes:
            name = f"{prefix}_workspace"
            fixed[name] = plan._workspace
            names.append(name)
            arrays.append((u8, 1))
        if recompress:
            names.append(a)
            arrays.append((f16, 2))
            if plan.compression_buffer_bytes:
                name = f"{prefix}_compression"
                fixed[name] = plan._compression_buffer
                names.append(name)
                arrays.append((u8, 1))
        super().__init__(
            backend="cuda",
            binding_names=tuple(names),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_lease", _PlanLease(plan, recompress))
        for name, value in dict(
            plan=plan,
            matmul_count=len(products),
            recompress=recompress,
            alpha=alpha,
            beta=beta,
            _public_names=public,
            _fixed_bindings=fixed,
            _arrays=tuple(arrays),
        ).items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_cuda_capture_recipe", _MatmulCaptureRecipe(self))
        identity = make_retained_plan_identity(
            "tensor.matmul.cusparselt",
            "cusparselt",
            "cuda",
            provider_scope=dict(plan.provider.identity),
            problem_scope=dict(
                m=plan.m,
                n=plan.n,
                k=plan.k,
                matmul_count=len(products),
                alpha=alpha,
                beta=beta,
                input_contract="already_valid_fp16_row_2of4",
                weights=plan._capture_mode,
            ),
            execution_scope=dict(
                plan_handle=plan._handle.value,
                compressed_bytes=plan.compressed_bytes,
                compression_buffer_bytes=plan.compression_buffer_bytes,
                workspace_bytes=plan.workspace_bytes,
                binding_names=self.binding_names,
            ),
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("provider_load", "process"),
                        fixed_cost("plan_and_buffers", "provider_generation"),
                        fixed_cost("native_capture", "graph_instance"),
                        scale_cost(
                            (
                                "compression_and_matmul"
                                if recompress
                                else "snapshot_matmul"
                            ),
                            "matrix_extents",
                        ),
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
                    GraphAccess.WRITE
                    if i < 3 * self.matmul_count and i % 3 == 2
                    else (
                        GraphAccess.READ_WRITE
                        if dtype == u8
                        and (i != 3 * self.matmul_count or self.recompress)
                        else GraphAccess.READ
                    )
                ),
            )
            for i, (name, (dtype, _)) in enumerate(
                zip(self.binding_names, self._arrays)
            )
        )

    def _graph_provider_memory_report(self):
        from taichi_forge.hardware._memory import (
            HardwareMemoryComponent,
            make_memory_report,
        )

        return make_memory_report(
            "cusparselt_matmul",
            "cuda",
            tuple(
                HardwareMemoryComponent(
                    name,
                    amount,
                    amount is not None,
                    "provider_generation",
                    "provider" if amount is not None else "driver",
                    resident=not self.plan.closed,
                )
                for name, amount in (
                    ("compressed_a", self.plan.compressed_bytes),
                    ("compression_buffer", self.plan.compression_buffer_bytes),
                    ("matmul_workspace", self.plan.workspace_bytes),
                    ("vendor_state", None),
                )
            ),
            lifecycle_state="closed" if self.plan.closed else "ready",
            ownership_scope="plan_generation",
        )

    def execute(self, bindings):
        raise TaichiRuntimeError("cuSPARSELt capture is a Graph-only recording")

    def _as_graph_native_node(self):
        return _CaptureNode(self)
