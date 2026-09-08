"""Optional CUDA vendor linear-algebra providers."""

import math
from numbers import Real
from types import MappingProxyType

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    validate_exact_bindings,
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
from taichi_forge._hardware_telemetry import (
    hardware_failure_phase,
    hardware_provider_call,
    instrument_hardware_recording,
    operation_executed,
)
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


def _dimension(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > 0x7FFFFFFF
    ):
        raise ValueError(f"CUDA cuBLAS {name} must be in [1, INT_MAX]")
    return value


def _scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"CUDA cuBLAS {name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"CUDA cuBLAS {name} must be finite")
    return result


_CUBLAS_GEMM_EXECUTION_CONTRACT = RetainedExecutionContract(
    identity=None,
    cost_model=HardwareExecutionCostModel(
        (
            fixed_cost("provider_library_load", "process"),
            fixed_cost("provider_handle", "runtime_generation"),
            scale_cost("gemm_execution", "rows", "columns", "inner"),
        )
    ),
    workspace_ownership="none",
    concurrency_policy="independent_invocations",
    automatic_selection_policy="forbidden",
)


@instrument_hardware_recording("linalg.gemm.cublas", runtime_resource=True)
class CublasGemmRecording(BackendCommandRecording):
    """One compact row-major f32 GEMM executed by the user's cuBLAS."""

    def __init__(
        self,
        rows,
        columns,
        inner,
        *,
        alpha=1.0,
        beta=0.0,
        a="a",
        b="b",
        output="output",
    ):
        rows = _dimension(rows, "rows")
        columns = _dimension(columns, "columns")
        inner = _dimension(inner, "inner dimension")
        alpha = _scalar(alpha, "alpha")
        beta = _scalar(beta, "beta")
        names = (a, b, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuBLAS binding names must be nonempty strings")
        if len(set(names)) != 3:
            raise ValueError("CUDA cuBLAS binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "output", output)
        attach_retained_execution_contract(
            self,
            _CUBLAS_GEMM_EXECUTION_CONTRACT,
        )

    @property
    def resource_effects(self):
        output_access = (
            GraphAccess.WRITE if self.beta == 0.0 else GraphAccess.READ_WRITE
        )
        return (
            ResourceEffect(self.a, GraphAccess.READ),
            ResourceEffect(self.b, GraphAccess.READ),
            ResourceEffect(self.output, output_access),
        )

    @staticmethod
    def _validate_array(value, name, shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must be a Taichi ndarray"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuBLAS binding {name!r} must have compact scalar f32 "
                f"shape {shape}"
            )
        return value.arr

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuBLAS")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuBLAS GEMM requires an active runtime")
        a = self._validate_array(bindings[self.a], self.a, (self.rows, self.inner))
        b = self._validate_array(bindings[self.b], self.b, (self.inner, self.columns))
        output = self._validate_array(
            bindings[self.output], self.output, (self.rows, self.columns)
        )
        if (
            bindings[self.output] is bindings[self.a]
            or bindings[self.output] is bindings[self.b]
        ):
            raise TaichiRuntimeError(
                "CUDA cuBLAS GEMM output must not alias either input"
            )
        with hardware_provider_call("cublas"):
            program._cuda_cublas_gemm_f32(
                a,
                b,
                output,
                self.rows,
                self.columns,
                self.inner,
                self.alpha,
                self.beta,
            )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            debug_info=lambda item: {
                "kind": "cuda_cublas_gemm_f32",
                "shape": (item.rows, item.columns, item.inner),
                "alpha": item.alpha,
                "beta": item.beta,
            },
        )

    def memory_report(self):
        """cuBLAS state is runtime-global and opaque to the basic ABI slice."""

        return make_memory_report(
            "cublas_gemm_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "runtime_handle_and_driver_state",
                    None,
                    False,
                    "runtime",
                    "driver",
                    resident=operation_executed("linalg.gemm.cublas"),
                ),
            ),
            ownership_scope="runtime_global",
        )


class _CusparseSpmvCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cusparse_spmv_f32"

    def __init__(self, matrix, input_name, output_name):
        self._matrix = matrix
        self._input_name = input_name
        self._output_name = output_name

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind  # pylint: disable=C0415

        input_arg = Arg(ArgKind.NDARRAY, self._input_name, f32, ndim=1)
        output_arg = Arg(ArgKind.NDARRAY, self._output_name, f32, ndim=1)
        builder._dispatch_cuda_cusparse_spmv_capture_recipe(
            self._matrix.matrix,
            program,
            input_arg,
            output_arg,
        )


class _CusparseSpmmCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "cusparse_spmm_f32"

    def __init__(self, matrix, rhs_count, algorithm, input_name, output_name):
        self._matrix = matrix
        self._rhs_count = rhs_count
        self._algorithm = algorithm
        self._input_name = input_name
        self._output_name = output_name

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind  # pylint: disable=C0415

        input_arg = Arg(ArgKind.NDARRAY, self._input_name, f32, ndim=2)
        output_arg = Arg(ArgKind.NDARRAY, self._output_name, f32, ndim=2)
        builder._dispatch_cuda_cusparse_spmm_capture_recipe(
            self._matrix.matrix,
            program,
            input_arg,
            output_arg,
            self._rhs_count,
            self._algorithm,
        )


class _CusparseTriangularCaptureRecipe(_CudaGraphCaptureRecipe):
    def __init__(
        self,
        matrix,
        rhs_count,
        fill_mode,
        unit_diagonal,
        transpose,
        input_name,
        output_name,
    ):
        self.kind = "cusparse_spsv_f32" if rhs_count == 1 else "cusparse_spsm_f32"
        self._matrix = matrix
        self._rhs_count = rhs_count
        self._fill_mode = fill_mode
        self._unit_diagonal = unit_diagonal
        self._transpose = transpose
        self._input_name = input_name
        self._output_name = output_name

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind  # pylint: disable=C0415

        ndim = 1 if self._rhs_count == 1 else 2
        input_arg = Arg(ArgKind.NDARRAY, self._input_name, f32, ndim=ndim)
        output_arg = Arg(ArgKind.NDARRAY, self._output_name, f32, ndim=ndim)
        builder._dispatch_cuda_cusparse_triangular_capture_recipe(
            self._matrix.matrix,
            program,
            input_arg,
            output_arg,
            self._rhs_count,
            self._fill_mode,
            self._unit_diagonal,
            self._transpose,
        )


def _cusparse_spmv_execution_contract(matrix, identity):
    cache_key = int(impl.runtime_generation())
    cached = getattr(matrix, "_cusparse_spmv_retained_contract", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1], cached[2]

    runtime_stats = matrix._debug_runtime_stats()  # pylint: disable=W0212
    pattern_version = runtime_stats["identity"]["pattern_version"]
    provider = runtime_stats["provider"]
    provider_scope = passive_dynamic_provider_scope(
        "cusparse",
        "cusparse-dynamic-symbols-v1",
        version=provider["library_version"],
    )
    retained_identity = make_retained_plan_identity(
        "linalg.spmv.cusparse_explicit",
        "cusparse",
        "cuda",
        provider_scope=provider_scope,
        problem_scope={
            "rows": matrix.n,
            "columns": matrix.m,
            "nonzeros": runtime_stats["identity"]["nnz"],
            "storage_format": identity["storage_format"],
            "block_size": identity.get("block_size"),
            "dtype": "f32",
            "topology_fingerprint": matrix._topology_fingerprint,
            "pattern_id": runtime_stats["identity"]["pattern_id"],
            "pattern_version": pattern_version,
            "resource_object_token": id(matrix.matrix),
            "resource_generation": pattern_version,
        },
        execution_scope={
            "algorithm": "cusparse_spmv_default",
            "workspace_limit_bytes": None,
            "stream_binding": "runtime_ordered",
            "capture_compatible": True,
        },
    )
    fixed_components = [
        fixed_cost("provider_library_load", "process"),
        fixed_cost("handle_and_descriptors", "provider_generation"),
        fixed_cost("workspace_allocation", "provider_generation"),
        fixed_cost("graph_capture", "graph_instance"),
    ]
    if provider["spmv_preprocess_available"]:
        fixed_components.append(fixed_cost("spmv_preprocess", "provider_generation"))
    contract = RetainedExecutionContract(
        identity=retained_identity,
        cost_model=HardwareExecutionCostModel(
            (*fixed_components, scale_cost("spmv_execution", "rows", "nonzeros"))
        ),
        workspace_ownership="provider_generation",
        concurrency_policy="runtime_ordered",
        automatic_selection_policy="qualification_gated",
    )
    memory_resources = dict(runtime_stats["resources"])
    matrix._cusparse_spmv_retained_contract = (
        cache_key,
        contract,
        memory_resources,
    )
    return contract, memory_resources


@instrument_hardware_recording("linalg.spmv.cusparse_explicit")
class CusparseSpmvRecording(BackendCommandRecording):
    """One f32 stored-matrix SpMV executed by the user's cuSPARSE."""

    def __init__(self, matrix, *, input="input", output="output"):
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuSPARSE SpMV matrix must be a SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if identity["backend_family"] != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires a CUDA SparseMatrix; the matrix "
                f"backend is {identity['backend_family']}"
            )
        if identity["storage_format"] not in ("csr", "bsr"):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires scalar CSR or fixed-block BSR " "storage"
            )
        if identity["dtype"] != "f32":
            raise TaichiRuntimeError("CUDA cuSPARSE SpMV requires an f32 SparseMatrix")
        if not contract["operations"]["ndarray_spmv"]:
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV is unavailable for this SparseMatrix"
            )
        names = (input, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuSPARSE binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuSPARSE binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
            no_host_readback=True,
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "_cuda_capture_recipe",
            _CusparseSpmvCaptureRecipe(matrix, input, output),
        )
        retained_contract, memory_resources = _cusparse_spmv_execution_contract(
            matrix, identity
        )
        attach_retained_execution_contract(
            self,
            retained_contract,
        )
        object.__setattr__(
            self,
            "_memory_resources",
            memory_resources,
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    @staticmethod
    def _validate_array(value, name, shape):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} must be a Taichi ndarray"
            )
        program = impl.get_runtime().prog
        if value.arr is None or value._runtime_prog is not program:
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} belongs to another " "Taichi runtime"
            )
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != shape
        ):
            raise TaichiRuntimeError(
                f"CUDA cuSPARSE binding {name!r} must have compact scalar "
                f"f32 shape {shape}"
            )
        return value.arr

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuSPARSE")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuSPARSE SpMV requires an active runtime")
        self.matrix._ensure_valid()  # pylint: disable=W0212
        input_value = bindings[self.input]
        output_value = bindings[self.output]
        input_array = self._validate_array(input_value, self.input, (self.matrix.m,))
        output_array = self._validate_array(output_value, self.output, (self.matrix.n,))
        if (
            input_value._runtime_allocation_identity
            == output_value._runtime_allocation_identity
        ):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMV output must not alias the input"
            )
        with hardware_provider_call("cusparse"):
            self.matrix.matrix.spmv(program, input_array, output_array)

    def validate_graph_lifetime(self):
        self.matrix._ensure_valid()  # pylint: disable=W0212

    def memory_report(self):
        """Report matrix/workspace bytes and keep vendor descriptors opaque."""

        lifecycle_state = "ready"
        resident = True
        resources = self._memory_resources
        try:
            resources = dict(
                self.matrix._debug_runtime_stats()["resources"]  # pylint: disable=W0212
            )
            object.__setattr__(self, "_memory_resources", resources)
        except TaichiRuntimeError:
            lifecycle_state = "runtime_invalid"
            resident = False

        pattern_shared = bool(resources["pattern_storage_shared"])
        owned_bytes = int(
            resources[
                (
                    "operator_exclusive_reserved_bytes"
                    if pattern_shared
                    else "operator_owned_reserved_bytes"
                )
            ]
        )
        components = [
            HardwareMemoryComponent(
                "matrix_values_and_spmv_workspace",
                owned_bytes,
                True,
                "provider_generation",
                "shared_user_object",
                resident=resident,
            )
        ]
        if pattern_shared:
            components.append(
                HardwareMemoryComponent(
                    "shared_sparse_pattern",
                    None,
                    False,
                    "provider_generation",
                    "shared_user_object",
                    resident=resident,
                )
            )
        components.append(
            HardwareMemoryComponent(
                "cusparse_descriptors_and_preprocess_state",
                None,
                False,
                "provider_generation",
                "driver",
                resident=resident,
            )
        )
        return make_memory_report(
            "cusparse_spmv_f32",
            "cuda",
            components,
            lifecycle_state=lifecycle_state,
            ownership_scope="sparse_matrix_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def _graph_provider_memory_identity(self):
        return ("cusparse_spmv_f32", id(self.matrix))

    def _as_graph_native_node(self):
        def debug_info(item):
            contract = item.matrix._get_format_contract()  # pylint: disable=W0212
            return {
                "kind": "cuda_cusparse_spmv_f32",
                "shape": item.matrix.shape,
                "storage_format": contract["identity"]["storage_format"],
            }

        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info=debug_info,
        )


def gemm_f32(a, b, output, *, alpha=1.0, beta=0.0):
    """Compute row-major ``output = alpha * a @ b + beta * output``."""

    a_shape = tuple(getattr(a, "shape", ()))
    b_shape = tuple(getattr(b, "shape", ()))
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        raise TaichiRuntimeError(
            "CUDA cuBLAS GEMM inputs must be compatible two-dimensional arrays"
        )
    recording = CublasGemmRecording(
        a_shape[0], b_shape[1], a_shape[1], alpha=alpha, beta=beta
    )
    recording.execute({"a": a, "b": b, "output": output})
    return output


def spmv_f32(matrix, input, output):
    """Compute ``output = matrix @ input`` with stored CUDA cuSPARSE state."""

    recording = CusparseSpmvRecording(matrix)
    recording.execute({"input": input, "output": output})
    return output


def spmm_f32(matrix, input, output, *, algorithm="row_major"):
    """Compute ``output = matrix @ input`` for a compact row-major RHS batch."""

    input_shape = tuple(getattr(input, "shape", ()))
    if len(input_shape) != 2:
        raise TaichiRuntimeError(
            "CUDA cuSPARSE SpMM input must be a two-dimensional ndarray"
        )
    recording = CusparseSpmmRecording(matrix, input_shape[1], algorithm=algorithm)
    recording.execute({"input": input, "output": output})
    return output


def spsv_f32(
    matrix,
    input,
    output,
    *,
    fill_mode="lower",
    unit_diagonal=False,
    transpose=False,
    algorithm="default",
):
    """Solve one f32 triangular CSR system with retained cuSPARSE analysis.

    Non-unit systems require finite nonzero pivots. Generic cuSPARSE SpSV has
    no zero-pivot query, so this low-level explicit operation does not perform
    a host postsolve check and singular input may produce nonfinite output.
    """

    recording = CusparseSpsvRecording(
        matrix,
        fill_mode=fill_mode,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        algorithm=algorithm,
    )
    recording.execute({"input": input, "output": output})
    return output


def spsm_f32(
    matrix,
    input,
    output,
    *,
    fill_mode="lower",
    unit_diagonal=False,
    transpose=False,
    algorithm="default",
):
    """Solve a compact row-major f32 triangular CSR multi-RHS system.

    Non-unit systems require finite nonzero pivots. Generic cuSPARSE SpSM has
    no zero-pivot query, so this low-level explicit operation does not perform
    a host postsolve check and singular input may produce nonfinite output.
    """

    input_shape = tuple(getattr(input, "shape", ()))
    if len(input_shape) != 2:
        raise TaichiRuntimeError(
            "CUDA cuSPARSE SpSM input must be a two-dimensional ndarray"
        )
    recording = CusparseSpsmRecording(
        matrix,
        input_shape[1],
        fill_mode=fill_mode,
        unit_diagonal=unit_diagonal,
        transpose=transpose,
        algorithm=algorithm,
    )
    recording.execute({"input": input, "output": output})
    return output


def cublas_is_available():
    """Explicitly probe whether a compatible cuBLAS provider is present."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cublas")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.gemm.cublas"
    )
    return operation.discovery == "available"


def is_available():
    """Compatibility alias for :func:`cublas_is_available`."""

    return cublas_is_available()


def cusparse_is_available():
    """Explicitly probe whether a compatible cuSPARSE provider is present."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cusparse")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.spmv.cusparse_explicit"
    )
    return operation.discovery == "available"


def cusparse_spmm_is_available():
    """Probe the optional cuSPARSE dense-matrix SpMM symbol slice."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cusparse")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.spmm.cusparse_explicit"
    )
    return operation.discovery == "available"


def cusparse_spsv_is_available():
    """Probe the retained cuSPARSE SpSV analysis/solve symbol slice."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cusparse")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.spsv.cusparse_explicit"
    )
    return operation.discovery == "available"


def cusparse_spsm_is_available():
    """Probe the retained cuSPARSE SpSM analysis/solve symbol slice."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cusparse")
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.spsm.cusparse_explicit"
    )
    return operation.discovery == "available"


def _cusparse_spmm_execution_contract(matrix, identity, rhs_count, algorithm):
    cache_key = (int(impl.runtime_generation()), rhs_count, algorithm)
    cached_contracts = getattr(matrix, "_cusparse_spmm_retained_contracts", None)
    if cached_contracts is None:
        cached_contracts = {}
        matrix._cusparse_spmm_retained_contracts = cached_contracts
    cached = cached_contracts.get(cache_key)
    if cached is not None:
        return cached

    runtime_stats = matrix._debug_runtime_stats()  # pylint: disable=W0212
    pattern_version = runtime_stats["identity"]["pattern_version"]
    provider = runtime_stats["provider"]
    provider_scope = passive_dynamic_provider_scope(
        "cusparse",
        "cusparse-dynamic-symbols-v1",
        version=provider["library_version"],
    )
    retained_identity = make_retained_plan_identity(
        "linalg.spmm.cusparse_explicit",
        "cusparse",
        "cuda",
        provider_scope=provider_scope,
        problem_scope={
            "rows": matrix.n,
            "columns": matrix.m,
            "nonzeros": runtime_stats["identity"]["nnz"],
            "rhs_count": rhs_count,
            "storage_format": identity["storage_format"],
            "dtype": "f32",
            "topology_fingerprint": matrix._topology_fingerprint,
            "pattern_id": runtime_stats["identity"]["pattern_id"],
            "pattern_version": pattern_version,
            "resource_object_token": id(matrix.matrix),
            "resource_generation": pattern_version,
        },
        execution_scope={
            "algorithm": algorithm,
            "workspace_limit_bytes": None,
            "stream_binding": "runtime_ordered",
            "capture_compatible": True,
        },
    )
    fixed_components = [
        fixed_cost("provider_library_load", "process"),
        fixed_cost("handle_and_descriptors", "provider_generation"),
        fixed_cost("workspace_allocation", "provider_generation"),
        fixed_cost("graph_capture", "graph_instance"),
    ]
    if algorithm in ("deterministic", "csr3_preprocessed") and provider["spmm_preprocess_available"]:
        fixed_components.append(fixed_cost("spmm_preprocess", "provider_generation"))
    contract = RetainedExecutionContract(
        identity=retained_identity,
        cost_model=HardwareExecutionCostModel(
            (
                *fixed_components,
                scale_cost("spmm_execution", "rows", "nonzeros", "rhs_count"),
            )
        ),
        workspace_ownership="provider_generation",
        concurrency_policy="single_inflight",
        automatic_selection_policy="forbidden",
    )
    result = (contract, dict(runtime_stats["resources"]))
    cached_contracts[cache_key] = result
    return result


@instrument_hardware_recording("linalg.spmm.cusparse_explicit")
class CusparseSpmmRecording(BackendCommandRecording):
    """Retained row-major f32 CSR SpMM over two or more right-hand sides.

    ``deterministic`` is the legacy name for vendor CSR_ALG3, not a
    cross-version bitwise guarantee. Numerical behavior depends on the
    loaded cuSPARSE release.
    """

    _ALGORITHMS = {
        "row_major": 0,
        "deterministic": 1,  # compatibility: optional preprocessing
        "csr3_direct": 2,
        "csr3_preprocessed": 3,
    }

    def __init__(
        self,
        matrix,
        rhs_count,
        *,
        algorithm="row_major",
        input="input",
        output="output",
    ):
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuSPARSE SpMM matrix must be a SparseMatrix")
        if isinstance(rhs_count, bool) or not isinstance(rhs_count, int):
            raise TypeError("CUDA cuSPARSE SpMM rhs_count must be an integer")
        if rhs_count < 2 or rhs_count > 0x7FFFFFFF:
            raise ValueError("CUDA cuSPARSE SpMM rhs_count must be in [2, INT_MAX]")
        if algorithm not in self._ALGORITHMS:
            raise ValueError(
                "CUDA cuSPARSE SpMM algorithm must be row_major, deterministic, "
                "csr3_direct, or csr3_preprocessed"
            )
        matrix._ensure_valid()  # pylint: disable=W0212
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if identity["backend_family"] != "cuda" or identity["storage_format"] != "csr":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMM requires a scalar CUDA CSR SparseMatrix"
            )
        if identity["dtype"] != "f32":
            raise TaichiRuntimeError("CUDA cuSPARSE SpMM requires an f32 SparseMatrix")
        runtime_stats = matrix._debug_runtime_stats()  # pylint: disable=W0212
        if not runtime_stats["provider"]["spmm_f32_available"]:
            raise TaichiRuntimeError(
                "The loaded cuSPARSE provider does not expose the optional "
                "f32 SpMM dynamic-symbol contract"
            )
        names = (input, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuSPARSE binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuSPARSE binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
            no_host_readback=True,
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs_count", rhs_count)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "_algorithm_code", self._ALGORITHMS[algorithm])
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)
        object.__setattr__(
            self,
            "_cuda_capture_recipe",
            _CusparseSpmmCaptureRecipe(
                matrix, rhs_count, self._algorithm_code, input, output
            ),
        )
        retained_contract, memory_resources = _cusparse_spmm_execution_contract(
            matrix, identity, rhs_count, algorithm
        )
        attach_retained_execution_contract(self, retained_contract)
        object.__setattr__(self, "_memory_resources", memory_resources)
        object.__setattr__(self, "_plan_info_snapshot", None)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuSPARSE SpMM")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMM requires the CUDA backend; the active "
                f"backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuSPARSE SpMM requires an active runtime")
        self.matrix._ensure_valid()  # pylint: disable=W0212
        input_value = bindings[self.input]
        output_value = bindings[self.output]
        input_array = CusparseSpmvRecording._validate_array(
            input_value, self.input, (self.matrix.m, self.rhs_count)
        )
        output_array = CusparseSpmvRecording._validate_array(
            output_value, self.output, (self.matrix.n, self.rhs_count)
        )
        if (
            input_value._runtime_allocation_identity
            == output_value._runtime_allocation_identity
        ):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE SpMM output must not alias the input"
            )
        with hardware_provider_call("cusparse"):
            self.matrix.matrix._cuda_cusparse_spmm_f32(
                program,
                input_array,
                output_array,
                self.rhs_count,
                self._algorithm_code,
            )

    def validate_graph_lifetime(self):
        self.matrix._ensure_valid()  # pylint: disable=W0212

    def plan_info(self):
        """Observe this RHS/algorithm plan without preparing it or synchronizing.

        Workspace belongs to the matrix-owned plan, not to the recording or
        the entire matrix cache. A missing native observation capability is
        reported as unknown, never replaced with another plan's byte count.
        """
        self.matrix._ensure_valid()  # pylint: disable=W0212
        query = getattr(self.matrix.matrix, "_cuda_cusparse_spmm_plan_info", None)
        info = (
            {"status": "available", **dict(query(self.rhs_count, self._algorithm_code))}
            if query is not None
            else {
                "status": "unavailable",
                "prepared": None,
                "preprocess_attempted": None,
                "preprocessed": None,
                "preprocess_error": None,
                "workspace_bytes": None,
            }
        )
        object.__setattr__(self, "_plan_info_snapshot", info)
        return dict(info)

    def memory_report(self):
        lifecycle_state = "ready"
        resident = True
        info = self._plan_info_snapshot
        try:
            info = self.plan_info()
        except TaichiRuntimeError:
            lifecycle_state = "runtime_invalid"
            resident = False
        workspace_bytes = None if info is None else info["workspace_bytes"]
        prepared = None if info is None else info["prepared"]
        resident = resident and prepared is not False
        return make_memory_report(
            "cusparse_spmm_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "retained_spmm_workspace",
                    workspace_bytes,
                    workspace_bytes is not None,
                    "provider_generation",
                    "shared_user_object",
                    resident=resident,
                ),
                HardwareMemoryComponent(
                    "cusparse_handle_and_descriptors",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=resident,
                ),
            ),
            lifecycle_state=lifecycle_state,
            ownership_scope="sparse_matrix_rhs_algorithm_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def _graph_provider_memory_identity(self):
        return (
            "cusparse_spmm_f32",
            id(self.matrix),
            self.rhs_count,
            self.algorithm,
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info=lambda item: {
                "kind": "cuda_cusparse_spmm_f32",
                "shape": item.matrix.shape,
                "rhs_count": item.rhs_count,
                "algorithm": item.algorithm,
            },
        )


def _cusparse_triangular_execution_contract(
    matrix,
    identity,
    operation_id,
    rhs_count,
    fill_mode,
    unit_diagonal,
    transpose,
):
    cache_key = (
        int(impl.runtime_generation()),
        operation_id,
        rhs_count,
        fill_mode,
        unit_diagonal,
        transpose,
    )
    cached_contracts = getattr(
        matrix, "_cusparse_triangular_retained_contracts", None
    )
    if cached_contracts is None:
        cached_contracts = {}
        matrix._cusparse_triangular_retained_contracts = cached_contracts
    cached = cached_contracts.get(cache_key)
    if cached is not None:
        return cached

    runtime_stats = matrix._debug_runtime_stats()  # pylint: disable=W0212
    pattern_version = runtime_stats["identity"]["pattern_version"]
    provider = runtime_stats["provider"]
    provider_scope = passive_dynamic_provider_scope(
        "cusparse",
        "cusparse-dynamic-symbols-v1",
        version=provider["library_version"],
    )
    update_fact = (
        "spsv_value_update_available"
        if rhs_count == 1
        else "spsm_value_update_available"
    )
    value_update = (
        "in_place_retained_plan_update"
        if provider[update_fact]
        else "immutable_after_triangular_analysis"
    )
    retained_identity = make_retained_plan_identity(
        operation_id,
        "cusparse",
        "cuda",
        provider_scope=provider_scope,
        problem_scope={
            "rows": matrix.n,
            "columns": matrix.m,
            "nonzeros": runtime_stats["identity"]["nnz"],
            "rhs_count": rhs_count,
            "storage_format": identity["storage_format"],
            "dtype": "f32",
            "topology_fingerprint": matrix._topology_fingerprint,
            "pattern_id": runtime_stats["identity"]["pattern_id"],
            "pattern_version": pattern_version,
            "resource_object_token": id(matrix.matrix),
            "resource_generation": pattern_version,
        },
        execution_scope={
            "algorithm": "default",
            "fill_mode": fill_mode,
            "unit_diagonal": unit_diagonal,
            "transpose": transpose,
            "value_update": value_update,
            "stream_binding": "runtime_ordered",
            "capture_compatible": True,
        },
    )
    scale_dimensions = ["rows", "nonzeros", "dependency_depth"]
    if rhs_count > 1:
        scale_dimensions.append("rhs_count")
    contract = RetainedExecutionContract(
        identity=retained_identity,
        cost_model=HardwareExecutionCostModel(
            (
                fixed_cost("provider_library_load", "process"),
                fixed_cost("handle_and_descriptors", "provider_generation"),
                fixed_cost("triangular_analysis", "provider_generation"),
                fixed_cost("workspace_allocation", "provider_generation"),
                fixed_cost("graph_capture", "graph_instance"),
                scale_cost("triangular_solve", *scale_dimensions),
            )
        ),
        workspace_ownership="provider_generation",
        concurrency_policy="single_inflight",
        automatic_selection_policy="forbidden",
    )
    result = (contract, dict(runtime_stats["resources"]))
    cached_contracts[cache_key] = result
    return result


class _CusparseTriangularRecording(BackendCommandRecording):
    _FILL_MODES = {"lower": 0, "upper": 1}

    def __init__(
        self,
        matrix,
        rhs_count,
        operation_id,
        provider_fact,
        *,
        fill_mode="lower",
        unit_diagonal=False,
        transpose=False,
        algorithm="default",
        input="input",
        output="output",
    ):
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuSPARSE triangular matrix must be a SparseMatrix")
        if isinstance(rhs_count, bool) or not isinstance(rhs_count, int):
            raise TypeError("CUDA cuSPARSE triangular rhs_count must be an integer")
        if rhs_count < 1 or rhs_count > 0x7FFFFFFF:
            raise ValueError(
                "CUDA cuSPARSE triangular rhs_count must be in [1, INT_MAX]"
            )
        if fill_mode not in self._FILL_MODES:
            raise ValueError("CUDA cuSPARSE fill_mode must be lower or upper")
        if not isinstance(unit_diagonal, bool):
            raise TypeError("CUDA cuSPARSE unit_diagonal must be a bool")
        if not isinstance(transpose, bool):
            raise TypeError("CUDA cuSPARSE transpose must be a bool")
        if algorithm != "default":
            raise ValueError("CUDA cuSPARSE triangular algorithm must be default")
        matrix._ensure_valid()  # pylint: disable=W0212
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if (
            identity["backend_family"] != "cuda"
            or identity["storage_format"] != "csr"
            or identity["dtype"] != "f32"
            or matrix.n != matrix.m
        ):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE triangular solve requires a square scalar f32 "
                "CUDA CSR SparseMatrix"
            )
        runtime_stats = matrix._debug_runtime_stats()  # pylint: disable=W0212
        if not runtime_stats["provider"][provider_fact]:
            raise TaichiRuntimeError(
                "The loaded cuSPARSE provider does not expose the retained "
                "triangular solve and value-update symbol contract"
            )
        names = (input, output)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuSPARSE binding names must be nonempty strings")
        if input == output:
            raise ValueError("CUDA cuSPARSE binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
            no_host_readback=True,
        )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "rhs_count", rhs_count)
        object.__setattr__(self, "fill_mode", fill_mode)
        object.__setattr__(self, "_fill_mode_code", self._FILL_MODES[fill_mode])
        object.__setattr__(self, "unit_diagonal", unit_diagonal)
        object.__setattr__(self, "transpose", transpose)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "input", input)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "_operation_id", operation_id)
        object.__setattr__(
            self,
            "_cuda_capture_recipe",
            _CusparseTriangularCaptureRecipe(
                matrix,
                rhs_count,
                self._fill_mode_code,
                unit_diagonal,
                transpose,
                input,
                output,
            ),
        )
        retained_contract, memory_resources = (
            _cusparse_triangular_execution_contract(
                matrix,
                identity,
                operation_id,
                rhs_count,
                fill_mode,
                unit_diagonal,
                transpose,
            )
        )
        attach_retained_execution_contract(self, retained_contract)
        object.__setattr__(self, "_memory_resources", memory_resources)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.input, GraphAccess.READ),
            ResourceEffect(self.output, GraphAccess.WRITE),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuSPARSE triangular solve")
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuSPARSE triangular solve requires the CUDA backend; "
                f"the active backend is {active_backend()}"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError(
                "CUDA cuSPARSE triangular solve requires an active runtime"
            )
        self.matrix._ensure_valid()  # pylint: disable=W0212
        input_value = bindings[self.input]
        output_value = bindings[self.output]
        shape = (
            (self.matrix.n,)
            if self.rhs_count == 1
            else (self.matrix.n, self.rhs_count)
        )
        input_array = CusparseSpmvRecording._validate_array(
            input_value, self.input, shape
        )
        output_array = CusparseSpmvRecording._validate_array(
            output_value, self.output, shape
        )
        if (
            input_value._runtime_allocation_identity
            == output_value._runtime_allocation_identity
        ):
            raise TaichiRuntimeError(
                "CUDA cuSPARSE triangular output must not alias the input"
            )
        with hardware_provider_call("cusparse"):
            if self.rhs_count == 1:
                self.matrix.matrix._cuda_cusparse_spsv_f32(
                    program,
                    input_array,
                    output_array,
                    self._fill_mode_code,
                    self.unit_diagonal,
                    self.transpose,
                )
            else:
                self.matrix.matrix._cuda_cusparse_spsm_f32(
                    program,
                    input_array,
                    output_array,
                    self.rhs_count,
                    self._fill_mode_code,
                    self.unit_diagonal,
                    self.transpose,
                )

    def validate_graph_lifetime(self):
        self.matrix._ensure_valid()  # pylint: disable=W0212

    def memory_report(self):
        lifecycle_state = "ready"
        resident = True
        resources = self._memory_resources
        try:
            resources = dict(
                self.matrix._debug_runtime_stats()["resources"]  # pylint: disable=W0212
            )
            object.__setattr__(self, "_memory_resources", resources)
        except TaichiRuntimeError:
            lifecycle_state = "runtime_invalid"
            resident = False
        prefix = "spsv" if self.rhs_count == 1 else "spsm"
        workspace_bytes = int(
            resources.get(f"{prefix}_workspace_reserved_bytes", 0)
        )
        return make_memory_report(
            f"cusparse_{prefix}_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    f"retained_{prefix}_workspace",
                    workspace_bytes,
                    True,
                    "provider_generation",
                    "shared_user_object",
                    resident=resident,
                ),
                HardwareMemoryComponent(
                    "cusparse_triangular_analysis_and_descriptors",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=resident,
                ),
            ),
            lifecycle_state=lifecycle_state,
            ownership_scope="sparse_matrix_triangle_rhs_generation",
        )

    def _graph_provider_memory_report(self):
        return self.memory_report()

    def _graph_provider_memory_identity(self):
        return (
            self._operation_id,
            id(self.matrix),
            self.rhs_count,
            self.fill_mode,
            self.unit_diagonal,
            self.transpose,
        )

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info=lambda item: {
                "kind": (
                    "cuda_cusparse_spsv_f32"
                    if item.rhs_count == 1
                    else "cuda_cusparse_spsm_f32"
                ),
                "shape": item.matrix.shape,
                "rhs_count": item.rhs_count,
                "fill_mode": item.fill_mode,
                "unit_diagonal": item.unit_diagonal,
                "transpose": item.transpose,
            },
        )


@instrument_hardware_recording("linalg.spsv.cusparse_explicit")
class CusparseSpsvRecording(_CusparseTriangularRecording):
    """Retained f32 CSR triangular solve for one nonsingular right-hand side."""

    def __init__(self, matrix, **options):
        super().__init__(
            matrix,
            1,
            "linalg.spsv.cusparse_explicit",
            "spsv_f32_available",
            **options,
        )


@instrument_hardware_recording("linalg.spsm.cusparse_explicit")
class CusparseSpsmRecording(_CusparseTriangularRecording):
    """Retained f32 CSR triangular solve for multiple nonsingular RHSs."""

    def __init__(self, matrix, rhs_count, **options):
        if isinstance(rhs_count, bool) or not isinstance(rhs_count, int):
            raise TypeError("CUDA cuSPARSE SpSM rhs_count must be an integer")
        if rhs_count < 2 or rhs_count > 0x7FFFFFFF:
            raise ValueError("CUDA cuSPARSE SpSM rhs_count must be in [2, INT_MAX]")
        super().__init__(
            matrix,
            rhs_count,
            "linalg.spsm.cusparse_explicit",
            "spsm_f32_available",
            **options,
        )


class CudssPlan:
    """Explicit staged cuDSS 0.8.x direct-solver plan for one CUDA CSR matrix.

    This is a Python-scope hardware provider, not a kernel intrinsic. The
    caller controls analysis, factorization/refactorization, and solve. The
    plan is also the explicit provider object used by ``SparseSolver`` when
    its automatic CUDA selection admits cuDSS. Taichi kernels are never
    rewritten to call this plan.
    """

    _MATRIX_TYPES = {"general": 0, "symmetric": 1, "spd": 3}
    _MATRIX_VIEWS = {"full": 0, "lower": 1, "upper": 2}

    def __init__(
        self,
        matrix,
        *,
        matrix_type="general",
        matrix_view=None,
        library_path=None,
    ):
        self._initialize(matrix, matrix_type, matrix_view, library_path, None)

    @classmethod
    def _create_configured(
        cls,
        matrix,
        configuration,
        *,
        matrix_type="general",
        matrix_view=None,
        library_path=None,
        _graph_owned=False,
    ):
        """Private recipe preparation, never a mutable execution option."""
        plan = cls.__new__(cls)
        plan._handle = None
        try:
            plan._initialize(
                matrix,
                matrix_type,
                matrix_view,
                library_path,
                configuration,
                _graph_owned,
            )
            return plan
        except BaseException:
            plan.close()
            raise

    def _initialize(
        self,
        matrix,
        matrix_type,
        matrix_view,
        library_path,
        configuration,
        graph_owned=False,
    ):
        from taichi_forge.hardware._cudss import (  # pylint: disable=C0415
            _register_loaded_plan,
            cudss_dll_directories,
            resolve_cudss_provider,
        )
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuDSS matrix must be a Taichi SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        if active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CUDA cuDSS requires the CUDA backend; the active backend is "
                f"{active_backend()}"
            )
        if matrix_type not in self._MATRIX_TYPES:
            raise ValueError("matrix_type must be general, symmetric, or spd")
        if matrix_view is None:
            matrix_view = "full" if matrix_type == "general" else "lower"
        if matrix_view not in self._MATRIX_VIEWS:
            raise ValueError("matrix_view must be full, lower, or upper")
        if matrix_type == "general" and matrix_view != "full":
            raise ValueError("general cuDSS matrices require matrix_view='full'")
        contract = matrix._get_format_contract()  # pylint: disable=W0212
        identity = contract["identity"]
        if (
            identity["backend_family"] != "cuda"
            or identity["storage_format"] != "csr"
            or matrix.dtype != f32
            or matrix.n != matrix.m
        ):
            raise TaichiRuntimeError(
                "CUDA cuDSS requires a square scalar f32 CUDA CSR matrix; no "
                "conversion or host fallback was performed"
            )
        program = impl.get_runtime().prog
        if program is None:
            raise TaichiRuntimeError("CUDA cuDSS requires an active runtime")
        with hardware_provider_call("cudss", failure_phase="provider_plan_failure"):
            resolved = resolve_cudss_provider(library_path)
            with cudss_dll_directories(resolved.runtime_library_path):
                create = program._create_cuda_cudss_plan
                extra = ()
                if configuration is not None:
                    create = program._create_cuda_cudss_configured_plan
                    extra = (configuration, True) if graph_owned else (configuration,)
                handle = create(
                    matrix.matrix,
                    self._MATRIX_TYPES[matrix_type],
                    self._MATRIX_VIEWS[matrix_view],
                    resolved.adapter_path,
                    resolved.runtime_library_path,
                    *extra,
                )
        self._program = program
        self._runtime_generation = impl.runtime_generation()
        self._matrix = matrix
        self._handle = handle
        self._graph_owned = graph_owned
        self._capture_leases = 0
        self._rows = matrix.n
        self._nnz = int(matrix.matrix.num_nonzero())
        self._effect_name = f"__cudss_plan_{self._runtime_generation}_{self._handle}"
        self.matrix_type = matrix_type
        self.matrix_view = matrix_view
        self.library_path = resolved.runtime_library_path or None
        self.provider_identity = MappingProxyType(
            {
                "library_candidate": resolved.adapter_path,
                "vendor_library_candidate": self.library_path or "system_default",
                "provider_source": "forge_runtime_wheel",
                "provider_abi": "taichi-forge-cudss-provider-c-abi1",
                "provider_version": resolved.provider_version,
                "provider_adapter_binary_sha256": (resolved.adapter_binary_sha256),
                "cudss_header_version": resolved.provider_header_version,
                "provider_name": resolved.provider_name,
                "build_identity": resolved.build_identity,
                "feature_bits": resolved.feature_bits,
            }
        )
        provider_scope = dict(self.provider_identity)
        provider_scope["provider_binary_identity"] = {
            "adapter_sha256": resolved.adapter_binary_sha256,
            # Do not hash a potentially large user-managed DLL during plan
            # creation.  Path + reported version qualify process-local reuse;
            # the absent content hash keeps persistent reuse fail-closed.
            "vendor_sha256": None,
        }
        self._retained_identity = make_retained_plan_identity(
            "linalg.solve.cudss",
            "cudss",
            "cuda",
            provider_scope=provider_scope,
            problem_scope={
                "rows": self._rows,
                "columns": self._rows,
                "nonzeros": self._nnz,
                "storage_format": "csr",
                "dtype": "f32",
                "matrix_type": self.matrix_type,
                "matrix_view": self.matrix_view,
                "topology_fingerprint": matrix._topology_fingerprint,
                "resource_handle": self._handle,
            },
            execution_scope={
                "algorithm": (
                    "cudss_default" if configuration is None else "cudss_configured"
                ),
                **(
                    {}
                    if configuration is None
                    else {"configuration": tuple(configuration)}
                ),
                "workspace_limit_bytes": None,
                "stream_binding": "runtime_ordered",
                "capture_compatible": graph_owned,
            },
        )

        self._solve_execution_contract = RetainedExecutionContract(
            identity=self._retained_identity,
            cost_model=HardwareExecutionCostModel(
                (
                    fixed_cost("provider_library_load", "process"),
                    fixed_cost("plan_creation", "provider_generation"),
                    fixed_cost("analysis", "provider_generation"),
                    fixed_cost("factorization", "provider_generation"),
                    fixed_cost("workspace_allocation", "provider_generation"),
                    scale_cost("solve_execution", "rows", "nonzeros", "rhs_count"),
                )
            ),
            workspace_ownership="provider_generation",
            concurrency_policy="runtime_ordered",
            automatic_selection_policy="qualification_gated",
        )
        self._refactor_solve_execution_contract = RetainedExecutionContract(
            identity=self._retained_identity,
            cost_model=HardwareExecutionCostModel(
                (
                    fixed_cost("provider_library_load", "process"),
                    fixed_cost("plan_creation", "provider_generation"),
                    fixed_cost("analysis", "provider_generation"),
                    fixed_cost("workspace_allocation", "provider_generation"),
                    scale_cost("refactorization", "rows", "nonzeros"),
                    scale_cost("solve_execution", "rows", "nonzeros", "rhs_count"),
                )
            ),
            workspace_ownership="provider_generation",
            concurrency_policy="single_inflight",
            automatic_selection_policy="forbidden",
        )
        _register_loaded_plan(self)

    @property
    def closed(self):
        return self._handle is None

    def _ensure_open(self):
        if self._handle is None:
            raise TaichiRuntimeError("CUDA cuDSS plan is closed")
        if (
            impl.get_runtime().prog is not self._program
            or impl.runtime_generation() != self._runtime_generation
        ):
            raise TaichiRuntimeError(
                "CUDA cuDSS plan cannot be used after its runtime was reset"
            )

    def analyze(self):
        """Run reorder and symbolic factorization for the stored CSR pattern."""

        self._ensure_open()
        self._program._cuda_cudss_analyze(self._handle, self._matrix.matrix)
        return self

    def factorize(self, matrix=None, *, refactorize=False):
        """Factor values from a matrix with the analyzed CSR pattern."""

        self._ensure_open()
        if matrix is None:
            matrix = self._matrix
        from taichi_forge.linalg.sparse_matrix import (  # pylint: disable=C0415
            SparseMatrix,
        )

        if not isinstance(matrix, SparseMatrix):
            raise TypeError("CUDA cuDSS matrix must be a Taichi SparseMatrix")
        matrix._ensure_valid()  # pylint: disable=W0212
        self._program._cuda_cudss_factorize(
            self._handle, matrix.matrix, bool(refactorize)
        )
        self._matrix = matrix
        return self

    def refactorize(self):
        """Refactor updated values while retaining the analyzed pattern."""

        return self.factorize(refactorize=True)

    def compute(self):
        """Run analysis followed by initial numeric factorization."""

        return self.analyze().factorize()

    @staticmethod
    def _validate_vector(value, role, size):
        if not isinstance(value, Ndarray):
            raise TaichiRuntimeError(f"CUDA cuDSS {role} must be a Taichi ndarray")
        if (
            value.dtype != f32
            or tuple(value.element_shape) != ()
            or tuple(value.shape) != (size,)
        ):
            raise TaichiRuntimeError(
                f"CUDA cuDSS {role} must be a compact scalar f32 ndarray with "
                f"shape ({size},)"
            )
        return value.arr

    def solve(self, rhs, solution):
        """Solve into an explicit output ndarray; no host fallback is used."""

        self._ensure_open()
        rhs_array = self._validate_vector(rhs, "right-hand side", self._matrix.n)
        solution_array = self._validate_vector(solution, "solution", self._matrix.n)
        if rhs is solution:
            raise TaichiRuntimeError(
                "The first CUDA cuDSS slice requires distinct rhs and solution arrays"
            )
        self._program._cuda_cudss_solve(
            self._handle, self._matrix.matrix, rhs_array, solution_array
        )
        return solution

    def refactor_solve(self, values, rhs, solution):
        """Refactor explicit fixed-pattern values and immediately solve."""

        self._ensure_open()
        values_array = self._validate_vector(values, "matrix values", self._nnz)
        rhs_array = self._validate_vector(rhs, "right-hand side", self._rows)
        solution_array = self._validate_vector(solution, "solution", self._rows)
        if values is rhs or values is solution or rhs is solution:
            raise TaichiRuntimeError(
                "CUDA cuDSS refactorize+solve values, rhs, and solution "
                "arrays must be distinct"
            )
        self._program._cuda_cudss_refactor_solve(
            self._handle, values_array, rhs_array, solution_array
        )
        return solution

    def recording(self, *, rhs="rhs", solution="solution"):
        """Return a root-Graph native solve action for this factored plan."""

        return CudssSolveRecording(self, rhs=rhs, solution=solution)

    def record_refactor_solve(
        self,
        *,
        values="matrix_values",
        rhs="rhs",
        solution="solution",
    ):
        """Record one root action that refactors current values then solves."""

        return CudssRefactorSolveRecording(
            self, values=values, rhs=rhs, solution=solution
        )

    def validate_graph_lifetime(self, *, allow_explicit_values=False):
        """Fail closed when a compiled Graph outlives or invalidates the plan."""

        self._ensure_open()
        statistics = self.statistics()
        if statistics["refactor_solve_inflight"]:
            raise TaichiRuntimeError(
                "CUDA cuDSS refactorize+solve transaction is in flight"
            )
        if not statistics["factorized"] and not allow_explicit_values:
            raise TaichiRuntimeError(
                "CUDA cuDSS Graph solve requires a successful factorization"
            )
        if statistics["factorized_from_explicit_values"] and not allow_explicit_values:
            raise TaichiRuntimeError(
                "CUDA cuDSS standalone solve cannot reuse factors from "
                "explicit Graph values; refactor the stored matrix or use "
                "record_refactor_solve()"
            )

    def memory_report(self):
        """Report known shared CSR bytes and opaque provider-owned state."""

        lifecycle_state = "ready"
        if self._handle is None:
            lifecycle_state = "closed"
        elif impl.get_runtime().prog is not self._program:
            lifecycle_state = "runtime_invalid"
        csr_bytes = (self._rows + 1 + self._nnz) * 4 + self._nnz * 4
        return make_memory_report(
            "cudss_csr_f32",
            "cuda",
            (
                HardwareMemoryComponent(
                    "shared_csr_storage",
                    csr_bytes,
                    True,
                    "provider_generation",
                    "shared_user_object",
                    resident=lifecycle_state == "ready",
                ),
                HardwareMemoryComponent(
                    "analysis_factors_and_workspace",
                    None,
                    False,
                    "provider_generation",
                    "provider",
                    resident=lifecycle_state == "ready",
                ),
            ),
            lifecycle_state=lifecycle_state,
            ownership_scope="provider_plan",
        )

    def statistics(self):
        """Return staged lifecycle state without synchronizing the device."""

        self._ensure_open()
        return dict(self._program._cuda_cudss_plan_statistics(self._handle))

    def _configuration_report(self):
        """Return cached preparation facts without invoking the vendor runtime."""
        self._ensure_open()
        from taichi_forge.hardware._cudss_config import (  # pylint: disable=C0415
            configuration_report,
        )

        return configuration_report(
            self._program._cuda_cudss_plan_configuration(self._handle)
        )

    def _debug_fail_next_refactor_solve(self):
        """Inject one post-provider refactor failure for lifecycle tests."""

        self._ensure_open()
        self._program._debug_cuda_cudss_fail_next_refactor_solve(self._handle)

    def close(self):
        """Release this owner; in-flight work retains state until completion."""

        if getattr(self, "_capture_leases", 0):
            raise TaichiRuntimeError("cuDSS plan is retained by a Graph recording")
        handle = self._handle
        self._handle = None
        if handle is not None and impl.get_runtime().prog is self._program:
            self._program._destroy_cuda_cudss_plan(handle)
        self._matrix = None
        self._program = None

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@instrument_hardware_recording("linalg.solve.cudss", runtime_resource=True)
class CudssSolveRecording(BackendCommandRecording):
    """One factored cuDSS solve re-recorded by a root Forge Graph replay."""

    def __init__(self, plan, *, rhs="rhs", solution="solution"):
        if not isinstance(plan, CudssPlan):
            raise TypeError("CUDA cuDSS Graph solve requires a CudssPlan")
        plan.validate_graph_lifetime()
        names = (rhs, solution)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("CUDA cuDSS binding names must be nonempty strings")
        if rhs == solution:
            raise ValueError("CUDA cuDSS binding names must be unique")
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "solution", solution)
        attach_retained_execution_contract(
            self,
            plan._solve_execution_contract,  # pylint: disable=W0212
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.rhs, GraphAccess.READ),
            ResourceEffect(self.solution, GraphAccess.WRITE),
            ResourceEffect(
                self.plan._effect_name,
                GraphAccess.READ_WRITE,
                runtime_bound=False,
            ),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuDSS")
        self.plan.validate_graph_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            self.plan.solve(bindings[self.rhs], bindings[self.solution])

    def validate_graph_lifetime(self):
        self.plan.validate_graph_lifetime()

    def _graph_provider_memory_report(self):
        return self.plan.memory_report()

    def _graph_provider_memory_identity(self):
        return ("cudss_plan", id(self.plan))

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item.plan,),
            debug_info=lambda item: {
                "kind": "cuda_cudss_solve_f32",
                "shape": (item.plan._rows, item.plan._rows),
                "replay_mode": "runtime_ordered_rerecord",
            },
        )


@instrument_hardware_recording("linalg.refactor_solve.cudss", runtime_resource=True)
class CudssRefactorSolveRecording(BackendCommandRecording):
    """One transactional fixed-pattern cuDSS refactorize+solve root action."""

    def __init__(
        self,
        plan,
        *,
        values="matrix_values",
        rhs="rhs",
        solution="solution",
    ):
        if not isinstance(plan, CudssPlan):
            raise TypeError("CUDA cuDSS Graph refactorize+solve requires a CudssPlan")
        plan.validate_graph_lifetime(allow_explicit_values=True)
        names = (values, rhs, solution)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(
                "CUDA cuDSS refactorize+solve binding names must be nonempty strings"
            )
        if len(set(names)) != len(names):
            raise ValueError(
                "CUDA cuDSS refactorize+solve binding names must be unique"
            )
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="declared_effects",
            workspace_ownership="provider_generation",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "solution", solution)
        attach_retained_execution_contract(
            self,
            plan._refactor_solve_execution_contract,  # pylint: disable=W0212
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.values, GraphAccess.READ),
            ResourceEffect(self.rhs, GraphAccess.READ),
            ResourceEffect(self.solution, GraphAccess.WRITE),
            ResourceEffect(
                self.plan._effect_name,
                GraphAccess.READ_WRITE,
                runtime_bound=False,
            ),
        )

    def execute(self, bindings):
        validate_exact_bindings(self, bindings, "CUDA cuDSS refactorize+solve")
        self.plan.validate_graph_lifetime(allow_explicit_values=True)
        with hardware_failure_phase("provider_execution_failure"):
            self.plan.refactor_solve(
                bindings[self.values],
                bindings[self.rhs],
                bindings[self.solution],
            )

    def validate_graph_lifetime(self):
        self.plan.validate_graph_lifetime(allow_explicit_values=True)

    def _graph_provider_memory_report(self):
        return self.plan.memory_report()

    def _graph_provider_memory_identity(self):
        return ("cudss_plan", id(self.plan))

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            lifetime_leases=lambda item: (item,),
            debug_info=lambda item: {
                "kind": "cuda_cudss_refactor_solve_f32",
                "rows": item.plan._rows,
                "nonzeros": item.plan._nnz,
                "transaction_policy": "single_inflight_fail_closed",
            },
        )


def cudss_is_available(*, library_path=None):
    """Probe the optional tested cuDSS provider without installing it."""

    if impl.get_runtime().prog is None or active_backend() != "cuda":
        return False
    from taichi_forge.hardware._capabilities import probe  # pylint: disable=C0415

    report = probe("cudss", library_path=library_path)
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.solve.cudss"
    )
    return operation.discovery == "available"


__all__ = [
    "CudssPlan",
    "CudssRefactorSolveRecording",
    "CudssSolveRecording",
    "CublasGemmRecording",
    "CusparseSpmmRecording",
    "CusparseSpmvRecording",
    "CusparseSpsmRecording",
    "CusparseSpsvRecording",
    "cublas_is_available",
    "cusparse_is_available",
    "cusparse_spmm_is_available",
    "cusparse_spsm_is_available",
    "cusparse_spsv_is_available",
    "cudss_is_available",
    "gemm_f32",
    "is_available",
    "spmv_f32",
    "spmm_f32",
    "spsm_f32",
    "spsv_f32",
]
