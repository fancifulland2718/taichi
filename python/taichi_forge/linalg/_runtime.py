"""Capability-qualified runtime linear operators and solve plans.

The stable ``ti.linalg.LinearOperator`` binds one current Taichi runtime,
retains a scalar-flat mathematical ABI, and never changes providers through a
hidden fallback. Public apply/solve boundaries accept scalar 1-D ndarrays or
qualified dense-storage vector views without host staging.
"""

import copy
from dataclasses import dataclass
import json
import math
import operator as _operator
import platform
import threading
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    RuntimeBinding,
    TemporaryRequirement,
)
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
    PreparedGraphBindings,
    ProviderOwnedNdarrayBinding,
)
from taichi_forge.graph._graph import _normalize_submission_telemetry_mode
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.impl import get_runtime
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.lang._storage_view import (
    DenseNdarrayView,
    _flatten_storage_to_scalar_vector,
    _slice_storage_description,
    analyze_storage_alias,
    describe_storage,
)
from taichi_forge.linalg._vector_io import (
    VectorView,
    _VectorIOCache,
    vector_io_capabilities as _vector_io_capabilities,
    vector_view,
)
from taichi_forge.linalg.sparse_matrix import SparseMatrix
from taichi_forge.types import f32, f64, i32


@dataclass(frozen=True)
class OperatorTraits:
    """Mathematical claims attached to an operator; ``None`` means unknown."""

    self_adjoint: Optional[bool] = None
    positive_definite: Optional[bool] = None
    positive_semidefinite: Optional[bool] = None
    singular: Optional[bool] = None

    def __post_init__(self):
        for name in (
            "self_adjoint",
            "positive_definite",
            "positive_semidefinite",
            "singular",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"OperatorTraits.{name} must be bool or None")
        if self.positive_definite is True and self.self_adjoint is False:
            raise ValueError(
                "positive_definite=True conflicts with self_adjoint=False"
            )
        if self.positive_definite is True and self.singular is True:
            raise ValueError(
                "positive_definite=True conflicts with singular=True"
            )

    @classmethod
    def spd(cls):
        """Returns the standard symmetric-positive-definite claim set."""
        return cls(True, True, True, False)

    def _native_values(self):
        def encode(value):
            return -1 if value is None else int(value)

        return tuple(
            encode(getattr(self, name))
            for name in (
                "self_adjoint",
                "positive_definite",
                "positive_semidefinite",
                "singular",
            )
        )


@dataclass(frozen=True)
class OperatorCapabilities:
    forward_apply: bool
    adjoint_apply: bool
    native_generalized_apply: bool
    asynchronous_submit: bool
    explicit_sequence: bool
    compiled_graph: bool
    runtime_capture: bool
    binding_rebind: bool
    persistent_workspace: bool
    dense_storage_operands: bool
    dense_storage_affine_operands: bool


class OperatorQualificationReport:
    """Immutable, JSON-serializable LinearOperator qualification evidence."""

    SCHEMA = "taichi_forge.linalg.operator_qualification.v1"

    def __init__(self, record):
        self._record = copy.deepcopy(record)

    @property
    def passed(self):
        return bool(self._record["passed"])

    @property
    def record(self):
        return _readonly_copy(self._record)

    def to_dict(self):
        """Returns a detached mutable copy suitable for persistence."""
        return copy.deepcopy(self._record)

    def to_json(self, *, indent=2):
        """Serializes the qualification record without writing a file."""
        return json.dumps(self._record, indent=indent, sort_keys=True)


@dataclass(frozen=True)
class SolveResult:
    """Immutable terminal snapshot returned by :meth:`SolvePlan.solve`."""

    solution: object
    status_code: int
    termination_reason: str
    converged: bool
    breakdown: bool
    reached_max_iterations: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    absolute_tolerance: float
    relative_tolerance: float
    relative_reference_norm: float
    effective_tolerance: float
    breakdown_reason: str


class _SolvePlanGraphActionRecord:
    """Host telemetry at enclosing-Graph and explicit terminal boundaries."""

    def __init__(self, *, serial, name, method):
        self._lock = threading.Lock()
        self._serial = int(serial)
        self._name = name
        self._method = method
        self._compiled_executables = 0
        self._enclosing_submissions = 0
        self._observed_completions = 0
        self._terminal_snapshots = 0
        self._terminal_iteration_sum = 0
        self._last_terminal_iterations = None
        self._last_terminal_status = "not_materialized"

    def record_compiled(self):
        with self._lock:
            self._compiled_executables += 1

    def begin_submission(self):
        with self._lock:
            self._enclosing_submissions += 1
        return self

    def record_synchronous_execution(self):
        with self._lock:
            self._enclosing_submissions += 1
            self._observed_completions += 1

    def _observe_graph_completion(self):
        with self._lock:
            self._observed_completions += 1

    def record_terminal_snapshot(self, snapshot):
        with self._lock:
            self._terminal_snapshots += 1
            self._terminal_iteration_sum += int(snapshot.iterations)
            self._last_terminal_iterations = int(snapshot.iterations)
            self._last_terminal_status = snapshot.termination_reason

    def statistics(self):
        with self._lock:
            return {
                "serial": self._serial,
                "name": self._name,
                "method": self._method,
                "compiled_executables": self._compiled_executables,
                "enclosing_graph_submissions": self._enclosing_submissions,
                "observed_completions": self._observed_completions,
                "terminal_snapshots": self._terminal_snapshots,
                "terminal_iteration_sum": self._terminal_iteration_sum,
                "last_terminal_iterations": self._last_terminal_iterations,
                "last_terminal_status": self._last_terminal_status,
                "terminal_scope": "last_action_invocation_in_packet",
            }


class _CompletedSolvePlanTicket:
    __slots__ = ()

    def done(self):
        return True

    def wait(self):
        return None

    def telemetry(self):
        return None

    def pipeline_report(self):
        return None

    @property
    def backend(self):
        return "cpu"

    @property
    def sequence(self):
        return None

    @property
    def workspace_lane(self):
        return 0

    @property
    def _has_backend_work(self):
        return False


class SolvePlanSubmission:
    """One asynchronous single-system :class:`SolvePlan` invocation.

    On CUDA/Vulkan, ``wait()`` waits only for backend completion and
    ``result()`` additionally materializes the submission-owned device
    terminal packet exactly once. CPU submissions contain an already-complete
    native result. The wrapper retains every runtime operand, the optional
    cached Graph, and the plan until completion, including when the user drops
    the object early.
    """

    __slots__ = (
        "_absolute_tolerance",
        "_graph",
        "_initial_guess",
        "_lock",
        "_packet",
        "_plan",
        "_relative_tolerance",
        "_result",
        "_rhs",
        "_solution",
        "_ticket",
    )

    def __init__(
        self,
        plan,
        graph,
        ticket,
        packet,
        rhs,
        solution,
        initial_guess,
        completed_result=None,
    ):
        self._plan = plan
        self._graph = graph
        self._ticket = ticket
        self._packet = packet
        self._rhs = rhs
        self._solution = solution
        self._initial_guess = initial_guess
        self._absolute_tolerance = float(plan.atol)
        self._relative_tolerance = float(plan.rtol)
        self._result = completed_result
        self._lock = threading.Lock()

    def done(self):
        """Return whether backend work is complete without terminal readback."""

        return self._ticket.done()

    def wait(self):
        """Wait for backend completion without reading the terminal packet."""

        self._ticket.wait()

    def result(self):
        """Wait and return the immutable solve result and terminal snapshot."""

        with self._lock:
            if self._result is None:
                self._ticket.wait()
                snapshot = self._packet.snapshot()
                self._result = SolveResult(
                    solution=self._solution,
                    absolute_tolerance=self._absolute_tolerance,
                    relative_tolerance=self._relative_tolerance,
                    **snapshot.__dict__,
                )
                plan = self._plan
                lock = getattr(plan, "_submission_lock", None)
                if lock is not None:
                    with lock:
                        plan._submission_terminal_materializations += 1
            return self._result

    def telemetry(self):
        """Wait if needed and return opt-in Graph submission telemetry."""

        return self._ticket.telemetry()

    def pipeline_report(self):
        """Return this submission's opt-in execution-pipeline report."""

        return self._ticket.pipeline_report()

    @property
    def terminal_packet(self):
        """The device terminal packet, or ``None`` for completed CPU work."""

        return self._packet

    @property
    def backend(self):
        return self._ticket.backend

    @property
    def sequence(self):
        return self._ticket.sequence

    @property
    def workspace_lane(self):
        return self._ticket.workspace_lane


@dataclass(frozen=True)
class SolveGraphTerminalSnapshot:
    """Host snapshot of one completed recordable SolvePlan invocation."""

    status_code: int
    termination_reason: str
    converged: bool
    breakdown: bool
    reached_max_iterations: bool
    iterations: int
    initial_residual_norm: float
    residual_norm: float
    relative_reference_norm: float
    effective_tolerance: float
    breakdown_reason: str


class SolveGraphTerminalPacket:
    """Runtime storage for a :class:`SolveGraphTerminal` resource pair."""

    def __init__(self, terminal, *, initialize=True):
        self._terminal = terminal
        self._action_record = terminal._action_record
        self._program = _current_program()
        self._submission_ticket = None
        self._initialized = bool(initialize)
        self._snapshot_recorded = False
        self._snapshot_lock = threading.Lock()
        self.state = ScalarNdarray(i32, (4,))
        self.metrics = ScalarNdarray(f32, (4,))
        if initialize:
            self.state.fill(0)
            self.metrics.fill(0)
        self._arguments = MappingProxyType(
            {
                terminal.state.name: self.state,
                terminal.metrics.name: self.metrics,
            }
        )

    @property
    def arguments(self):
        """Runtime Graph arguments owned by this packet."""

        if self._program is not _current_program():
            raise TaichiRuntimeError(
                "Solve Graph terminal packet belongs to another runtime"
            )
        return self._arguments

    def snapshot(self):
        """Read a completed packet; call after its SubmissionTicket completes."""

        with self._snapshot_lock:
            if self._program is not _current_program():
                raise TaichiRuntimeError(
                    "Solve Graph terminal packet belongs to another runtime"
                )
            if not self._initialized and self._submission_ticket is None:
                raise TaichiRuntimeError(
                    "Solve Graph terminal packet has not been submitted"
                )
            # State and metrics are separate typed resources, but they form a
            # single public terminal packet. Read both through one backend
            # transfer transaction so a completed GPU solve does not pay two
            # stream flush/readback boundaries during snapshot materialization.
            state = np.empty((4,), dtype=np.int32)
            metrics = np.empty((4,), dtype=np.float32)
            self._program.copy_ndarrays_to_host(
                (self.state.arr, self.metrics.arr),
                (state, metrics),
            )
            if int(state[3]) != 1:
                raise TaichiRuntimeError(
                    "Solve Graph terminal packet has not been completed"
                )
            status = int(state[0])
            if status == 2:
                reason = "converged"
            elif status == 1:
                reason = "breakdown"
            elif status == 0:
                reason = "max_iterations"
            else:
                raise TaichiRuntimeError(
                    f"Solve Graph terminal packet has invalid status {status}"
                )
            snapshot = SolveGraphTerminalSnapshot(
                status_code=status,
                termination_reason=reason,
                converged=status == 2,
                breakdown=status == 1,
                reached_max_iterations=status == 0,
                iterations=int(state[1]),
                initial_residual_norm=math.sqrt(max(float(metrics[0]), 0.0)),
                residual_norm=math.sqrt(max(float(metrics[1]), 0.0)),
                relative_reference_norm=math.sqrt(max(float(metrics[2]), 0.0)),
                effective_tolerance=math.sqrt(max(float(metrics[3]), 0.0)),
                breakdown_reason=(
                    "alpha_denominator" if int(state[2]) != 0 else "none"
                ),
            )
            if self._action_record is not None and not self._snapshot_recorded:
                self._action_record.record_terminal_snapshot(snapshot)
                self._snapshot_recorded = True
            return snapshot

    def _attach_submission(self, ticket):
        if self._submission_ticket is not None:
            raise TaichiRuntimeError(
                "Solve Graph terminal packet is already submitted"
            )
        self._submission_ticket = ticket


class SolveGraphTerminal:
    """Symbolic, device-resident terminal state for one SolvePlan action."""

    def __init__(self, state_name, metrics_name, *, action_record=None):
        from taichi_forge.graph._graph import Arg, ArgKind

        self.state = Arg(ArgKind.NDARRAY, state_name, i32, ndim=1)
        self.metrics = Arg(ArgKind.NDARRAY, metrics_name, f32, ndim=1)
        self._action_record = action_record

    def allocate(self, *, initialize=True):
        """Allocate one independently submitable runtime terminal packet."""

        return SolveGraphTerminalPacket(self, initialize=initialize)


def _current_program():
    program = get_runtime().prog
    if program is None:
        raise TaichiRuntimeError(
            "ti.init() must be called before constructing a LinearOperator"
        )
    return program


def _require_positive_size(size, role="size"):
    if isinstance(size, bool):
        raise TaichiRuntimeError(f"LinearOperator {role} must be positive")
    try:
        result = _operator.index(size)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"LinearOperator {role} must be a positive integer"
        ) from exc
    if result <= 0:
        raise TaichiRuntimeError(f"LinearOperator {role} must be positive")
    return result


def _normalize_operator_shape(size):
    if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        if len(size) != 2:
            raise TaichiRuntimeError(
                "LinearOperator shape must contain (range, domain) extents"
            )
        return (
            _require_positive_size(size[0], "range extent"),
            _require_positive_size(size[1], "domain extent"),
        ), False
    extent = _require_positive_size(size)
    return (extent, extent), True


def _require_current_scalar_ndarray(value, role, size=None, dtype=None):
    if not isinstance(value, ScalarNdarray):
        raise TaichiRuntimeError(
            f"{role} must be a scalar Taichi ndarray; implicit host "
            "transfers are not performed"
        )
    if value.arr is None or value._runtime_prog is not _current_program():
        raise TaichiRuntimeError(
            f"{role} belongs to an inactive or different Taichi runtime"
        )
    if len(value.shape) != 1:
        raise TaichiRuntimeError(f"{role} must be one-dimensional")
    if size is not None and value.shape != (size,):
        raise TaichiRuntimeError(
            f"{role} must have shape ({size},), got {value.shape}"
        )
    if dtype is not None and value.dtype != dtype:
        raise TaichiRuntimeError(
            f"{role} must have dtype {dtype}, got {value.dtype}"
        )
    return value


@dataclass(frozen=True)
class _VectorOperand:
    public: object
    array: ScalarNdarray
    view: Optional[VectorView]
    exact_key: tuple
    alias_owner_key: tuple


@dataclass(frozen=True)
class _DirectStorageOperand:
    public: object
    description: object
    runtime_argument: object
    storage_kind: str
    execution_mode: str
    alias_owner_key: object


@dataclass(frozen=True)
class _GraphKrylovOperand:
    public: object
    runtime_value: object
    staged: Optional[_VectorOperand]
    description: object
    storage_kind: str
    execution_mode: str
    exact_key: tuple
    alias_owner_key: object
    direct_dense: bool


def _graph_krylov_staged_operand(operand):
    description = _flatten_storage_to_scalar_vector(describe_storage(operand.array))
    if description.descriptor is None:
        raise TaichiRuntimeError("Graph Krylov staging ndarray could not be described")
    return _GraphKrylovOperand(
        public=operand.public,
        runtime_value=operand.array,
        staged=operand,
        description=description,
        storage_kind="ndarray",
        execution_mode="kDirectContiguous",
        exact_key=operand.exact_key,
        alias_owner_key=operand.alias_owner_key,
        direct_dense=False,
    )


def _try_direct_graph_krylov_operand(value, role, size, dtype, vector_cache, binding_cache):
    if isinstance(value, ScalarNdarray):
        return None
    normalized = _direct_scalar_storage_description(value, role, vector_cache)
    if normalized is None:
        return None
    public_view, description, storage_kind, alias_owner_key = normalized
    descriptor = description.descriptor
    properties = description.properties
    if (
        descriptor.scalar_type != dtype
        or int(properties["scalar_count"]) != int(size)
        or tuple(descriptor.index_shape) != (int(size),)
        or tuple(descriptor.element_shape)
    ):
        raise TaichiRuntimeError(f"{role} must have scalar dtype {dtype} and extent {size}")

    source = public_view._field if isinstance(public_view, VectorView) else value
    cached = binding_cache.get(role)
    if cached is not None and cached[0] is value and cached[1] == int(size):
        dense_view, execution_mode, exact_key = cached[2:]
    else:
        try:
            dense_view = DenseNdarrayView(source, description)
            if _current_program().config().arch == _ti_core.Arch.cuda:
                try:
                    argument = dense_view._runtime_storage_argument("graph_capture", "capture")
                except ValueError:
                    argument = dense_view._runtime_storage_argument("graph_replay", "replay")
            else:
                argument = dense_view._runtime_storage_argument("graph_replay", "replay")
        except ValueError:
            return None
        qualification = dict(argument.qualification)
        execution_mode = qualification["execution_mode"]
        if (
            not qualification["zero_copy_qualified"]
            or not qualification["replayable"]
            or qualification["reason"] != "kNone"
            or execution_mode != "kDirectContiguous"
        ):
            return None
        if isinstance(public_view, VectorView):
            exact_key = ("dense_field", public_view._exact_view_key)
        else:
            exact_key = (
                "dense_storage",
                tuple(descriptor.resource_identity),
                int(descriptor.fingerprint),
            )
        binding_cache[role] = (value, int(size), dense_view, execution_mode, exact_key)
    return _GraphKrylovOperand(
        public=value,
        runtime_value=dense_view,
        staged=None,
        description=description,
        storage_kind=storage_kind,
        execution_mode=execution_mode,
        exact_key=exact_key,
        alias_owner_key=alias_owner_key,
        direct_dense=True,
    )


def _graph_krylov_operands_overlap(left, right):
    alias = analyze_storage_alias(left.description, right.description)
    return alias != "kProvenDisjoint" and left.alias_owner_key == right.alias_owner_key


def _graph_krylov_operands_exact(left, right):
    return left.exact_key == right.exact_key


def _direct_scalar_storage_description(value, role, cache):
    if isinstance(value, DenseNdarrayView):
        if value._runtime_prog is not _current_program():
            raise TaichiRuntimeError(
                f"{role} belongs to an inactive or different Taichi runtime"
            )
        description = value.description
        storage_kind = "dense_ndarray_view"
        public = value
        alias_owner_key = (
            "runtime_storage",
            tuple(description.descriptor.resource_identity),
        )
    else:
        view = cache.view(value, role)
        if view is None or view._direct_storage_description is None:
            return None
        description = view._direct_storage_description
        storage_kind = (
            "dense_field_range" if view.ranged else "dense_field"
        )
        public = view
        alias_owner_key = view._alias_owner_key

    flattened = _flatten_storage_to_scalar_vector(description)
    if flattened.supported:
        return public, flattened, storage_kind, alias_owner_key

    descriptor = description.descriptor
    if (
        descriptor is not None
        and len(tuple(descriptor.index_shape)) == 1
        and not tuple(descriptor.element_shape)
    ):
        return public, description, storage_kind, alias_owner_key
    return None


def _require_compatible_vector_view(value, role, size, dtype, cache):
    view = cache.view(value, role)
    if view is None:
        raise TaichiRuntimeError(
            f"{role} must be a one-dimensional scalar Taichi ndarray or a "
            "supported dense field/vector_view; implicit host transfers are "
            "not performed"
        )
    if view.scalar_extent != size:
        raise TaichiRuntimeError(
            f"{role} must have scalar extent {size}, got "
            f"{view.scalar_extent}"
        )
    if view.dtype != dtype:
        raise TaichiRuntimeError(
            f"{role} must have dtype {dtype}, got {view.dtype}"
        )
    return view


def _prepare_vector_input(
    value,
    role,
    size,
    dtype,
    cache,
    staging_role,
):
    if isinstance(value, ScalarNdarray):
        array = _require_current_scalar_ndarray(value, role, size, dtype)
        cache.record_direct_input()
        identity = ("ndarray", id(array.arr))
        return _VectorOperand(value, array, None, identity, identity)
    view = _require_compatible_vector_view(value, role, size, dtype, cache)
    array = cache.pack(view, staging_role, dtype, size)
    return _VectorOperand(
        value,
        array,
        view,
        ("dense_field", view._exact_view_key),
        ("dense_field", view._alias_owner_key),
    )


def _prepare_vector_output(
    value,
    role,
    size,
    dtype,
    cache,
    staging_role,
):
    if isinstance(value, ScalarNdarray):
        array = _require_current_scalar_ndarray(value, role, size, dtype)
        cache.record_direct_output()
        identity = ("ndarray", id(array.arr))
        return _VectorOperand(value, array, None, identity, identity)
    view = _require_compatible_vector_view(value, role, size, dtype, cache)
    array = cache.buffer(staging_role, dtype, size)
    return _VectorOperand(
        value,
        array,
        view,
        ("dense_field", view._exact_view_key),
        ("dense_field", view._alias_owner_key),
    )


def _vector_operands_overlap(left, right):
    if left.alias_owner_key != right.alias_owner_key:
        return False
    if left.view is None or right.view is None:
        return True
    if left.view.indexed or right.view.indexed:
        return True
    left_storage = left.view._direct_storage_description
    right_storage = right.view._direct_storage_description
    if left_storage is None or right_storage is None:
        return True
    return analyze_storage_alias(left_storage, right_storage) != "kProvenDisjoint"


def _vector_operands_exact(left, right):
    return left.exact_key == right.exact_key


def _finish_vector_output(
    cache, operand, dtype, size, synchronized_session=None
):
    if operand.view is not None:
        cache.unpack(operand.array, operand.view, dtype, size)
        get_runtime().sync()
        if synchronized_session is not None:
            synchronized_session._mark_synchronized()
            cache.record_coalesced_operator_sync()
        cache.record_completion_sync()


def _normalized_resource_mapping(values, role, require_nonempty=False):
    values = {} if values is None else values
    if not isinstance(values, Mapping):
        raise TaichiRuntimeError(f"{role} must be a mapping")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(f"{role} names must be non-empty strings")
        result[name] = _require_current_scalar_ndarray(
            value, f"{role}[{name!r}]"
        )
    if require_nonempty and not result:
        raise TaichiRuntimeError(f"{role} must contain at least one ndarray")
    return result


def _normalized_fixed_field_state(values):
    values = {} if values is None else values
    if not isinstance(values, Mapping):
        raise TaichiRuntimeError("state must be a mapping")
    descriptors = {}
    retained = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "state names must be non-empty strings"
            )
        if not isinstance(value, (ScalarField, MatrixField)):
            raise TaichiRuntimeError(
                "state entries must be root-dense scalar, vector, or "
                f"matrix Fields; state[{name!r}] is "
                f"{type(value).__name__}"
            )
        description = describe_storage(value)
        if description.descriptor is None:
            raise TaichiRuntimeError(
                "state entries must be live root-dense scalar, vector, or "
                f"matrix Fields; state[{name!r}] is unavailable: "
                f"{description.failure_reason}"
            )
        descriptors[name] = description.descriptor
        retained[name] = value
    return descriptors, retained


def _readonly_copy(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_copy(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_copy(item) for item in value)
    return value


def _canonical_parameter_scalar(value, dtype, role):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaichiRuntimeError(f"{role} must be finite") from exc
    if not math.isfinite(value):
        raise TaichiRuntimeError(f"{role} must be finite")
    if dtype == f32:
        if abs(value) > float(np.finfo(np.float32).max):
            raise TaichiRuntimeError(f"{role} must be representable as f32")
        value = float(np.float32(value))
    return value


def _normalize_parameter_range(value, initial, role, dtype):
    if value is None:
        raise TaichiRuntimeError(f"{role} requires an explicit finite (minimum, maximum) range")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TaichiRuntimeError(f"{role} must be a finite (minimum, maximum) pair")
    value = tuple(value)
    if len(value) != 2:
        raise TaichiRuntimeError(f"{role} must be a finite (minimum, maximum) pair")
    minimum = _canonical_parameter_scalar(value[0], dtype, f"{role} minimum")
    maximum = _canonical_parameter_scalar(value[1], dtype, f"{role} maximum")
    if minimum > maximum:
        raise TaichiRuntimeError(f"{role} must be a finite ordered closed interval")
    if initial < minimum or initial > maximum:
        raise TaichiRuntimeError(f"{role} does not contain its initial value {initial}")
    return minimum, maximum


class _AffineParameterState:
    def __init__(self, handle, alpha_range, beta_range, dtype):
        self._handle = handle
        self.alpha_range = tuple(alpha_range)
        self.beta_range = tuple(beta_range)
        self.dtype = dtype

    def _validate_values(self, alpha, beta):
        values = []
        for value, bounds, role in (
            (alpha, self.alpha_range, "alpha"),
            (beta, self.beta_range, "beta"),
        ):
            value = _canonical_parameter_scalar(
                value, self.dtype, f"parameterized affine {role}"
            )
            if not math.isfinite(value) or value < bounds[0] or value > bounds[1]:
                raise TaichiRuntimeError(
                    f"parameterized affine {role}={value} is outside its " f"declared range {bounds}"
                )
            values.append(value)
        return tuple(values)

    def update(self, alpha, beta, expected_version):
        alpha, beta = self._validate_values(alpha, beta)
        return self._handle._update_affine_parameters(
            alpha, beta, expected_version, expected_version + 1
        )

    def recordable_generation(self):
        return self._handle._affine_parameter_snapshot()

    def recordable_snapshot(self):
        generation = self.recordable_generation()
        return generation, generation.alpha, generation.beta

    def snapshot(self):
        generation = self.recordable_generation()
        return MappingProxyType(
            {
                "alpha": generation.alpha,
                "beta": generation.beta,
                "alpha_range": self.alpha_range,
                "beta_range": self.beta_range,
                "version": generation.version,
            }
        )


class LinearOperator:
    """A runtime-bound, capability-qualified linear map."""

    _TOKEN = object()

    def __init__(
        self,
        token,
        handle,
        *,
        provider_kind,
        provider_core=None,
        source=None,
        retained=(),
        composition_spec=None,
    ):
        if token is not self._TOKEN:
            raise TypeError("Use a LinearOperator factory method")
        self._program = _current_program()
        self._handle = handle
        self._provider_kind = provider_kind
        self._provider_core = provider_core
        self._source = source
        self._retained = tuple(retained)
        self._composition_spec = composition_spec
        self._parameter_state = None
        self._runtime_storage_arguments = {}
        self._direct_storage_operand_cache = {}
        self._vector_io = _VectorIOCache(
            allow_native_bulk=self._program.config().arch
            in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.vulkan,
            )
        )
        metadata = dict(handle._metadata())
        metadata["capabilities"] = dict(metadata["capabilities"])
        metadata["traits"] = {
            name: dict(claim) for name, claim in metadata["traits"].items()
        }
        metadata["resource_stamp"] = dict(metadata["resource_stamp"])
        self._metadata_snapshot = metadata
        self.shape = tuple(metadata["shape"])
        self.dtype = {"f32": f32, "f64": f64}[metadata["dtype"]]
        self.provider = metadata["provider"]
        self.execution_kind = metadata["execution_kind"]
        self.capabilities = OperatorCapabilities(
            **dict(metadata["capabilities"])
        )
        get_runtime().register_runtime_object(self)

    @classmethod
    def _from_handle(cls, handle, **kwargs):
        return cls(cls._TOKEN, handle, **kwargs)

    @classmethod
    def from_sparse_matrix(cls, matrix, *, traits=None):
        """Binds a fixed CSR/BSR ``SparseMatrix`` without copying."""
        if not isinstance(matrix, SparseMatrix):
            raise TaichiRuntimeError(
                "LinearOperator.from_sparse_matrix expects SparseMatrix"
            )
        matrix._ensure_valid()
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")
        handle = _ti_core._make_linear_operator(
            _current_program(), matrix.matrix, *traits._native_values()
        )
        return cls._from_handle(
            handle,
            provider_kind="stored",
            provider_core=matrix.matrix,
            source=matrix,
            retained=(matrix,),
        )

    @classmethod
    def from_kernel(
        cls,
        kernel,
        size,
        topology,
        *,
        adjoint=None,
        numeric=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Compiles the exact f32 ndarray operator-kernel ABI.

        With ``numeric`` the signature is ``(active_size, topology, numeric,
        input, output)``; otherwise it is ``(active_size, operator_data,
        input, output)``. ``size`` may be an integer square shorthand or a
        ``(range, domain)`` shape. Rectangular operators and explicit adjoints
        use an action provider independent of ``SparseMatrix``. Resources are
        copied into operator-owned snapshots.
        """
        shape, legacy_square = _normalize_operator_shape(size)
        range_extent, domain_extent = shape
        topology = _require_current_scalar_ndarray(topology, "topology")
        if numeric is not None:
            numeric = _require_current_scalar_ndarray(numeric, "numeric")
        topology_version = _require_positive_size(
            topology_version, "topology_version"
        )
        numeric_version = _require_positive_size(
            numeric_version, "numeric_version"
        )
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")

        def compile_action(action, active_size, input_size, output_size, role):
            try:
                primal = action._primal
            except AttributeError as exc:
                raise TaichiRuntimeError(
                    f"LinearOperator.from_kernel {role} must be a @ti.kernel"
                ) from exc
            compile_input = ScalarNdarray(f32, (input_size,))
            compile_output = ScalarNdarray(f32, (output_size,))
            # Compiled-kernel operators may bind qualified affine Field or
            # ndarray views to these two ABI slots. Keep their specialization
            # distinct from branch-free canonical ndarray kernels so direct
            # strided operands use the runtime shape/stride metadata.
            compile_input._runtime_affine_exemplar = True
            compile_output._runtime_affine_exemplar = True
            compile_args = (
                active_size,
                topology,
                compile_input,
                compile_output,
            )
            if numeric is not None:
                compile_args = (
                    active_size,
                    topology,
                    numeric,
                    compile_input,
                    compile_output,
                )
            key = primal.ensure_compiled(*compile_args)
            return primal.compiled_kernels[key]

        kernel_cpp = compile_action(
            kernel,
            range_extent,
            domain_extent,
            range_extent,
            "forward kernel",
        )
        adjoint_cpp = None
        if adjoint is not None:
            adjoint_cpp = compile_action(
                adjoint,
                domain_extent,
                range_extent,
                domain_extent,
                "adjoint kernel",
            )
        program = _current_program()
        if legacy_square and adjoint is None and numeric is None:
            core = program._create_compiled_kernel_linear_operator(
                kernel_cpp,
                range_extent,
                topology_version,
                numeric_version,
                topology.arr,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="kernel",
                provider_core=core,
                source=kernel,
                retained=(kernel, topology),
            )
        if legacy_square and adjoint is None:
            core = program._create_compiled_kernel_linear_operator_with_numeric_data(
                kernel_cpp,
                range_extent,
                topology_version,
                numeric_version,
                topology.arr,
                numeric.arr,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="kernel",
                provider_core=core,
                source=kernel,
                retained=(kernel, topology, numeric),
            )
        handle = _ti_core._make_compiled_kernel_operator(
            program,
            kernel_cpp,
            adjoint_cpp,
            range_extent,
            domain_extent,
            topology_version,
            numeric_version,
            topology.arr,
            None if numeric is None else numeric.arr,
            *traits._native_values(),
        )
        return cls._from_handle(
            handle,
            provider_kind="kernel_action",
            source=(kernel, adjoint),
            retained=(kernel, adjoint, topology, numeric),
        )

    @classmethod
    def from_graph(
        cls,
        graph,
        size,
        *,
        adjoint=None,
        fixed_i32=None,
        topology,
        numeric=None,
        workspace=None,
        state=None,
        topology_version=1,
        numeric_version=1,
        traits=None,
    ):
        """Binds compiled multi-kernel f32 Graph actions.

        Runtime vector arguments must be named ``input`` and ``output``.
        Every other argument is assigned exactly one fixed, topology, numeric,
        or workspace role. ``size`` may be an integer square shorthand or a
        ``(range, domain)`` shape. An explicit adjoint Graph must expose the
        same fixed resource schema. SNode-dependent Graphs are accepted only
        when every distinct dependent SNodeTree is represented in ``state`` by
        a live root-dense scalar, vector, or matrix Field and the complete tree
        is purely dense. State keys are diagnostic labels; dependency matching
        and lifetime ownership are tree-granular. State storage is referenced
        in place and is never snapshotted.
        """
        shape, legacy_square = _normalize_operator_shape(size)
        range_extent, domain_extent = shape
        topology_version = _require_positive_size(
            topology_version, "topology_version"
        )
        numeric_version = _require_positive_size(
            numeric_version, "numeric_version"
        )
        fixed_i32 = {} if fixed_i32 is None else fixed_i32
        if not isinstance(fixed_i32, Mapping):
            raise TaichiRuntimeError("fixed_i32 must be a mapping")
        fixed = {}
        for name, value in fixed_i32.items():
            if not isinstance(name, str) or not name:
                raise TaichiRuntimeError(
                    "fixed_i32 names must be non-empty strings"
                )
            if isinstance(value, bool):
                raise TaichiRuntimeError("fixed_i32 values must be integers")
            try:
                value = _operator.index(value)
            except TypeError as exc:
                raise TaichiRuntimeError(
                    "fixed_i32 values must be integers"
                ) from exc
            if value < -(2**31) or value >= 2**31:
                raise TaichiRuntimeError("fixed_i32 value is outside i32")
            fixed[name] = value
        topology_arrays = _normalized_resource_mapping(
            topology, "topology", require_nonempty=True
        )
        numeric_arrays = _normalized_resource_mapping(numeric, "numeric")
        workspace_arrays = _normalized_resource_mapping(workspace, "workspace")
        state_descriptors, state_fields = _normalized_fixed_field_state(state)
        traits = OperatorTraits() if traits is None else traits
        if not isinstance(traits, OperatorTraits):
            raise TypeError("traits must be OperatorTraits")
        try:
            compiled_graph = graph._compiled_graph
        except AttributeError as exc:
            raise TaichiRuntimeError(
                "LinearOperator.from_graph expects a compiled ti.graph.Graph"
            ) from exc
        compiled_adjoint = None
        if adjoint is not None:
            try:
                compiled_adjoint = adjoint._compiled_graph
            except AttributeError as exc:
                raise TaichiRuntimeError(
                    "LinearOperator.from_graph adjoint must be a compiled "
                    "ti.graph.Graph"
                ) from exc
        program = _current_program()
        topology_native = {
            name: value.arr for name, value in topology_arrays.items()
        }
        numeric_native = {
            name: value.arr for name, value in numeric_arrays.items()
        }
        workspace_native = {
            name: value.arr for name, value in workspace_arrays.items()
        }
        if legacy_square and adjoint is None:
            core = program._create_compiled_graph_linear_operator(
                compiled_graph,
                range_extent,
                topology_version,
                numeric_version,
                fixed,
                topology_native,
                numeric_native,
                workspace_native,
                state_descriptors,
            )
            handle = _ti_core._make_linear_operator(
                program, core, *traits._native_values()
            )
            return cls._from_handle(
                handle,
                provider_kind="graph",
                provider_core=core,
                source=graph,
                retained=(
                    graph,
                    tuple(topology_arrays.values()),
                    tuple(numeric_arrays.values()),
                    tuple(workspace_arrays.values()),
                    tuple(state_fields.values()),
                ),
            )
        handle = _ti_core._make_compiled_graph_operator(
            program,
            compiled_graph,
            compiled_adjoint,
            range_extent,
            domain_extent,
            topology_version,
            numeric_version,
            fixed,
            topology_native,
            numeric_native,
            workspace_native,
            state_descriptors,
            *traits._native_values(),
        )
        return cls._from_handle(
            handle,
            provider_kind="graph_action",
            source=(graph, adjoint),
            retained=(
                graph,
                adjoint,
                tuple(topology_arrays.values()),
                tuple(numeric_arrays.values()),
                tuple(workspace_arrays.values()),
                tuple(state_fields.values()),
            ),
        )

    def _ensure_valid(self):
        if self._handle is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "LinearOperator cannot be used after ti.reset() or with a "
                "different runtime"
            )

    def _invalidate_runtime(self):
        # Release the execution plan before provider snapshots and retained
        # operands while the Program is still alive.
        self._handle = None
        self._provider_core = None
        self._source = None
        self._retained = ()
        self._composition_spec = None
        self._parameter_state = None
        self._vector_io = None
        self._runtime_storage_arguments = None
        self._direct_storage_operand_cache = None
        self._program = None

    @property
    def metadata(self):
        """Returns a read-only construction metadata snapshot."""
        return _readonly_copy(self._metadata_snapshot)

    @property
    def traits(self):
        return _readonly_copy(self._metadata_snapshot["traits"])

    def _runtime_storage_argument(self, description):
        descriptor = description.descriptor
        if descriptor is None or descriptor.source_kind == "kExternalDense":
            return None
        if self._provider_kind in ("graph", "graph_action"):
            if self._program.config().arch == _ti_core.Arch.cuda:
                candidates = (
                    ("graph_capture", "capture", "capturable"),
                    ("graph_replay", "replay", "replayable"),
                )
            else:
                candidates = (("graph_replay", "replay", "replayable"),)
        elif self._provider_kind in ("kernel", "kernel_action"):
            candidates = (("ordinary_kernel", "ordinary", "bindable"),)
        else:
            candidates = (("native_consumer", "ordinary", "bindable"),)

        for consumer, mode, required_capability in candidates:
            key = (int(descriptor.fingerprint), consumer, mode)
            argument = self._runtime_storage_arguments.get(key)
            if argument is None:
                argument = _ti_core._make_runtime_storage_argument(
                    self._program, descriptor, consumer, mode
                )
                self._runtime_storage_arguments[key] = argument
            qualification = dict(argument.qualification)
            if (
                qualification["zero_copy_qualified"]
                and qualification[required_capability]
                and qualification["reason"] == "kNone"
            ):
                return argument
        return None

    def _direct_storage_operand(self, value, role, size, dtype):
        if isinstance(value, DenseNdarrayView):
            if value._runtime_prog is not _current_program():
                raise TaichiRuntimeError(
                    f"{role} belongs to an inactive or different Taichi runtime"
                )
            source_description = value.description
            public = value
            source_kind = "dense_ndarray_view"
        else:
            source_view = self._vector_io.view(value, role)
            if (
                source_view is None
                or source_view._direct_storage_description is None
            ):
                return None
            source_description = source_view._direct_storage_description
            public = source_view
            source_kind = "dense_field"
        source_descriptor = source_description.descriptor
        cache_key = (
            source_kind,
            int(source_descriptor.fingerprint),
            int(size),
            str(dtype),
        )
        cached = self._direct_storage_operand_cache.get(cache_key)
        if cached is not None:
            self._vector_io.record_direct_storage_operand(reused=True)
            return _DirectStorageOperand(public, *cached)

        normalized = _direct_scalar_storage_description(
            public, role, self._vector_io
        )
        if normalized is None:
            return None
        public, description, storage_kind, alias_owner_key = normalized
        descriptor = description.descriptor
        properties = description.properties
        if (
            descriptor.scalar_type != dtype
            or int(properties["scalar_count"]) != int(size)
            or tuple(descriptor.index_shape) != (int(size),)
            or tuple(descriptor.element_shape)
        ):
            raise TaichiRuntimeError(
                f"{role} must have scalar dtype {dtype} and extent {size}"
            )
        argument = self._runtime_storage_argument(description)
        if argument is None:
            return None
        execution_mode = dict(argument.qualification)["execution_mode"]
        if (
            execution_mode == "kDirectAffine"
            and not self.capabilities.dense_storage_affine_operands
        ):
            return None
        cached = (
            description,
            argument,
            storage_kind,
            execution_mode,
            alias_owner_key,
        )
        self._direct_storage_operand_cache[cache_key] = cached
        self._vector_io.record_direct_storage_operand(reused=False)
        return _DirectStorageOperand(public, *cached)

    def apply(self, input, out=None, *, alpha=1.0, beta=0.0, addend=None):
        """Synchronously computes ``alpha * self(input) + beta * addend``.

        ``input`` and ``out`` may not alias. ``addend`` may be the exact same
        vector/view as ``out`` so callers can express in-place accumulation;
        nonexact field-view overlap is rejected. When ``beta`` is zero,
        ``addend`` is neither validated nor read. Generalized coefficients are
        lowered without host readback on CPU, CUDA, and Vulkan. GPU lowering
        currently requires f32 ndarray operands; an exact ``addend is out``
        alias uses one persistent operator scratch so the old addend remains
        available until the provider result has been produced.
        """
        self._ensure_valid()
        try:
            alpha = float(alpha)
            beta = float(beta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError(
                "LinearOperator apply coefficients must be finite"
            ) from exc
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise TaichiRuntimeError(
                "LinearOperator apply coefficients must be finite"
            )
        if (
            out is not None
            and alpha == 1.0
            and beta == 0.0
            and self.capabilities.dense_storage_operands
        ):
            direct_input = self._direct_storage_operand(
                input,
                "LinearOperator input",
                self.shape[1],
                self.dtype,
            )
            direct_output = self._direct_storage_operand(
                out,
                "LinearOperator output",
                self.shape[0],
                self.dtype,
            )
            if direct_input is not None and direct_output is not None:
                alias = analyze_storage_alias(
                    direct_input.description, direct_output.description
                )
                if (
                    alias != "kProvenDisjoint"
                    and direct_input.alias_owner_key
                    == direct_output.alias_owner_key
                ):
                    raise TaichiRuntimeError(
                        "LinearOperator.apply does not permit input/output "
                        f"aliasing ({alias})"
                    )
                self._handle._apply_dense_storage(
                    self._program,
                    direct_input.runtime_argument,
                    direct_output.runtime_argument,
                )
                self._vector_io.record_direct_dense_storage(
                    direct_input.storage_kind,
                    direct_output.storage_kind,
                    direct_input.execution_mode,
                    direct_output.execution_mode,
                )
                return out
        if isinstance(input, DenseNdarrayView) or isinstance(
            out, DenseNdarrayView
        ):
            raise TaichiRuntimeError(
                "DenseNdarrayView operands require an explicit out, the "
                "overwrite form alpha=1 and beta=0, and a provider-qualified "
                "direct runtime-storage binding"
            )

        input_operand = _prepare_vector_input(
            input,
            "LinearOperator input",
            self.shape[1],
            self.dtype,
            self._vector_io,
            "apply_input",
        )
        if out is None:
            out = ScalarNdarray(self.dtype, (self.shape[0],))
        output_operand = _prepare_vector_output(
            out,
            "LinearOperator output",
            self.shape[0],
            self.dtype,
            self._vector_io,
            "apply_output",
        )
        if _vector_operands_overlap(input_operand, output_operand):
            raise TaichiRuntimeError(
                "LinearOperator.apply does not permit input/output aliasing"
            )
        addend_operand = None
        if beta != 0.0:
            if addend is None:
                raise TaichiRuntimeError(
                    "LinearOperator.apply with nonzero beta requires addend"
                )
            addend_view = self._vector_io.view(addend, "LinearOperator addend")
            shares_output = (
                addend_view is not None
                and output_operand.view is not None
                and addend_view._exact_view_key
                == output_operand.view._exact_view_key
            )
            addend_operand = _prepare_vector_input(
                (addend_view if addend_view is not None else addend),
                "LinearOperator addend",
                self.shape[0],
                self.dtype,
                self._vector_io,
                "apply_output" if shares_output else "apply_addend",
            )
            if _vector_operands_overlap(
                addend_operand, output_operand
            ) and not _vector_operands_exact(addend_operand, output_operand):
                raise TaichiRuntimeError(
                    "LinearOperator addend and output overlap without being "
                    "the same vector view"
                )
        coalesced_session = None
        if alpha == 1.0 and beta == 0.0:
            if output_operand.view is not None:
                coalesced_session = self._handle._begin_session()
                coalesced_session._submit(
                    self._program,
                    input_operand.array.arr,
                    output_operand.array.arr,
                )
            else:
                self._handle._apply(
                    self._program,
                    input_operand.array.arr,
                    output_operand.array.arr,
                )
        else:
            self._handle._apply_generalized(
                self._program,
                input_operand.array.arr,
                (None if addend_operand is None else addend_operand.array.arr),
                output_operand.array.arr,
                alpha,
                beta,
            )
        _finish_vector_output(
            self._vector_io,
            output_operand,
            self.dtype,
            self.shape[0],
            coalesced_session,
        )
        return out

    def __matmul__(self, input):
        return self.apply(input)

    def scaled(self, scale):
        """Returns ``scale * self`` with qualified f32 GPU lowering."""
        self._ensure_valid()
        try:
            scale = float(scale)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError("operator scale must be finite") from exc
        if not math.isfinite(scale):
            raise TaichiRuntimeError("operator scale must be finite")
        handle = _ti_core._make_scaled_operator(scale, self._handle)
        return self._from_handle(
            handle,
            provider_kind="composition",
            retained=(self,),
            composition_spec=("scale", scale, self),
        )

    def shifted(self, shift):
        """Returns ``self + shift * I`` with fused recordable lowering.

        The operator must be square. On GPU Graph paths the identity term is
        accumulated directly from the input after the provider action, so the
        shift does not create a second operator dispatch or temporary vector.
        """
        self._ensure_valid()
        if self.shape[0] != self.shape[1]:
            raise TaichiRuntimeError("operator shifts require a square operator")
        try:
            shift = float(shift)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError("operator shift must be finite") from exc
        if not math.isfinite(shift):
            raise TaichiRuntimeError("operator shift must be finite")
        if shift == 0.0:
            return self
        retained = (self,)
        arch = self._program.config().arch
        if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            identity_handle = _ti_core._make_identity_operator(
                self._program, self.dtype, self.shape[0]
            )
        else:
            if self.dtype != f32:
                raise TaichiRuntimeError(
                    "GPU shifted operators currently require f32"
                )
            from taichi_forge.linalg._composition_kernels import identity_f32

            topology = ScalarNdarray(i32, (1,))
            topology.fill(0)
            identity_operator = LinearOperator.from_kernel(
                identity_f32,
                self.shape[0],
                topology,
                traits=OperatorTraits.spd(),
            )
            identity_handle = identity_operator._handle
            retained = (self, identity_operator)
        shifted_handle = _ti_core._make_sum_operator(
            self._handle,
            _ti_core._make_scaled_operator(shift, identity_handle),
        )
        return self._from_handle(
            shifted_handle,
            provider_kind="composition",
            retained=retained,
            composition_spec=("shift", shift, self),
        )

    def parameterized_affine(
        self,
        other=None,
        *,
        alpha=1.0,
        beta=0.0,
        alpha_range,
        beta_range,
    ):
        """Returns an updateable ``alpha * self + beta * other`` operator.

        Omitting ``other`` uses the identity and represents an updateable
        shift. Both closed coefficient ranges are mandatory: mathematical
        traits are derived for the complete declared ranges, and later
        updates outside them fail instead of silently invalidating an SPD
        qualification. One update publishes alpha and beta atomically.
        """
        self._ensure_valid()
        alpha = _canonical_parameter_scalar(alpha, self.dtype, "parameterized affine alpha")
        beta = _canonical_parameter_scalar(beta, self.dtype, "parameterized affine beta")
        alpha_range = _normalize_parameter_range(alpha_range, alpha, "alpha_range", self.dtype)
        beta_range = _normalize_parameter_range(beta_range, beta, "beta_range", self.dtype)
        identity_shift = other is None
        if identity_shift:
            if self.shape[0] != self.shape[1]:
                raise TaichiRuntimeError("parameterized identity shifts require a square operator")
            arch = self._program.config().arch
            if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
                other = identity(self.shape[0], self.dtype)
            else:
                if self.dtype != f32:
                    raise TaichiRuntimeError("GPU parameterized affine operators require f32")
                from taichi_forge.linalg._composition_kernels import identity_f32

                topology = ScalarNdarray(i32, (1,))
                topology.fill(0)
                other = LinearOperator.from_kernel(
                    identity_f32,
                    self.shape[0],
                    topology,
                    traits=OperatorTraits.spd(),
                )
        if not isinstance(other, LinearOperator):
            raise TypeError("other must be LinearOperator or None")
        other._ensure_valid()
        if self.shape != other.shape or self.dtype != other.dtype:
            raise TaichiRuntimeError("parameterized affine operands require identical shape and dtype")
        if self._program is not other._program:
            raise TaichiRuntimeError("parameterized affine operands belong to different runtimes")
        if self.dtype != f32 and self._program.config().arch not in (
            _ti_core.Arch.x64,
            _ti_core.Arch.arm64,
        ):
            raise TaichiRuntimeError("GPU parameterized affine operators require f32")
        handle = _ti_core._make_parameterized_affine_operator(
            self._handle,
            other._handle,
            alpha,
            beta,
            alpha_range[0],
            alpha_range[1],
            beta_range[0],
            beta_range[1],
        )
        state = _AffineParameterState(
            handle, alpha_range, beta_range, self.dtype
        )
        result = self._from_handle(
            handle,
            provider_kind="composition",
            retained=(self, other, state),
            composition_spec=(
                "parameterized_affine",
                state,
                self,
                other,
                identity_shift,
            ),
        )
        result._parameter_state = state
        return result

    def update_parameters(self, *, alpha, beta, expected_version):
        """Atomically publishes the next affine coefficient generation."""
        self._ensure_valid()
        state = self._parameter_state
        if state is None:
            raise TaichiRuntimeError("LinearOperator does not own updateable affine parameters")
        if isinstance(expected_version, bool):
            raise TaichiRuntimeError("expected_version must be a positive integer")
        try:
            expected_version = _operator.index(expected_version)
        except TypeError as exc:
            raise TaichiRuntimeError("expected_version must be a positive integer") from exc
        if expected_version <= 0:
            raise TaichiRuntimeError("expected_version must be a positive integer")
        return state.update(alpha, beta, expected_version)

    @property
    def parameters(self):
        """Returns the current affine coefficients, ranges, and version."""
        self._ensure_valid()
        if self._parameter_state is None:
            raise TaichiRuntimeError("LinearOperator does not own updateable affine parameters")
        return self._parameter_state.snapshot()

    def __mul__(self, scale):
        return self.scaled(scale)

    def __rmul__(self, scale):
        return self.scaled(scale)

    def __add__(self, other):
        if not isinstance(other, LinearOperator):
            return NotImplemented
        self._ensure_valid()
        other._ensure_valid()
        handle = _ti_core._make_sum_operator(self._handle, other._handle)
        return self._from_handle(
            handle,
            provider_kind="composition",
            retained=(self, other),
            composition_spec=("sum", self, other),
        )

    def compose(self, inner):
        """Returns ``self(inner(x))`` with qualified f32 GPU lowering."""
        if not isinstance(inner, LinearOperator):
            raise TypeError("inner must be LinearOperator")
        self._ensure_valid()
        inner._ensure_valid()
        handle = _ti_core._make_composed_operator(self._handle, inner._handle)
        return self._from_handle(
            handle,
            provider_kind="composition",
            retained=(self, inner),
            composition_spec=("compose", self, inner),
        )

    def adjoint(self):
        """Returns the explicit adjoint or fails if it is unavailable."""
        self._ensure_valid()
        handle = _ti_core._make_adjoint_operator(self._handle)
        return self._from_handle(
            handle,
            provider_kind="composition",
            retained=(self,),
            composition_spec=("adjoint", self),
        )

    def update_numeric(
        self,
        values,
        *,
        expected_topology_version=None,
        expected_numeric_version=None,
    ):
        """Publishes a provider-specific numeric generation."""
        self._ensure_valid()
        if self._provider_kind == "stored":
            if expected_topology_version is not None or (
                expected_numeric_version is not None
            ):
                raise TaichiRuntimeError(
                    "stored operators do not accept explicit version arguments"
                )
            self._source.update_values(values)
            return
        if (
            expected_topology_version is None
            or expected_numeric_version is None
        ):
            raise TaichiRuntimeError(
                "compiled operator updates require expected_topology_version "
                "and expected_numeric_version"
            )
        topology_version = _require_positive_size(
            expected_topology_version, "expected_topology_version"
        )
        numeric_version = _require_positive_size(
            expected_numeric_version, "expected_numeric_version"
        )
        if self._provider_kind == "kernel":
            values = _require_current_scalar_ndarray(values, "numeric")
            self._provider_core.update_numeric_data(
                self._program,
                values.arr,
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "graph":
            values = _normalized_resource_mapping(
                values, "numeric", require_nonempty=True
            )
            self._provider_core.update_numeric_data(
                self._program,
                {name: value.arr for name, value in values.items()},
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "kernel_action":
            values = _require_current_scalar_ndarray(values, "numeric")
            self._handle._update_numeric(
                self._program,
                {"numeric": values.arr},
                topology_version,
                numeric_version,
            )
            return
        if self._provider_kind == "graph_action":
            values = _normalized_resource_mapping(
                values, "numeric", require_nonempty=True
            )
            self._handle._update_numeric(
                self._program,
                {name: value.arr for name, value in values.items()},
                topology_version,
                numeric_version,
            )
            return
        raise TaichiRuntimeError(
            "numeric update is not defined for composed operators"
        )

    def prepare_apply(self, input, out, *, adjoint=False):
        """Pin a recordable action generation and fixed dense operands.

        Contents/live field state may change in place. Published numeric or
        coefficient generations are not followed: prepare again to adopt them.
        Unlike apply(), preparation performs no mathematics or host staging.
        See PreparedOperatorPlan for recording, ownership and scope.
        """
        from taichi_forge.linalg._prepared_operator import PreparedOperatorPlan

        return PreparedOperatorPlan(self, input, out, adjoint=adjoint)

    def graph_action(self, input, output, *, adjoint=False):
        """Return a recordable f32 Graph action for this operator apply.

        ``input`` and ``output`` are symbolic one-dimensional ndarray Graph
        arguments. The resulting provider action may be appended directly to
        a :class:`ti.graph.GraphBuilder` or embedded in a structured
        ``Sequential`` region. Provider topology/numeric snapshots remain
        owned by the operator; no second device allocation or copy is made.
        Compatible values-only updates are rebound when the Graph is launched,
        and each submission pins the immutable numeric generation that it
        actually uses. A topology, schema, state-tree, or runtime-generation
        change still invalidates the compiled Graph and fails closed instead
        of rebuilding implicitly.
        """
        self._ensure_valid()
        if not isinstance(adjoint, bool):
            raise TypeError("adjoint must be bool")
        if self.dtype != f32:
            raise TaichiRuntimeError(
                "LinearOperator Graph actions currently require f32"
            )
        if self._composition_spec is not None:
            return _LinearOperatorCompositionGraphNode(self, input, output, adjoint)
        if not self._handle._supports_recordable_kernel():
            raise TaichiRuntimeError(
                "LinearOperator provider does not expose a recordable "
                "kernel action"
            )
        if adjoint and not self.capabilities.adjoint_apply:
            raise TaichiRuntimeError(
                "LinearOperator provider does not expose an adjoint action"
            )
        return _LinearOperatorGraphNode(self, input, output, adjoint)

    def _supports_graph_action(self):
        if self.dtype != f32:
            return False
        spec = self._composition_spec
        if spec is None:
            return bool(self._handle._supports_recordable_kernel())
        if spec[0] in ("scale", "shift"):
            return spec[2]._supports_graph_action()
        if spec[0] == "parameterized_affine":
            return spec[2]._supports_graph_action() and (bool(spec[4]) or spec[3]._supports_graph_action())
        if spec[0] == "block_diagonal":
            return all(block._supports_graph_action() for block in spec[1])
        return all(operand._supports_graph_action() for operand in spec[1:] if isinstance(operand, LinearOperator))

    def statistics(self):
        """Returns native execution counters for this operator plan."""
        self._ensure_valid()
        result = dict(self._handle._debug_runtime_stats())
        result["vector_io"] = self._vector_io.statistics()
        return result

    def vector_io_capabilities(self):
        """Returns supported operand storage and conversion modes."""
        self._ensure_valid()
        return _readonly_copy(_vector_io_capabilities())


def _cached_graph_dense_vector_binding(cache, name, value, extent):
    extent = int(extent)
    cached = cache.get(name)
    if cached is not None and cached[0] is value and cached[1] == extent:
        return cached[2], cached[3]
    if isinstance(value, VectorView):
        value._ensure_valid(f"LinearOperator Graph argument {name!r}")
        description = value._direct_storage_description
        if description is None:
            description = describe_storage(value)
    else:
        description = _flatten_storage_to_scalar_vector(
            describe_storage(value)
        )
    descriptor = description.descriptor
    if (
        descriptor is None
        or descriptor.scalar_type != f32
        or tuple(descriptor.index_shape) != (extent,)
        or tuple(descriptor.element_shape)
    ):
        raise TaichiRuntimeError(
            f"LinearOperator Graph argument {name!r} must expose "
            f"exactly {extent} scalar f32 values without staging"
        )
    view = (
        None
        if isinstance(value, ScalarNdarray)
        else DenseNdarrayView(value, description)
    )
    cache[name] = (value, extent, description, view)
    return description, view


def _validate_graph_dense_vector_disjoint(descriptions, message):
    alias = analyze_storage_alias(*descriptions)
    if alias == "kProvenDisjoint":
        return
    descriptors = tuple(description.descriptor for description in descriptions)
    source_kinds = {str(descriptor.source_kind) for descriptor in descriptors}
    has_dense_field = any(
        kind.startswith("kDense") and kind.endswith("Field")
        for kind in source_kinds
    )
    if "kNdarray" in source_kinds and has_dense_field and all(
        kind == "kNdarray" or kind.startswith("kDense") and kind.endswith("Field")
        for kind in source_kinds
    ):
        return
    raise TaichiRuntimeError(f"{message} ({alias})")


class _LinearOperatorGraphExecutable(NativeGraphExecutable):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        from taichi_forge.graph._graph import Arg, ArgKind

        for value, role in ((input_arg, "input"), (output_arg, "output")):
            if (
                getattr(value, "tag", None) != ArgKind.NDARRAY
                or value.dtype() != f32
                or int(value.field_dim) != 1
                or tuple(value.element_shape) != ()
            ):
                raise TaichiRuntimeError(
                    f"LinearOperator Graph {role} must be a symbolic scalar "
                    "f32 1-D ndarray"
                )
        if input_arg.name == output_arg.name:
            raise TaichiRuntimeError(
                "LinearOperator Graph input and output must use distinct "
                "symbolic resources"
            )
        self._operator = operator
        self._input_name = input_arg.name
        self._output_name = output_arg.name
        self._adjoint = adjoint
        self._record = operator._handle._recordable_kernel(adjoint)
        self._expected_stamp = tuple(self._record._resource_stamp())
        self._record_signature = self._compatible_record_signature(self._record)
        prefix = f"__linear_operator_{id(self._record):x}"
        graph_dispatches = tuple(self._record._graph_dispatches)
        self._opaque_graph_ir = bool(graph_dispatches)
        self._fixed_binding_sources = {}
        fixed = {}
        if graph_dispatches:
            remapped = {
                "input": input_arg,
                "output": output_arg,
            }
            for name, value in dict(self._record._fixed_i32).items():
                private_name = f"{prefix}_{name}"
                fixed[private_name] = int(value)
                self._fixed_binding_sources[private_name] = ("i32", name)
                remapped[name] = Arg(
                    ArgKind.SCALAR, private_name, i32
                )
            for name, value in dict(
                self._record._fixed_ndarrays
            ).items():
                private_name = f"{prefix}_{name}"
                fixed[private_name] = ProviderOwnedNdarrayBinding(
                    value, self._record
                )
                self._fixed_binding_sources[private_name] = (
                    "ndarray",
                    name,
                )
                remapped[name] = Arg(
                    ArgKind.NDARRAY,
                    private_name,
                    value.element_data_type(),
                    ndim=len(tuple(value.shape)),
                )
            dispatches = []
            for kernel, original_symbols in graph_dispatches:
                symbols = []
                for original in original_symbols:
                    symbol = remapped.get(original.name)
                    if symbol is None:
                        raise TaichiRuntimeError(
                            "LinearOperator recordable Graph dispatch "
                            f"references undeclared argument {original.name!r}"
                        )
                    symbols.append(symbol)
                dispatches.append((kernel, tuple(symbols)))
        else:
            active_name = f"{prefix}_active_size"
            topology_name = f"{prefix}_topology"
            numeric_name = f"{prefix}_numeric"
            topology = self._record._topology
            fixed = {
                active_name: int(self._record.active_size),
                topology_name: ProviderOwnedNdarrayBinding(
                    topology, self._record
                ),
            }
            self._fixed_binding_sources[active_name] = (
                "active_size",
                None,
            )
            self._fixed_binding_sources[topology_name] = (
                "topology",
                None,
            )
            symbols = [
                Arg(ArgKind.SCALAR, active_name, i32),
                Arg(
                    ArgKind.NDARRAY,
                    topology_name,
                    topology.element_data_type(),
                    ndim=len(tuple(topology.shape)),
                )
            ]
            numeric = self._record._numeric
            if numeric is not None:
                fixed[numeric_name] = ProviderOwnedNdarrayBinding(
                    numeric, self._record
                )
                self._fixed_binding_sources[numeric_name] = (
                    "numeric",
                    None,
                )
                symbols.append(
                    Arg(
                        ArgKind.NDARRAY,
                        numeric_name,
                        numeric.element_data_type(),
                        ndim=len(tuple(numeric.shape)),
                    )
                )
            symbols.extend((input_arg, output_arg))
            dispatches = ((self._record._kernel, tuple(symbols)),)
        self._action = DispatchGraphAction(
            dispatches,
            backends=("cpu", "cuda", "vulkan"),
            conditional_body_safe=True,
            fixed_bindings=fixed,
            update_policy="rebind",
            synchronization_domain="runtime_ordered",
        )
        self._runtime_arg_schema = (
            RuntimeBinding(self._input_name, "dense_vector"),
            RuntimeBinding(self._output_name, "dense_vector"),
        )
        state_effects = tuple(
            ResourceEffect(
                f"__linear_operator_state_{tree_id}_{generation}",
                GraphAccess.READ_WRITE,
                runtime_bound=False,
            )
            for tree_id, generation, _ in self._record._state_dependencies
        )
        self._resource_effects = (
            ResourceEffect(self._input_name, GraphAccess.READ),
            ResourceEffect(self._output_name, GraphAccess.WRITE),
            *state_effects,
        )
        self._fallback_operator = operator.adjoint() if adjoint else operator
        self._graph_binding_views = {}

    def run(self, runtime_args):
        self._fallback_operator.apply(
            runtime_args[self._input_name],
            out=runtime_args[self._output_name],
        )

    @property
    def runtime_arg_schema(self):
        return self._runtime_arg_schema

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def graph_ir_node(self):
        if not self._opaque_graph_ir:
            return super().graph_ir_node
        # A generic provider Graph can mutate fixed workspace or fixed
        # resources between physical dispatches. Keep the definition-time
        # native call opaque until per-dispatch metadata has been carried
        # through this provider boundary; recording into the outer CGraph is
        # still enabled.
        return NativeCallNode(
            name="linear_operator_apply",
            effects=self._resource_effects,
            bindings=self._runtime_arg_schema,
            opaque=True,
        )

    @property
    def debug_info(self):
        return {
            "kind": "linear_operator_apply",
            "provider": self._operator.provider,
            "adjoint": self._adjoint,
            "update_policy": "rebind",
            "graph_ir_opaque": self._opaque_graph_ir,
            "recordable": dict(self._record._recordable_stats),
        }

    @staticmethod
    def _ndarray_schema(value):
        return (
            str(value.element_data_type()),
            tuple(value.shape),
        )

    @classmethod
    def _compatible_record_signature(cls, record):
        graph_dispatches = tuple(record._graph_dispatches)
        if graph_dispatches:
            dispatch_schema = tuple(
                tuple(
                    (
                        symbol.name,
                        str(getattr(symbol, "tag", "")),
                        str(symbol.dtype()),
                        int(symbol.field_dim),
                        tuple(symbol.element_shape),
                    )
                    for symbol in symbols
                )
                for _, symbols in graph_dispatches
            )
            return (
                "graph",
                tuple(sorted(dict(record._fixed_i32).items())),
                tuple(
                    sorted(
                        (name, cls._ndarray_schema(value))
                        for name, value in dict(
                            record._fixed_ndarrays
                        ).items()
                    )
                ),
                tuple(record._state_dependencies),
                dispatch_schema,
            )
        topology = record._topology
        numeric = record._numeric
        return (
            "kernel",
            int(record.active_size),
            cls._ndarray_schema(topology),
            None if numeric is None else cls._ndarray_schema(numeric),
            tuple(record._state_dependencies),
        )

    def _current_compatible_record(self):
        self._operator._ensure_valid()
        record = self._operator._handle._recordable_kernel(self._adjoint)
        current = tuple(record._resource_stamp())
        if current[:4] != self._expected_stamp[:4]:
            raise TaichiRuntimeError(
                "LinearOperator provider topology changed; rebuild the Graph"
            )
        if self._compatible_record_signature(record) != self._record_signature:
            raise TaichiRuntimeError(
                "LinearOperator recordable action schema changed; rebuild the Graph"
            )
        return record

    def _bind_provider_generation(self):
        record = self._current_compatible_record()
        replacements = {}
        fixed_i32 = dict(record._fixed_i32)
        fixed_ndarrays = dict(record._fixed_ndarrays)
        for private_name, (kind, source_name) in (
            self._fixed_binding_sources.items()
        ):
            if kind == "active_size":
                value = int(record.active_size)
            elif kind == "topology":
                value = ProviderOwnedNdarrayBinding(
                    record._topology, record
                )
            elif kind == "numeric":
                value = ProviderOwnedNdarrayBinding(record._numeric, record)
            elif kind == "i32":
                value = int(fixed_i32[source_name])
            else:
                value = ProviderOwnedNdarrayBinding(
                    fixed_ndarrays[source_name], record
                )
            replacements[private_name] = value
        return PreparedGraphBindings(replacements, (record,))

    def validate_graph_lifetime(self):
        self._current_compatible_record()

    def bind_graph_arguments(self, runtime_args):
        prepared = self._bind_provider_generation()
        replacements = dict(prepared.replacements)
        expected = (
            (self._input_name, self._operator.shape[0 if self._adjoint else 1]),
            (self._output_name, self._operator.shape[1 if self._adjoint else 0]),
        )
        for name, extent in expected:
            _, view = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                extent,
            )
            if view is not None:
                replacements[name] = view
        return PreparedGraphBindings(
            replacements, prepared.submission_owners
        )

    def validate_graph_bindings(self, runtime_args):
        expected = (
            (self._input_name, self._operator.shape[0 if self._adjoint else 1]),
            (self._output_name, self._operator.shape[1 if self._adjoint else 0]),
        )
        descriptions = []
        for name, extent in expected:
            description, _ = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                extent,
            )
            descriptions.append(description)
        _validate_graph_dense_vector_disjoint(
            descriptions,
            "LinearOperator Graph input and output must be proven disjoint",
        )


class _LinearOperatorGraphNode(NativeGraphNode):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        self._operator = operator
        self._input_arg = input_arg
        self._output_arg = output_arg
        self._adjoint = adjoint

    def compile(self):
        return _LinearOperatorGraphExecutable(
            self._operator,
            self._input_arg,
            self._output_arg,
            self._adjoint,
        )


class _LinearOperatorCompositionGraphAction(DispatchGraphAction):
    def __init__(
        self,
        dispatches,
        *,
        fixed_bindings,
        temporary_bindings,
        temporary_requirements,
        allow_unused_public_bindings=False,
    ):
        super().__init__(
            dispatches,
            backends=("cpu", "cuda", "vulkan"),
            conditional_body_safe=True,
            fixed_bindings=fixed_bindings,
            allow_unused_public_bindings=allow_unused_public_bindings,
            update_policy="rebind",
            synchronization_domain="runtime_ordered",
        )
        self._composition_temporary_bindings = MappingProxyType(
            dict(temporary_bindings)
        )
        self._composition_temporary_requirements = {
            requirement.name: requirement for requirement in temporary_requirements
        }

    @property
    def temporary_bindings(self):
        return self._composition_temporary_bindings

    def bind_graph_temporaries(self, temporaries):
        result = {}
        for symbol, requirement_name in self.temporary_bindings.items():
            requirement = self._composition_temporary_requirements[requirement_name]
            binding = temporaries[requirement_name]
            if (
                binding.offset != 0
                or binding.bytes != requirement.bytes
                or binding.alignment < requirement.alignment
            ):
                return None
            result[symbol] = binding.storage
        return result


def _merge_linear_operator_graph_fixed_bindings(executables):
    fixed = {}
    for executable in executables:
        for name, value in executable.recordable_action.fixed_bindings.items():
            previous = fixed.get(name)
            if previous is None:
                fixed[name] = value
                continue
            previous_array = getattr(previous, "arr", None)
            value_array = getattr(value, "arr", None)
            if previous != value and previous_array is not value_array:
                raise TaichiRuntimeError(
                    "LinearOperator composition fixed Graph binding collision"
                )
    return fixed


class _LinearOperatorCompositionGraphExecutable(NativeGraphExecutable):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        from taichi_forge.graph._graph import Arg, ArgKind, gen_cpp_kernel
        from taichi_forge.linalg._composition_kernels import (
            axpby_f32,
            parameter_axpby_f32,
            scale_f32,
        )

        for value, role in ((input_arg, "input"), (output_arg, "output")):
            if (
                getattr(value, "tag", None) != ArgKind.NDARRAY
                or value.dtype() != f32
                or int(value.field_dim) != 1
                or tuple(value.element_shape) != ()
            ):
                raise TaichiRuntimeError(
                    f"LinearOperator Graph {role} must be a symbolic scalar "
                    "f32 1-D ndarray"
                )
        if input_arg.name == output_arg.name:
            raise TaichiRuntimeError(
                "LinearOperator Graph input and output must use distinct "
                "symbolic resources"
            )

        self._operator = operator
        self._input_name = input_arg.name
        self._output_name = output_arg.name
        self._adjoint = adjoint
        self._graph_binding_views = {}
        self._expected_stamp = tuple(operator._handle._resource_stamp())
        prefix = f"__linear_operator_composition_{id(self):x}"
        spec = operator._composition_spec
        kind = spec[0]
        children = []
        dispatches = []
        requirements = []
        temporary_bindings = {}
        private_fixed = {}
        composition_chain_length = 0
        reuses_composition_temporary = False
        reads_output_before_final_write = False
        self._parameter_state = None
        self._parameter_fixed_names = None
        self._block_bindings = ()
        self._block_graph_binding_views = {}
        derived_runtime_bindings = []

        def append_child(child_operator, child_input, child_output, child_adjoint):
            child = child_operator.graph_action(
                child_input, child_output, adjoint=child_adjoint
            ).compile()
            action = child.recordable_action
            if action is None:
                raise TaichiRuntimeError(
                    "LinearOperator composition child is not recordable"
                )
            children.append(child)
            dispatches.extend(action.dispatches)
            requirements.extend(child.temporary_requirements)
            temporary_bindings.update(action.temporary_bindings)
            derived_runtime_bindings.extend(
                child.derived_runtime_arg_schema
            )

        def temporary_arg(extent, suffix):
            symbol_name = f"{prefix}_{suffix}"
            requirement_name = f"{symbol_name}_storage"
            requirement = TemporaryRequirement(
                requirement_name, int(extent) * 4, 16, "f32"
            )
            requirements.append(requirement)
            temporary_bindings[symbol_name] = requirement_name
            return Arg(ArgKind.NDARRAY, symbol_name, f32, ndim=1)

        def weighted_terms(candidate, coefficient, term_adjoint, result):
            candidate_spec = candidate._composition_spec
            if candidate_spec is None:
                result.append((float(coefficient), candidate, term_adjoint))
                return
            candidate_kind = candidate_spec[0]
            if candidate_kind == "scale":
                weighted_terms(
                    candidate_spec[2],
                    coefficient * float(candidate_spec[1]),
                    term_adjoint,
                    result,
                )
                return
            if candidate_kind == "sum":
                weighted_terms(
                    candidate_spec[1], coefficient, term_adjoint, result
                )
                weighted_terms(
                    candidate_spec[2], coefficient, term_adjoint, result
                )
                return
            if candidate_kind == "adjoint":
                weighted_terms(
                    candidate_spec[1],
                    coefficient,
                    not term_adjoint,
                    result,
                )
                return
            result.append((float(coefficient), candidate, term_adjoint))

        def composition_leaves(candidate, term_adjoint, result):
            candidate_spec = candidate._composition_spec
            if candidate_spec is None:
                result.append((candidate, term_adjoint))
                return
            candidate_kind = candidate_spec[0]
            if candidate_kind == "adjoint":
                composition_leaves(
                    candidate_spec[1], not term_adjoint, result
                )
                return
            if candidate_kind == "compose":
                outer, inner = candidate_spec[1], candidate_spec[2]
                if term_adjoint:
                    composition_leaves(outer, True, result)
                    composition_leaves(inner, True, result)
                else:
                    composition_leaves(inner, False, result)
                    composition_leaves(outer, False, result)
                return
            result.append((candidate, term_adjoint))

        def append_weighted_terms(terms, extent):
            first_scale, first_operator, first_adjoint = terms[0]
            append_child(
                first_operator, input_arg, output_arg, first_adjoint
            )
            if len(terms) == 1:
                if first_scale == 1.0:
                    return
                scale_name = f"{prefix}_scale"
                size_name = f"{prefix}_size"
                scale_arg = Arg(ArgKind.SCALAR, scale_name, f32)
                size_arg = Arg(ArgKind.SCALAR, size_name, i32)
                private_fixed.update(
                    {scale_name: first_scale, size_name: int(extent)}
                )
                dispatches.append(
                    (
                        gen_cpp_kernel(
                            scale_f32, (output_arg, scale_arg, size_arg)
                        ),
                        (output_arg, scale_arg, size_arg),
                    )
                )
                return

            scratch = temporary_arg(extent, "weighted_sum_scratch")
            size_name = f"{prefix}_size"
            size_arg = Arg(ArgKind.SCALAR, size_name, i32)
            private_fixed[size_name] = int(extent)
            for index, (term_scale, term_operator, term_adjoint) in enumerate(
                terms[1:], start=1
            ):
                append_child(
                    term_operator, input_arg, scratch, term_adjoint
                )
                output_scale_name = f"{prefix}_output_scale_{index}"
                addend_scale_name = f"{prefix}_addend_scale_{index}"
                output_scale_arg = Arg(
                    ArgKind.SCALAR, output_scale_name, f32
                )
                addend_scale_arg = Arg(
                    ArgKind.SCALAR, addend_scale_name, f32
                )
                private_fixed.update(
                    {
                        output_scale_name: (
                            first_scale if index == 1 else 1.0
                        ),
                        addend_scale_name: term_scale,
                    }
                )
                dispatches.append(
                    (
                        gen_cpp_kernel(
                            axpby_f32,
                            (
                                scratch,
                                output_arg,
                                output_scale_arg,
                                addend_scale_arg,
                                size_arg,
                            ),
                        ),
                        (
                            scratch,
                            output_arg,
                            output_scale_arg,
                            addend_scale_arg,
                            size_arg,
                        ),
                    )
                )

        if kind == "block_diagonal":
            blocks = tuple(spec[1])
            block_bindings = []
            input_offset = 0
            output_offset = 0
            input_parent_extent = operator.shape[0 if adjoint else 1]
            output_parent_extent = operator.shape[1 if adjoint else 0]
            for index, block in enumerate(blocks):
                input_extent = block.shape[0 if adjoint else 1]
                output_extent = block.shape[1 if adjoint else 0]
                block_input_name = f"{prefix}_block_{index}_input"
                block_output_name = f"{prefix}_block_{index}_output"
                block_input_arg = Arg(
                    ArgKind.NDARRAY, block_input_name, f32, ndim=1
                )
                block_output_arg = Arg(
                    ArgKind.NDARRAY, block_output_name, f32, ndim=1
                )
                append_child(
                    block,
                    block_input_arg,
                    block_output_arg,
                    adjoint,
                )
                derived_runtime_bindings.extend(
                    (
                        RuntimeBinding(block_input_name, "dense_vector"),
                        RuntimeBinding(block_output_name, "dense_vector"),
                    )
                )
                block_bindings.extend(
                    (
                        (
                            block_input_name,
                            self._input_name,
                            input_parent_extent,
                            input_offset,
                            input_extent,
                        ),
                        (
                            block_output_name,
                            self._output_name,
                            output_parent_extent,
                            output_offset,
                            output_extent,
                        ),
                    )
                )
                input_offset += input_extent
                output_offset += output_extent
            if (
                input_offset != input_parent_extent
                or output_offset != output_parent_extent
            ):
                raise TaichiRuntimeError(
                    "fixed-layout block operator extents do not cover the "
                    "parent spaces"
                )
            self._block_bindings = tuple(block_bindings)
        elif kind == "parameterized_affine":
            state, left, right, identity_shift = (
                spec[1],
                spec[2],
                spec[3],
                bool(spec[4]),
            )
            extent = operator.shape[1 if adjoint else 0]
            append_child(left, input_arg, output_arg, adjoint)
            if identity_shift:
                addend_arg = input_arg
            else:
                addend_arg = temporary_arg(extent, "parameterized_affine_scratch")
                append_child(right, input_arg, addend_arg, adjoint)
            _, parameter_alpha, parameter_beta = state.recordable_snapshot()
            alpha_name = f"{prefix}_alpha"
            beta_name = f"{prefix}_beta"
            alpha_arg = Arg(ArgKind.SCALAR, alpha_name, f32)
            beta_arg = Arg(ArgKind.SCALAR, beta_name, f32)
            size_name = f"{prefix}_size"
            size_arg = Arg(ArgKind.SCALAR, size_name, i32)
            private_fixed[alpha_name] = float(parameter_alpha)
            private_fixed[beta_name] = float(parameter_beta)
            private_fixed[size_name] = int(extent)
            dispatches.append(
                (
                    gen_cpp_kernel(
                        parameter_axpby_f32,
                        (
                            addend_arg,
                            output_arg,
                            alpha_arg,
                            beta_arg,
                            size_arg,
                        ),
                    ),
                    (
                        addend_arg,
                        output_arg,
                        alpha_arg,
                        beta_arg,
                        size_arg,
                    ),
                )
            )
            self._parameter_state = state
            self._parameter_fixed_names = (alpha_name, beta_name)
        elif kind in ("adjoint", "scale", "sum"):
            extent = operator.shape[1 if adjoint else 0]
            terms = []
            weighted_terms(operator, 1.0, adjoint, terms)
            append_weighted_terms(terms, extent)
        elif kind == "shift":
            shift, base = float(spec[1]), spec[2]
            extent = operator.shape[1 if adjoint else 0]
            append_child(base, input_arg, output_arg, adjoint)
            size_name = f"{prefix}_size"
            output_scale_name = f"{prefix}_output_scale"
            shift_name = f"{prefix}_shift"
            size_arg = Arg(ArgKind.SCALAR, size_name, i32)
            output_scale_arg = Arg(
                ArgKind.SCALAR, output_scale_name, f32
            )
            shift_arg = Arg(ArgKind.SCALAR, shift_name, f32)
            private_fixed.update(
                {
                    size_name: int(extent),
                    output_scale_name: 1.0,
                    shift_name: shift,
                }
            )
            dispatches.append(
                (
                    gen_cpp_kernel(
                        axpby_f32,
                        (
                            input_arg,
                            output_arg,
                            output_scale_arg,
                            shift_arg,
                            size_arg,
                        ),
                    ),
                    (
                        input_arg,
                        output_arg,
                        output_scale_arg,
                        shift_arg,
                        size_arg,
                    ),
                )
            )
        elif kind == "compose":
            outer, inner = spec[1], spec[2]
            leaves = []
            composition_leaves(operator, adjoint, leaves)
            output_extent = operator.shape[1 if adjoint else 0]
            leaf_output_extents = [
                leaf.shape[1 if leaf_adjoint else 0]
                for leaf, leaf_adjoint in leaves
            ]
            if len(leaves) > 2 and all(
                extent == output_extent for extent in leaf_output_extents
            ):
                scratch = temporary_arg(output_extent, "compose_scratch")
                source = input_arg
                target = output_arg if len(leaves) % 2 else scratch
                for leaf, leaf_adjoint in leaves:
                    append_child(leaf, source, target, leaf_adjoint)
                    source = target
                    target = scratch if target is output_arg else output_arg
                composition_chain_length = len(leaves)
                reuses_composition_temporary = True
                reads_output_before_final_write = True
            else:
                intermediate_extent = inner.shape[0]
                scratch = temporary_arg(
                    intermediate_extent, "compose_scratch"
                )
                if adjoint:
                    append_child(outer, input_arg, scratch, True)
                    append_child(inner, scratch, output_arg, True)
                else:
                    append_child(inner, input_arg, scratch, False)
                    append_child(outer, scratch, output_arg, False)
                composition_chain_length = 2
        else:
            raise TaichiRuntimeError(
                f"Unsupported LinearOperator composition kind {kind!r}"
            )

        fixed = _merge_linear_operator_graph_fixed_bindings(children)
        overlap = fixed.keys() & private_fixed.keys()
        if overlap:
            raise TaichiRuntimeError(
                "LinearOperator composition private Graph binding collision"
            )
        fixed.update(private_fixed)
        requirement_by_name = {}
        for requirement in requirements:
            previous = requirement_by_name.get(requirement.name)
            if previous is not None and previous != requirement:
                raise TaichiRuntimeError(
                    "LinearOperator composition temporary requirement collision"
                )
            requirement_by_name[requirement.name] = requirement
        self._children = tuple(children)
        self._temporary_requirements = tuple(requirement_by_name.values())
        self._action = _LinearOperatorCompositionGraphAction(
            dispatches,
            fixed_bindings=fixed,
            temporary_bindings=temporary_bindings,
            temporary_requirements=self._temporary_requirements,
            allow_unused_public_bindings=bool(derived_runtime_bindings),
        )
        self._runtime_arg_schema = (
            RuntimeBinding(self._input_name, "dense_vector"),
            RuntimeBinding(self._output_name, "dense_vector"),
        )
        self._derived_runtime_arg_schema = tuple(
            derived_runtime_bindings
        )
        state_effects = []
        seen_effects = set()
        for child in children:
            for effect in child.resource_effects:
                if effect.runtime_bound:
                    continue
                key = (
                    effect.resource,
                    effect.access,
                    effect.runtime_bound,
                    effect.subresource,
                )
                if key not in seen_effects:
                    seen_effects.add(key)
                    state_effects.append(effect)
        self._resource_effects = (
            ResourceEffect(self._input_name, GraphAccess.READ),
            ResourceEffect(
                self._output_name,
                (
                    GraphAccess.READ_WRITE
                    if reads_output_before_final_write
                    else GraphAccess.WRITE
                ),
            ),
            *state_effects,
        )
        self._composition_chain_length = composition_chain_length
        self._reuses_composition_temporary = reuses_composition_temporary
        self._fallback_operator = operator.adjoint() if adjoint else operator

    def run(self, runtime_args):
        self._fallback_operator.apply(
            runtime_args[self._input_name],
            out=runtime_args[self._output_name],
        )

    @property
    def runtime_arg_schema(self):
        return self._runtime_arg_schema

    @property
    def derived_runtime_arg_schema(self):
        return self._derived_runtime_arg_schema

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def temporary_requirements(self):
        return self._temporary_requirements

    @property
    def lifetime_leases(self):
        return (self._operator,)

    @property
    def recordable_action(self):
        return self._action

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="linear_operator_composition",
            effects=self._resource_effects,
            bindings=self._runtime_arg_schema,
            temporaries=self._temporary_requirements,
            opaque=True,
        )

    @property
    def debug_info(self):
        return {
            "kind": "linear_operator_composition",
            "provider": self._operator.provider,
            "adjoint": self._adjoint,
            "dispatch_count": len(self._action.dispatches),
            "temporary_bytes": sum(item.bytes for item in self._temporary_requirements),
            "composition_chain_length": self._composition_chain_length,
            "reuses_composition_temporary": self._reuses_composition_temporary,
            "block_count": len(self._block_bindings) // 2,
            "zero_copy_block_subviews": bool(self._block_bindings),
        }

    def validate_graph_lifetime(self):
        self._operator._ensure_valid()
        current = tuple(self._operator._handle._resource_stamp())
        if current[:4] != self._expected_stamp[:4]:
            raise TaichiRuntimeError(
                "LinearOperator composition topology changed; rebuild the Graph"
            )
        for child in self._children:
            child.validate_graph_lifetime()
        if self._parameter_state is not None:
            self._parameter_state.recordable_generation()

    def bind_graph_arguments(self, runtime_args):
        prepared = self._bind_provider_generation()
        replacements = dict(prepared.replacements)
        expected = (
            (
                self._input_name,
                self._operator.shape[0 if self._adjoint else 1],
            ),
            (
                self._output_name,
                self._operator.shape[1 if self._adjoint else 0],
            ),
        )
        for name, extent in expected:
            _, view = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                extent,
            )
            if view is not None:
                replacements[name] = view
        replacements.update(self._bind_block_subviews(runtime_args))
        augmented_runtime_args = dict(runtime_args)
        augmented_runtime_args.update(replacements)
        submission_owners = list(prepared.submission_owners)
        owner_ids = {id(owner) for owner in submission_owners}
        for child in self._children:
            derived_names = {
                binding.name
                for binding in child.derived_runtime_arg_schema
            }
            if not derived_names:
                continue
            child_prepared = child.bind_graph_arguments(
                augmented_runtime_args
            )
            for name in derived_names:
                if name not in child_prepared.replacements:
                    raise TaichiRuntimeError(
                        "LinearOperator composition child did not bind "
                        f"derived Graph argument {name!r}"
                    )
                value = child_prepared.replacements[name]
                replacements[name] = value
                augmented_runtime_args[name] = value
            for owner in child_prepared.submission_owners:
                identity = id(owner)
                if identity not in owner_ids:
                    owner_ids.add(identity)
                    submission_owners.append(owner)
        return PreparedGraphBindings(
            replacements, tuple(submission_owners)
        )

    def _bind_provider_generation(self):
        replacements = {}
        submission_owners = []
        owner_ids = set()
        for child in self._children:
            prepared = child._bind_provider_generation()
            child_replacements = prepared.replacements
            overlap = replacements.keys() & child_replacements.keys()
            if overlap:
                raise TaichiRuntimeError(
                    "LinearOperator composition generation binding collision"
                )
            replacements.update(child_replacements)
            for owner in prepared.submission_owners:
                identity = id(owner)
                if identity not in owner_ids:
                    owner_ids.add(identity)
                    submission_owners.append(owner)
        if self._parameter_state is not None:
            generation, alpha, beta = self._parameter_state.recordable_snapshot()
            alpha_name, beta_name = self._parameter_fixed_names
            replacements[alpha_name] = float(alpha)
            replacements[beta_name] = float(beta)
            submission_owners.append(generation)
        return PreparedGraphBindings(replacements, tuple(submission_owners))

    def _bind_block_subviews(self, runtime_args):
        replacements = {}
        for name, parent_name, parent_extent, offset, extent in self._block_bindings:
            value = runtime_args[parent_name]
            cached = self._block_graph_binding_views.get(name)
            if (
                cached is not None
                and cached[0] is value
                and cached[1] == offset
                and cached[2] == extent
            ):
                replacements[name] = cached[3]
                continue
            description, _ = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                parent_name,
                value,
                parent_extent,
            )
            sliced = _slice_storage_description(
                description,
                (slice(offset, offset + extent),),
            )
            view = DenseNdarrayView(value, sliced)
            self._block_graph_binding_views[name] = (
                value,
                offset,
                extent,
                view,
            )
            replacements[name] = view
        return replacements

    def validate_graph_bindings(self, runtime_args):
        expected = (
            (
                self._input_name,
                self._operator.shape[0 if self._adjoint else 1],
            ),
            (
                self._output_name,
                self._operator.shape[1 if self._adjoint else 0],
            ),
        )
        descriptions = []
        for name, extent in expected:
            description, _ = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                extent,
            )
            descriptions.append(description)
        _validate_graph_dense_vector_disjoint(
            descriptions,
            "LinearOperator Graph input and output must be proven disjoint",
        )


class _LinearOperatorCompositionGraphNode(NativeGraphNode):
    def __init__(self, operator, input_arg, output_arg, adjoint):
        self._operator = operator
        self._input_arg = input_arg
        self._output_arg = output_arg
        self._adjoint = adjoint

    def compile(self):
        return _LinearOperatorCompositionGraphExecutable(
            self._operator,
            self._input_arg,
            self._output_arg,
            self._adjoint,
        )


def vector_io_capabilities():
    """Returns the backend-neutral dense vector I/O capability contract."""
    return _readonly_copy(_vector_io_capabilities())


def identity(size, dtype=f32):
    """Creates a CPU identity operator with framework-derived SPD traits."""
    size = _require_positive_size(size)
    if dtype not in (f32, f64):
        raise TaichiRuntimeError("identity dtype must be ti.f32 or ti.f64")
    handle = _ti_core._make_identity_operator(_current_program(), dtype, size)
    return LinearOperator._from_handle(
        handle, provider_kind="composition", retained=()
    )


def inverse_block_diagonal(inverse_blocks, block_size, *, assume_spd):
    """Builds a recordable scalar or small-block inverse action.

    ``inverse_blocks`` is a flat row-major f32 ndarray containing independent
    1x1, 2x2, 3x3, or 4x4 inverse blocks. No device-to-host validation is
    performed. Passing ``assume_spd=True`` is therefore an explicit assertion
    that every supplied block is symmetric positive definite and suitable as
    a fixed-linear CG/PCG preconditioner. Numeric generations can later be
    published with :meth:`LinearOperator.update_numeric` without rebuilding a
    compatible Graph.
    """
    inverse_blocks = _require_current_scalar_ndarray(
        inverse_blocks, "inverse_blocks", dtype=f32
    )
    block_size = _require_positive_size(block_size, "block_size")
    if block_size not in (1, 2, 3, 4):
        raise TaichiRuntimeError(
            "inverse_block_diagonal supports block_size 1, 2, 3, or 4"
        )
    if assume_spd is not True:
        raise TaichiRuntimeError(
            "inverse_block_diagonal requires assume_spd=True; Forge does "
            "not read back or infer the mathematical validity of device "
            "preconditioner coefficients"
        )
    coefficients = int(inverse_blocks.shape[0])
    block_coefficients = block_size * block_size
    if coefficients == 0 or coefficients % block_coefficients != 0:
        raise TaichiRuntimeError(
            "inverse_blocks length must be a positive multiple of "
            "block_size * block_size"
        )
    block_count = coefficients // block_coefficients
    size = block_count * block_size
    topology = ScalarNdarray(i32, (1,))
    topology.fill(block_size)
    from taichi_forge.linalg._preconditioner_kernels import (
        apply_inverse_blocks_1_f32,
        apply_inverse_blocks_2_f32,
        apply_inverse_blocks_3_f32,
        apply_inverse_blocks_4_f32,
    )

    kernel = {
        1: apply_inverse_blocks_1_f32,
        2: apply_inverse_blocks_2_f32,
        3: apply_inverse_blocks_3_f32,
        4: apply_inverse_blocks_4_f32,
    }[block_size]

    return LinearOperator.from_kernel(
        kernel,
        size,
        topology,
        numeric=inverse_blocks,
        traits=OperatorTraits.spd(),
    )


@dataclass(frozen=True)
class SmallBlockInverseResult:
    """Device-resident outputs of :class:`SmallBlockInverseBuilder`."""

    inverse_blocks: object
    status: object


class _SmallBlockInverseGraphExecutable(NativeGraphExecutable):
    # The executable snapshots every builder scalar and the BindingVersion
    # retains the exact three public storage objects. Their dtype, extent, and
    # disjointness are therefore immutable publication facts; no provider
    # generation or replacement hook participates in replay.
    graph_publish_time_binding_validation_stable = True

    def __init__(self, builder, blocks_arg, output_arg, status_arg):
        from taichi_forge.graph._graph import Arg, ArgKind, gen_cpp_kernel

        expected = (
            (blocks_arg, "blocks", f32),
            (output_arg, "inverse_blocks", f32),
            (status_arg, "status", i32),
        )
        for value, role, dtype in expected:
            if (
                getattr(value, "tag", None) != ArgKind.NDARRAY
                or value.dtype() != dtype
                or int(value.field_dim) != 1
                or tuple(value.element_shape) != ()
            ):
                raise TaichiRuntimeError(
                    f"SmallBlockInverseBuilder Graph {role} must be a " f"symbolic scalar {dtype} 1-D ndarray"
                )
        names = (blocks_arg.name, output_arg.name, status_arg.name)
        if len(set(names)) != len(names):
            raise TaichiRuntimeError("SmallBlockInverseBuilder Graph resources must use distinct " "symbolic names")
        prefix = f"__small_block_inverse_{id(self):x}"
        count_name = f"{prefix}_block_count"
        regularization_name = f"{prefix}_regularization"
        tolerance_name = f"{prefix}_pivot_tolerance"
        count_arg = Arg(ArgKind.SCALAR, count_name, i32)
        regularization_arg = Arg(ArgKind.SCALAR, regularization_name, f32)
        tolerance_arg = Arg(ArgKind.SCALAR, tolerance_name, f32)
        # A Graph action is an immutable executable snapshot. Keep every fact
        # consumed by its compiled dispatch together instead of consulting the
        # publicly inspectable builder again during replay. Otherwise a later
        # Python attribute assignment could make validation accept one extent
        # while the already compiled kernel still launches with another.
        self._kernel = builder._kernel
        self._program = builder._program
        self._block_size = int(builder.block_size)
        self._block_count = int(builder.block_count)
        self._coefficient_count = int(builder.coefficient_count)
        self._regularization = float(builder.regularization)
        self._pivot_tolerance = float(builder.pivot_tolerance)
        dispatch_args = (
            count_arg,
            regularization_arg,
            tolerance_arg,
            blocks_arg,
            output_arg,
            status_arg,
        )
        self._names = names
        self._action = DispatchGraphAction(
            ((gen_cpp_kernel(self._kernel, dispatch_args), dispatch_args),),
            backends=("cpu", "cuda", "vulkan"),
            conditional_body_safe=True,
            fixed_bindings={
                count_name: self._block_count,
                regularization_name: self._regularization,
                tolerance_name: self._pivot_tolerance,
            },
            update_policy="rebind",
            synchronization_domain="runtime_ordered",
        )
        self._runtime_arg_schema = tuple(RuntimeBinding(name, "dense_vector") for name in names)
        self._resource_effects = (
            ResourceEffect(names[0], GraphAccess.READ),
            ResourceEffect(names[1], GraphAccess.WRITE),
            ResourceEffect(names[2], GraphAccess.WRITE),
        )

    @property
    def runtime_arg_schema(self):
        return self._runtime_arg_schema

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="small_block_inverse_build",
            effects=self._resource_effects,
            bindings=self._runtime_arg_schema,
            opaque=False,
        )

    @property
    def debug_info(self):
        return {
            "kind": "small_block_inverse_build",
            "block_size": self._block_size,
            "block_count": self._block_count,
            "dispatch_count": 1,
            "status": "device_resident_per_block",
        }

    def run(self, runtime_args):
        if self._program is not _current_program():
            raise TaichiRuntimeError(
                "SmallBlockInverseBuilder Graph action belongs to an inactive "
                "or different runtime"
            )
        self._kernel(
            self._block_count,
            self._regularization,
            self._pivot_tolerance,
            runtime_args[self._names[0]],
            runtime_args[self._names[1]],
            runtime_args[self._names[2]],
        )

    def validate_graph_bindings(self, runtime_args):
        expected = (
            (self._names[0], f32, self._coefficient_count),
            (self._names[1], f32, self._coefficient_count),
            (self._names[2], i32, self._block_count),
        )
        descriptions = []
        for name, dtype, extent in expected:
            description = _flatten_storage_to_scalar_vector(describe_storage(runtime_args[name]))
            descriptor = description.descriptor
            if (
                descriptor is None
                or descriptor.scalar_type != dtype
                or tuple(descriptor.index_shape) != (extent,)
                or tuple(descriptor.element_shape)
            ):
                raise TaichiRuntimeError(
                    f"SmallBlockInverseBuilder Graph argument {name!r} "
                    f"must expose exactly {extent} scalar {dtype} values"
                )
            descriptions.append(description)
        if any(
            analyze_storage_alias(descriptions[left], descriptions[right]) != "kProvenDisjoint"
            for left in range(len(descriptions))
            for right in range(left + 1, len(descriptions))
        ):
            raise TaichiRuntimeError(
                "SmallBlockInverseBuilder Graph blocks, output, and status " "must be proven disjoint"
            )


class _SmallBlockInverseGraphNode(NativeGraphNode):
    def __init__(self, builder, blocks_arg, output_arg, status_arg):
        self._builder = builder
        self._blocks_arg = blocks_arg
        self._output_arg = output_arg
        self._status_arg = status_arg

    def compile(self):
        return _SmallBlockInverseGraphExecutable(
            self._builder,
            self._blocks_arg,
            self._output_arg,
            self._status_arg,
        )


class SmallBlockInverseBuilder:
    """Builds independent row-major f32 inverse blocks on the device.

    The builder has fixed block size/count and supports only sizes 1--4.
    Status remains device resident: 0 is success, 1 is non-finite input or
    result, and 2 is a singular/ill-conditioned pivot. Failed blocks are
    written as zero. This primitive does not infer or assert SPD.
    """

    def __init__(
        self,
        block_size,
        block_count,
        *,
        regularization=0.0,
        pivot_tolerance=1.0e-8,
    ):
        self.block_size = _require_positive_size(block_size, "block_size")
        self.block_count = _require_positive_size(block_count, "block_count")
        if self.block_size not in (1, 2, 3, 4):
            raise TaichiRuntimeError("SmallBlockInverseBuilder supports block_size 1, 2, 3, or 4")
        try:
            requested_regularization = float(regularization)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError("regularization must be finite") from exc
        self.regularization = _canonical_parameter_scalar(
            regularization, f32, "regularization"
        )
        self.pivot_tolerance = _canonical_parameter_scalar(
            pivot_tolerance, f32, "pivot_tolerance"
        )
        if self.regularization < 0.0:
            raise TaichiRuntimeError("regularization must be finite and non-negative")
        if requested_regularization != 0.0 and self.regularization == 0.0:
            raise TaichiRuntimeError(
                "nonzero regularization must be representable as f32"
            )
        if self.pivot_tolerance <= 0.0:
            raise TaichiRuntimeError("pivot_tolerance must be finite and positive")
        self.coefficient_count = self.block_count * self.block_size * self.block_size
        from taichi_forge.linalg._preconditioner_kernels import (
            build_inverse_blocks_1_f32,
            build_inverse_blocks_2_f32,
            build_inverse_blocks_3_f32,
            build_inverse_blocks_4_f32,
        )

        self._kernel = {
            1: build_inverse_blocks_1_f32,
            2: build_inverse_blocks_2_f32,
            3: build_inverse_blocks_3_f32,
            4: build_inverse_blocks_4_f32,
        }[self.block_size]
        self._program = _current_program()

    def _ensure_valid(self):
        if self._program is not _current_program():
            raise TaichiRuntimeError("SmallBlockInverseBuilder belongs to an inactive or " "different runtime")

    def build(self, blocks, *, out=None, status=None):
        """Enqueues one device build and returns device-resident outputs."""
        self._ensure_valid()
        blocks = _require_current_scalar_ndarray(blocks, "blocks", dtype=f32)
        if tuple(blocks.shape) != (self.coefficient_count,):
            raise TaichiRuntimeError("blocks length does not match block_size * block_size * " "block_count")
        if out is None:
            out = ScalarNdarray(f32, (self.coefficient_count,))
        out = _require_current_scalar_ndarray(out, "out", dtype=f32)
        if tuple(out.shape) != (self.coefficient_count,):
            raise TaichiRuntimeError("inverse block output has an incompatible length")
        if status is None:
            status = ScalarNdarray(i32, (self.block_count,))
        status = _require_current_scalar_ndarray(status, "status", dtype=i32)
        if tuple(status.shape) != (self.block_count,):
            raise TaichiRuntimeError("inverse block status has an incompatible length")
        descriptions = tuple(
            _flatten_storage_to_scalar_vector(describe_storage(value)) for value in (blocks, out, status)
        )
        if any(
            analyze_storage_alias(descriptions[left], descriptions[right]) != "kProvenDisjoint"
            for left in range(len(descriptions))
            for right in range(left + 1, len(descriptions))
        ):
            raise TaichiRuntimeError("blocks, inverse block output, and status must be disjoint")
        self._kernel(
            self.block_count,
            self.regularization,
            self.pivot_tolerance,
            blocks,
            out,
            status,
        )
        return SmallBlockInverseResult(out, status)

    def graph_action(self, blocks, output, status):
        """Returns a one-dispatch recordable Graph build action."""
        self._ensure_valid()
        return _SmallBlockInverseGraphNode(self, blocks, output, status)


def aslinearoperator(value, *, traits=None):
    """Returns ``value`` as a stable :class:`ti.linalg.LinearOperator`."""
    if isinstance(value, LinearOperator):
        if traits is not None:
            raise TaichiRuntimeError(
                "traits cannot be replaced on an existing LinearOperator"
            )
        return value
    return LinearOperator.from_sparse_matrix(value, traits=traits)


def block_diagonal(blocks: Sequence[LinearOperator]):
    """Creates a fixed-layout block-diagonal operator from one or more blocks."""
    blocks = tuple(blocks)
    if not blocks or any(
        not isinstance(block, LinearOperator) for block in blocks
    ):
        raise TaichiRuntimeError(
            "block_diagonal expects one or more LinearOperator blocks"
        )
    for block in blocks:
        block._ensure_valid()
    handle = _ti_core._make_block_diagonal_operator(
        [block._handle for block in blocks]
    )
    return LinearOperator._from_handle(
        handle,
        provider_kind="composition",
        retained=blocks,
        composition_spec=("block_diagonal", blocks),
    )


def _qualification_non_negative_integer(value, role):
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{role} must be a non-negative integer")
    try:
        value = _operator.index(value)
    except TypeError as exc:
        raise TaichiRuntimeError(
            f"{role} must be a non-negative integer"
        ) from exc
    if value < 0:
        raise TaichiRuntimeError(f"{role} must be a non-negative integer")
    return value


def _qualification_tolerance(value, default, role):
    if value is None:
        return default
    if isinstance(value, bool):
        raise TaichiRuntimeError(f"{role} must be finite and non-negative")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TaichiRuntimeError(
            f"{role} must be finite and non-negative"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise TaichiRuntimeError(f"{role} must be finite and non-negative")
    return value


def _qualification_ndarray(values, dtype):
    result = ScalarNdarray(dtype, values.shape)
    result.from_numpy(values)
    return result


def _qualification_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    difference = actual - expected
    max_absolute = float(np.max(np.abs(difference), initial=0.0))
    reference_norm = float(np.linalg.norm(expected.reshape(-1)))
    error_norm = float(np.linalg.norm(difference.reshape(-1)))
    relative_l2 = error_norm / max(reference_norm, np.finfo(actual.dtype).tiny)
    return max_absolute, relative_l2


def _qualification_check(name, passed, metrics, tolerance, details=""):
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "metrics": metrics,
        "tolerance": tolerance,
        "details": details,
    }


def qualify_operator(
    operator,
    *,
    reference=None,
    adjoint_reference=None,
    samples=3,
    seed=0,
    atol=None,
    rtol=None,
    warmup=1,
    repetitions=5,
    metadata=None,
):
    """Qualifies public LinearOperator contracts and returns evidence.

    References may be NumPy matrices or callables accepting and returning
    one-dimensional NumPy arrays. A matrix reference uses the public
    ``(range, domain)`` shape convention. The runner performs synchronous
    public ``apply`` calls and never changes the provider or execution policy.
    """
    if not isinstance(operator, LinearOperator):
        raise TypeError("operator must be ti.linalg.LinearOperator")
    operator._ensure_valid()
    samples = _qualification_non_negative_integer(samples, "samples")
    warmup = _qualification_non_negative_integer(warmup, "warmup")
    repetitions = _qualification_non_negative_integer(
        repetitions, "repetitions"
    )
    if samples == 0:
        raise TaichiRuntimeError("samples must be positive")
    if repetitions == 0:
        raise TaichiRuntimeError("repetitions must be positive")
    if isinstance(seed, bool):
        raise TaichiRuntimeError("seed must be an integer")
    try:
        seed = _operator.index(seed)
    except TypeError as exc:
        raise TaichiRuntimeError("seed must be an integer") from exc
    default_tolerance = 5e-5 if operator.dtype == f32 else 1e-11
    atol = _qualification_tolerance(atol, default_tolerance, "atol")
    rtol = _qualification_tolerance(rtol, default_tolerance, "rtol")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, Mapping) or any(
        not isinstance(name, str) for name in metadata
    ):
        raise TaichiRuntimeError("metadata must be a mapping with string keys")
    custom_metadata = copy.deepcopy(dict(metadata))
    try:
        json.dumps(custom_metadata)
    except (TypeError, ValueError) as exc:
        raise TaichiRuntimeError("metadata must be JSON-serializable") from exc

    rows, columns = operator.shape
    numpy_dtype = np.float32 if operator.dtype == f32 else np.float64

    def normalize_reference(candidate, expected_shape, role):
        if candidate is None:
            return None
        if callable(candidate):
            return candidate
        matrix = np.asarray(candidate, dtype=numpy_dtype)
        if matrix.shape != expected_shape:
            raise TaichiRuntimeError(
                f"{role} must have shape {expected_shape}, got {matrix.shape}"
            )
        return lambda values: matrix @ values

    forward_oracle = normalize_reference(
        reference, (rows, columns), "reference"
    )
    if (
        adjoint_reference is None
        and reference is not None
        and not callable(reference)
    ):
        reference_matrix = np.asarray(reference, dtype=numpy_dtype)
        adjoint_reference = reference_matrix.T
    adjoint_oracle = normalize_reference(
        adjoint_reference, (columns, rows), "adjoint_reference"
    )

    rng = np.random.default_rng(seed)
    checks = []
    initial_statistics = operator.statistics()
    timing_input_host = rng.standard_normal(columns).astype(numpy_dtype)
    timing_input = _qualification_ndarray(timing_input_host, operator.dtype)
    timing_output = ScalarNdarray(operator.dtype, (rows,))
    start_ns = time.perf_counter_ns()
    operator.apply(timing_input, out=timing_output)
    first_apply_ns = time.perf_counter_ns() - start_ns

    finite = bool(np.all(np.isfinite(timing_output.to_numpy())))
    checks.append(
        _qualification_check(
            "finite_forward",
            finite,
            {"nonfinite_values": 0 if finite else 1},
            {"nonfinite_values": 0},
        )
    )

    baseline_output = timing_output.to_numpy()
    generalized_alpha = numpy_dtype(-0.75)
    generalized_beta = numpy_dtype(0.5)
    poison = np.full(rows, np.nan, dtype=numpy_dtype)
    finite_addend = rng.standard_normal(rows).astype(numpy_dtype)
    try:
        no_read_output = operator.apply(
            timing_input,
            alpha=generalized_alpha,
            beta=0.0,
            addend=_qualification_ndarray(poison, operator.dtype),
        ).to_numpy()
        accumulated_output = operator.apply(
            timing_input,
            alpha=generalized_alpha,
            beta=generalized_beta,
            addend=_qualification_ndarray(finite_addend, operator.dtype),
        ).to_numpy()
        no_read_error = _qualification_error(
            no_read_output, generalized_alpha * baseline_output
        )
        accumulation_error = _qualification_error(
            accumulated_output,
            generalized_alpha * baseline_output
            + generalized_beta * finite_addend,
        )
        absolute = max(no_read_error[0], accumulation_error[0])
        relative = max(no_read_error[1], accumulation_error[1])
        checks.append(
            _qualification_check(
                "generalized_apply",
                absolute <= atol or relative <= rtol,
                {
                    "max_absolute_error": absolute,
                    "max_relative_error": relative,
                    "beta_zero_output_finite": bool(
                        np.all(np.isfinite(no_read_output))
                    ),
                },
                {"atol": atol, "rtol": rtol},
                "beta=0 poison-addend no-read and finite accumulation",
            )
        )
    except RuntimeError as exc:
        details = str(exc)
        if "Generalized operator lowering is unavailable" in details:
            checks.append(
                {
                    "name": "generalized_apply",
                    "status": "unsupported",
                    "metrics": {},
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "details": details,
                }
            )
        else:
            checks.append(
                {
                    "name": "generalized_apply",
                    "status": "failed",
                    "metrics": {},
                    "tolerance": {"atol": atol, "rtol": rtol},
                    "details": details,
                }
            )

    maxima = {
        "linearity": [0.0, 0.0],
        "forward_reference": [0.0, 0.0],
        "adjoint_dot_product": [0.0, 0.0],
        "adjoint_reference": [0.0, 0.0],
    }
    adjoint_operator = (
        operator.adjoint() if operator.capabilities.adjoint_apply else None
    )

    def accumulate(name, absolute, relative):
        maxima[name][0] = max(maxima[name][0], absolute)
        maxima[name][1] = max(maxima[name][1], relative)

    for _ in range(samples):
        left = rng.standard_normal(columns).astype(numpy_dtype)
        right = rng.standard_normal(columns).astype(numpy_dtype)
        alpha = numpy_dtype(rng.uniform(-1.25, 1.25))
        beta = numpy_dtype(rng.uniform(-1.25, 1.25))
        combined = alpha * left + beta * right
        applied_left = operator.apply(
            _qualification_ndarray(left, operator.dtype)
        ).to_numpy()
        applied_right = operator.apply(
            _qualification_ndarray(right, operator.dtype)
        ).to_numpy()
        applied_combined = operator.apply(
            _qualification_ndarray(combined, operator.dtype)
        ).to_numpy()
        absolute, relative = _qualification_error(
            applied_combined, alpha * applied_left + beta * applied_right
        )
        accumulate("linearity", absolute, relative)

        if forward_oracle is not None:
            expected = np.asarray(forward_oracle(left), dtype=numpy_dtype)
            if expected.shape != (rows,):
                raise TaichiRuntimeError(
                    f"reference callable must return shape ({rows},)"
                )
            accumulate(
                "forward_reference",
                *_qualification_error(applied_left, expected),
            )

        if adjoint_operator is not None:
            range_vector = rng.standard_normal(rows).astype(numpy_dtype)
            applied_adjoint = adjoint_operator.apply(
                _qualification_ndarray(range_vector, operator.dtype)
            ).to_numpy()
            lhs = float(np.dot(applied_left.astype(np.float64), range_vector))
            rhs = float(np.dot(left.astype(np.float64), applied_adjoint))
            absolute = abs(lhs - rhs)
            relative = absolute / max(
                abs(lhs), abs(rhs), np.finfo(np.float64).tiny
            )
            accumulate("adjoint_dot_product", absolute, relative)
            if adjoint_oracle is not None:
                expected_adjoint = np.asarray(
                    adjoint_oracle(range_vector), dtype=numpy_dtype
                )
                if expected_adjoint.shape != (columns,):
                    raise TaichiRuntimeError(
                        "adjoint_reference callable must return shape "
                        f"({columns},)"
                    )
                accumulate(
                    "adjoint_reference",
                    *_qualification_error(applied_adjoint, expected_adjoint),
                )

    def append_error_check(name):
        absolute, relative = maxima[name]
        checks.append(
            _qualification_check(
                name,
                absolute <= atol or relative <= rtol,
                {
                    "max_absolute_error": absolute,
                    "max_relative_error": relative,
                },
                {"atol": atol, "rtol": rtol},
            )
        )

    append_error_check("linearity")
    if forward_oracle is None:
        checks.append(
            {
                "name": "forward_reference",
                "status": "not_requested",
                "metrics": {},
                "tolerance": {"atol": atol, "rtol": rtol},
                "details": "No reference was supplied.",
            }
        )
    else:
        append_error_check("forward_reference")

    if adjoint_operator is None:
        checks.append(
            {
                "name": "adjoint_dot_product",
                "status": "unsupported",
                "metrics": {},
                "tolerance": {"atol": atol, "rtol": rtol},
                "details": "The provider does not claim adjoint_apply.",
            }
        )
    else:
        append_error_check("adjoint_dot_product")
        if adjoint_oracle is not None:
            append_error_check("adjoint_reference")

    for _ in range(warmup):
        operator.apply(timing_input, out=timing_output)
    warm_ns = []
    for _ in range(repetitions):
        start_ns = time.perf_counter_ns()
        operator.apply(timing_input, out=timing_output)
        warm_ns.append(time.perf_counter_ns() - start_ns)

    program = _current_program()
    final_statistics = operator.statistics()
    capabilities = {
        name: getattr(operator.capabilities, name)
        for name in OperatorCapabilities.__dataclass_fields__
    }
    record = {
        "schema": OperatorQualificationReport.SCHEMA,
        "schema_version": 1,
        "passed": not any(check["status"] == "failed" for check in checks),
        "environment": {
            "taichi_version": _ti_core.get_version_string(),
            "taichi_commit": _ti_core.get_commit_hash(),
            "backend": _ti_core.arch_name(program.config().arch),
            "device": None,
            "driver": None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "operator": {
            "provider": operator.provider,
            "provider_kind": operator._provider_kind,
            "execution_kind": operator.execution_kind,
            "shape": list(operator.shape),
            "dtype": operator._metadata_snapshot["dtype"],
            "capabilities": capabilities,
            "traits": copy.deepcopy(operator._metadata_snapshot["traits"]),
            "resource_stamp": copy.deepcopy(
                operator._metadata_snapshot["resource_stamp"]
            ),
        },
        "configuration": {
            "samples": samples,
            "seed": seed,
            "atol": atol,
            "rtol": rtol,
            "warmup": warmup,
            "repetitions": repetitions,
        },
        "checks": checks,
        "timing": {
            "boundary": "synchronous_public_apply",
            "first_apply_ms": first_apply_ns / 1e6,
            "warm_apply_ms": {
                "minimum": min(warm_ns) / 1e6,
                "median": float(np.median(warm_ns)) / 1e6,
                "maximum": max(warm_ns) / 1e6,
            },
            "device_time_available": False,
        },
        "statistics": {
            "before": initial_statistics,
            "after": final_statistics,
        },
        "metadata": custom_metadata,
    }
    return OperatorQualificationReport(record)


def summarize_operator_qualifications(reports):
    """Builds a deterministic backend/provider matrix from reports."""
    reports = tuple(reports)
    rows = []
    passed = 0
    failed = 0
    for report in reports:
        if not isinstance(report, OperatorQualificationReport):
            raise TypeError(
                "reports must contain OperatorQualificationReport values"
            )
        record = report.to_dict()
        checks = {check["name"]: check["status"] for check in record["checks"]}
        row = {
            "backend": record["environment"]["backend"],
            "provider": record["operator"]["provider"],
            "provider_kind": record["operator"]["provider_kind"],
            "execution_kind": record["operator"]["execution_kind"],
            "dtype": record["operator"]["dtype"],
            "shape": list(record["operator"]["shape"]),
            "passed": bool(record["passed"]),
            "checks": checks,
            "unsupported_checks": sorted(
                name
                for name, status in checks.items()
                if status == "unsupported"
            ),
        }
        rows.append(row)
        if row["passed"]:
            passed += 1
        else:
            failed += 1
    rows.sort(
        key=lambda row: (
            row["backend"],
            row["provider"],
            row["provider_kind"],
            row["execution_kind"],
            row["dtype"],
            tuple(row["shape"]),
        )
    )
    return {
        "schema": "taichi_forge.linalg.operator_qualification_matrix.v1",
        "schema_version": 1,
        "summary": {
            "reports": len(rows),
            "passed": passed,
            "failed": failed,
        },
        "rows": rows,
    }


def _validate_solve_controls(dtype, max_iterations, atol, rtol):
    if isinstance(max_iterations, bool):
        raise TaichiRuntimeError("max_iterations must be non-negative")
    try:
        max_iterations = _operator.index(max_iterations)
    except TypeError as exc:
        raise TaichiRuntimeError(
            "max_iterations must be a non-negative integer"
        ) from exc
    if max_iterations < 0:
        raise TaichiRuntimeError("max_iterations must be non-negative")

    def tolerance(name, value):
        if isinstance(value, bool):
            raise TaichiRuntimeError(f"{name} must be finite and non-negative")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TaichiRuntimeError(
                f"{name} must be finite and non-negative"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise TaichiRuntimeError(f"{name} must be finite and non-negative")
        if dtype == f32 and value > 3.4028235e38:
            raise TaichiRuntimeError(f"{name} is not representable as f32")
        return value

    atol = tolerance("atol", atol)
    rtol = tolerance("rtol", rtol)
    if atol == 0.0 and rtol == 0.0:
        raise TaichiRuntimeError("atol > 0 or rtol > 0 is required")
    return max_iterations, atol, rtol


def _cuda_device_convergent_status(cuda_conditional, provider_kind):
    if provider_kind != "stored" and cuda_conditional.get(
        "internal_masked_graph_available", False
    ):
        reason = "none"
    elif not cuda_conditional["driver_version_eligible"]:
        reason = "cuda_driver_api_version_below_12_8"
    elif not cuda_conditional["conditional_graph_symbols_loaded"]:
        reason = "cuda_conditional_graph_symbols_not_loaded"
    elif not cuda_conditional["runtime_path_compiled"]:
        reason = "cuda_conditional_graph_runtime_path_not_compiled"
    elif provider_kind == "stored" and not cuda_conditional[
        "device_setter_lowering_compiled"
    ]:
        reason = "cuda_conditional_setter_lowering_not_compiled"
    elif provider_kind != "stored" and not cuda_conditional.get(
        "general_device_setter_lowering_compiled", False
    ):
        reason = "cuda_conditional_setter_lowering_not_compiled"
    elif provider_kind == "stored" and not cuda_conditional[
        "cublas_workspace_symbol_loaded"
    ]:
        reason = "cublas_user_workspace_symbol_not_loaded"
    else:
        reason = "none"

    if provider_kind == "stored":
        prerequisites = (
            "conditional graph driver functions in the CUDA dynamic table",
            "stored-solver conditional-handle setter lowering",
            "cuBLAS user-workspace support",
            "stored provider capture/body/update qualification",
        )
    else:
        prerequisites = (
            "conditional graph driver functions in the CUDA dynamic table",
            "general Graph conditional-handle setter lowering",
            "recordable provider body/update qualification",
        )
    return reason == "none", reason, prerequisites


def _solver_execution_capabilities(
    program,
    provider_kind,
    *,
    batched,
    method=None,
    dtype=None,
    preconditioner_replay_qualified=True,
    provider_recordable=False,
):
    arch = program.config().arch
    cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
    is_cpu = arch in cpu_arches
    is_cuda = arch == _ti_core.Arch.cuda
    is_vulkan = arch == _ti_core.Arch.vulkan
    vulkan_structured_runtime_mode = bool(
        is_vulkan
        and not program.config().kernel_profiler
        and not program.config().vulkan_dispatch_cache
    )
    if is_cuda:
        conditional_primitive = "cuda_conditional_graph"
        cuda_conditional = dict(_ti_core.cuda_conditional_graph_capabilities())
        (
            cuda_device_convergent_available,
            unavailable_reason,
            prerequisites,
        ) = _cuda_device_convergent_status(
            cuda_conditional,
            provider_kind,
        )
        if (
            unavailable_reason == "none"
            and provider_kind != "stored"
            and not provider_recordable
        ):
            unavailable_reason = "provider_action_not_recordable"
    elif is_vulkan:
        conditional_primitive = "vulkan_dispatch_indirect"
        if not vulkan_structured_runtime_mode:
            unavailable_reason = "vulkan_runtime_mode_disables_graph_replay"
        elif provider_recordable:
            unavailable_reason = "none"
        elif provider_kind == "stored":
            unavailable_reason = (
                "vulkan_stored_solver_indirect_dispatch_path_not_compiled"
            )
        else:
            unavailable_reason = "vulkan_provider_action_not_recordable"
        prerequisites = (
            "qualified Vulkan structured Graph runtime",
            "recordable provider action",
            "fixed f32 dense vector and workspace bindings",
        )
    else:
        conditional_primitive = "none"
        unavailable_reason = "device_convergent_is_gpu_only"
        prerequisites = ()

    bounded_provider_qualified = provider_kind in ("stored", "kernel")
    bounded_qualified = (
        not batched
        and bounded_provider_qualified
        and preconditioner_replay_qualified
        and method in ("cg", "pcg")
        and dtype == f32
        and (is_cpu or is_cuda or is_vulkan)
    )
    recordable_device_qualified = (
        provider_recordable
        and preconditioner_replay_qualified
        and method in ("cg", "pcg")
        and dtype == f32
        and (is_cuda or is_vulkan)
    )
    if is_cpu:
        bounded_primitive = "native_cpu_solver_loop"
    elif is_cuda:
        bounded_primitive = "cuda_graph_chunked_host_check"
    elif is_vulkan:
        bounded_primitive = "vulkan_command_chunked_host_check"
    else:
        bounded_primitive = "none"

    policies = {
        "host_each_iteration": (
            is_cpu
            or (
                provider_kind != "composition"
                and is_cuda
                and method not in ("gmres", "fgmres")
            )
            or (
                provider_kind != "composition"
                and batched
                and is_vulkan
            )
        ),
        "host_check_every_k": (
            provider_kind != "composition" and (is_cuda or is_vulkan)
        ),
        "fixed_budget_masked": (
            provider_kind != "composition"
            and (is_vulkan or (batched and is_cuda))
        ),
        "bounded_convergent": bounded_qualified,
        "device_convergent": (
            (bounded_qualified or recordable_device_qualified)
            and (
                (
                    is_cuda
                    and cuda_device_convergent_available
                    and (
                        provider_kind == "stored" or provider_recordable
                    )
                )
                or (
                    is_vulkan
                    and vulkan_structured_runtime_mode
                    and provider_recordable
                )
            )
        ),
    }
    native_upgrade_available = (
        bounded_qualified
        and (
            (
                is_cuda
                and cuda_device_convergent_available
                and (provider_kind == "stored" or provider_recordable)
            )
            or (
                is_vulkan
                and vulkan_structured_runtime_mode
                and provider_recordable
            )
        )
    )
    native_upgrade_automatic = (
        native_upgrade_available
        and (
            provider_kind == "stored"
            or (
                is_vulkan
                and provider_kind in ("kernel", "composition")
                and provider_recordable
            )
        )
    )
    device_automatic_selection = not batched and (
        native_upgrade_automatic
        or (
            recordable_device_qualified
            and (
                provider_kind in ("graph_action", "composition")
                or (provider_kind == "graph" and method == "pcg")
            )
            and policies["device_convergent"]
        )
    )
    native_replay_qualified = (
        not batched
        and provider_kind == "stored"
        and preconditioner_replay_qualified
        and dtype == f32
        and method in ("cg", "pcg", "minres", "bicgstab", "gmres")
        and (is_cuda or is_vulkan)
    )
    matrix_free_provider = provider_kind in ("kernel", "graph")
    matrix_free_batching_qualified = (
        not batched
        and matrix_free_provider
        and dtype == f32
        and (is_cuda or is_vulkan)
        and not (is_cuda and method == "bicgstab" and provider_kind == "graph")
        and (
            method in ("minres", "bicgstab", "gmres", "fgmres")
            or (method == "cg" and provider_kind in ("kernel", "graph"))
            or (method == "pcg" and provider_kind == "kernel")
        )
    )
    if batched:
        default_execution_policy = (
            "host_each_iteration" if is_cpu else "host_check_every_k"
        )
    elif is_cuda and bounded_qualified and provider_kind == "stored":
        default_execution_policy = "bounded_convergent"
    elif is_cuda and device_automatic_selection:
        default_execution_policy = "device_convergent"
    elif is_cuda and (
        native_replay_qualified
        or matrix_free_batching_qualified
        or method in ("gmres", "fgmres")
    ):
        default_execution_policy = "host_check_every_k"
    elif is_vulkan and device_automatic_selection:
        default_execution_policy = "device_convergent"
    elif is_vulkan and (
        native_replay_qualified or matrix_free_batching_qualified
    ):
        default_execution_policy = "host_check_every_k"
    elif is_vulkan:
        default_execution_policy = "fixed_budget_masked"
    else:
        default_execution_policy = "host_each_iteration"
    device_unavailable_reason = (
        unavailable_reason
        if bounded_qualified or recordable_device_qualified
        else "solver_contract_not_qualified_for_device_convergent"
    )
    return {
        "backend": _ti_core.arch_name(arch),
        "provider_kind": provider_kind,
        "execution_policies": policies,
        "bounded_convergent": {
            "supported": bounded_qualified,
            "primitive": bounded_primitive,
            "qualified_methods": ("cg", "pcg"),
            "qualified_provider_kinds": ("stored", "kernel"),
            "qualified_dtypes": ("f32",),
            "chunk_schedule": (1, 1, 2, 4, 8, 16),
            "host_observation_scope": (
                "none_inside_python" if is_cpu else "chunk_boundaries_only"
            ),
            "native_upgrade_available": native_upgrade_available,
            "native_upgrade_automatic": native_upgrade_automatic,
            "native_upgrade_primitive": conditional_primitive,
            "native_upgrade_unavailable_reason": unavailable_reason,
        },
        "device_convergent": {
            "supported": policies["device_convergent"],
            "primitive": conditional_primitive,
            "rhi_primitive_compiled": is_vulkan,
            "runtime_path_compiled": (
                cuda_conditional["runtime_path_compiled"]
                if is_cuda
                else is_vulkan
            ),
            "provider_qualified": (
                (bounded_qualified or recordable_device_qualified)
                and (
                    (is_cuda and (provider_kind == "stored" or provider_recordable))
                    or (
                        is_vulkan
                        and vulkan_structured_runtime_mode
                        and provider_recordable
                    )
                )
            ),
            "automatic_selection_qualified": device_automatic_selection,
            "qualification_scope": (
                "automatic"
                if device_automatic_selection
                else (
                    "explicit_only"
                    if policies["device_convergent"]
                    else "unsupported"
                )
            ),
            "automatic_selection_unavailable_reason": (
                "none"
                if device_automatic_selection
                else (
                    "compiled_kernel_graph_krylov_not_latency_qualified"
                    if policies["device_convergent"]
                    and provider_kind
                    in ("kernel", "graph", "graph_action", "composition")
                    else device_unavailable_reason
                )
            ),
            "unsupported_reason": (
                "none"
                if policies["device_convergent"]
                else device_unavailable_reason
            ),
            "prerequisites": prerequisites,
        },
        "cuda_conditional_graph": (cuda_conditional if is_cuda else None),
        "bounded_mode_selection": True,
        "default_execution_policy": default_execution_policy,
        "automatic_policy_change": (
            not batched
            and (
                default_execution_policy == "bounded_convergent"
                or default_execution_policy == "device_convergent"
                or native_replay_qualified
                or matrix_free_batching_qualified
            )
        ),
        "automatic_solver_batching": {
            "selected": (
                matrix_free_batching_qualified
                and default_execution_policy == "host_check_every_k"
            ),
            "qualified": matrix_free_batching_qualified,
            "unavailable_reason": (
                "none"
                if matrix_free_batching_qualified
                else (
                    "cuda_bicgstab_graph_k4_not_stably_beneficial"
                    if is_cuda
                    and method == "bicgstab"
                    and provider_kind == "graph"
                    else "matrix_free_solver_batching_not_qualified"
                )
            ),
            "qualified_provider_kinds": ("kernel", "graph"),
            "primitive": (
                "cuda_direct_chunk_host_check"
                if is_cuda and matrix_free_batching_qualified
                else (
                    "vulkan_direct_chunk_host_check"
                    if is_vulkan and matrix_free_batching_qualified
                    else "none"
                )
            ),
            "default_check_interval": (
                "restart"
                if method in ("gmres", "fgmres")
                and matrix_free_batching_qualified
                else (4 if matrix_free_batching_qualified else None)
            ),
            "solver_replay_required": False,
            "provider_execution": (
                "compiled_graph_plan_per_apply"
                if provider_kind == "graph" and matrix_free_batching_qualified
                else (
                    "compiled_kernel_direct_apply"
                    if provider_kind == "kernel"
                    and matrix_free_batching_qualified
                    else "none"
                )
            ),
        },
        "automatic_solver_replay": {
            "selected": (
                not batched
                and (
                    default_execution_policy == "bounded_convergent"
                    or default_execution_policy == "device_convergent"
                    or native_replay_qualified
                )
            ),
            "qualified": native_replay_qualified
            or (bounded_qualified and is_cuda)
            or policies["device_convergent"],
            "preconditioner_qualified": preconditioner_replay_qualified,
            "primitive": (
                "cuda_conditional_graph_or_chunk_replay"
                if is_cuda and bounded_qualified
                else (
                    "cuda_graph_chunk_replay"
                    if is_cuda and native_replay_qualified
                    else (
                        "vulkan_structured_graph"
                        if is_vulkan and policies["device_convergent"]
                        else (
                            "vulkan_command_replay"
                            if is_vulkan and native_replay_qualified
                            else "none"
                        )
                    )
                )
            ),
        },
        "explicit_request_fallback": False,
    }


class PreconditionerSession:
    """Pinned immutable target/action generations for one consumer scope.

    A variable-linear session owns the complete action-table snapshot.
    `iteration` selects the same cyclic action that FGMRES uses for that
    scheduled Arnoldi slot. Selection is local and deterministic; it does
    not invoke a Python callback from a solver hot loop.
    """

    def __init__(self, plan, native):
        self._plan = plan
        self._native = native
        self._natives = (
            tuple(native) if isinstance(native, (tuple, list)) else (native,)
        )
        self._program = plan._program
        action_metadata = tuple(
            _readonly_copy(dict(item._metadata())) for item in self._natives
        )
        if plan.behavior == "fixed_linear":
            self._metadata_snapshot = dict(action_metadata[0])
        else:
            self._metadata_snapshot = {
                "schema_version": 1,
                "behavior": "variable_linear",
                "selection": plan.selection,
                "period": len(action_metadata),
                "actions": action_metadata,
            }
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self):
        if self._native is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "PreconditionerSession cannot be used after ti.reset()"
            )

    def _invalidate_runtime(self):
        self._native = None
        self._natives = ()
        self._plan = None
        self._program = None

    @property
    def metadata(self):
        return _readonly_copy(self._metadata_snapshot)

    def apply(self, residual, out=None, *, iteration=0):
        """Applies the pinned approximate-inverse action synchronously."""
        self._ensure_valid()
        if isinstance(iteration, bool):
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            )
        try:
            iteration = _operator.index(iteration)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            ) from exc
        if iteration < 0:
            raise TaichiRuntimeError(
                "iteration must be a non-negative integer"
            )
        action_index = (
            iteration % len(self._natives)
            if self._plan.behavior == "variable_linear"
            else 0
        )
        action = self._plan.actions[action_index]
        residual = _require_current_scalar_ndarray(
            residual,
            "PreconditionerSession residual",
            action.shape[1],
            action.dtype,
        )
        if out is None:
            out = ScalarNdarray(action.dtype, (action.shape[0],))
        else:
            out = _require_current_scalar_ndarray(
                out,
                "PreconditionerSession output",
                action.shape[0],
                action.dtype,
            )
        if out is residual:
            raise TaichiRuntimeError(
                "PreconditionerSession input/output may not alias"
            )
        self._natives[action_index]._apply(
            self._program, residual.arr, out.arr
        )
        return out


class PreconditionerPlan:
    """Versions a target operator and external approximate-inverse actions.

    ``fixed_linear`` owns one action. ``variable_linear`` owns a finite cyclic
    table selected by the solve-global scheduled inner slot:
    ``actions[k % len(actions)]``. External code publishes target/action
    numeric generations through their ordinary
    :class:`LinearOperator` providers, then calls :meth:`update` to attest a
    rebuild or an explicit lagged reuse. No Python callback runs in
    :meth:`apply`, a pinned session, or a solver iteration.
    """

    _UNSUPPORTED_BEHAVIORS = {
        "nonlinear": (
            "nonlinear preconditioners have no qualified solver consumer"
        ),
    }

    def __init__(
        self,
        target,
        action,
        *,
        method="external",
        behavior="fixed_linear",
        selection=None,
    ):
        if not isinstance(target, LinearOperator):
            raise TypeError("target must be ti.linalg.LinearOperator")
        target._ensure_valid()
        if isinstance(action, LinearOperator):
            actions = (action,)
        elif isinstance(action, (tuple, list)):
            actions = tuple(action)
        else:
            raise TypeError(
                "action must be a LinearOperator or a finite action sequence"
            )
        if not actions:
            raise TaichiRuntimeError(
                "PreconditionerPlan action sequence must not be empty"
            )
        if len(actions) > 32:
            raise TaichiRuntimeError(
                "PreconditionerPlan supports at most 32 scheduled actions"
            )
        for item in actions:
            if not isinstance(item, LinearOperator):
                raise TypeError(
                    "every PreconditionerPlan action must be a "
                    "LinearOperator"
                )
            item._ensure_valid()
            if target._program is not item._program:
                raise TaichiRuntimeError(
                    "PreconditionerPlan target and actions must share a "
                    "runtime"
                )
        expected_shape = (target.shape[1], target.shape[0])
        if target.shape[0] != target.shape[1]:
            raise TaichiRuntimeError(
                "PreconditionerPlan currently requires a square target"
            )
        for item in actions:
            if item.shape != expected_shape or item.dtype != target.dtype:
                raise TaichiRuntimeError(
                    "every PreconditionerPlan action must map the target "
                    "range back to its domain with the same dtype"
                )
        if not isinstance(method, str) or not method.strip():
            raise TaichiRuntimeError(
                "PreconditionerPlan method must be a non-empty string"
            )
        if not isinstance(behavior, str):
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior must be a string"
            )
        behavior = behavior.casefold()
        if behavior not in (
            "fixed_linear",
            "variable_linear",
            *self._UNSUPPORTED_BEHAVIORS,
        ):
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior must be fixed_linear, "
                "variable_linear, or nonlinear"
            )
        if behavior == "fixed_linear":
            if len(actions) != 1:
                raise TaichiRuntimeError(
                    "fixed_linear PreconditionerPlan requires exactly one "
                    "action"
                )
            if selection is not None:
                raise TaichiRuntimeError(
                    "selection is accepted only for variable_linear behavior"
                )
            selection = "fixed"
        elif behavior == "variable_linear":
            if selection is None:
                selection = "cyclic"
            if not isinstance(selection, str) or (
                selection.casefold() != "cyclic"
            ):
                raise TaichiRuntimeError(
                    "variable_linear PreconditionerPlan currently requires "
                    "selection='cyclic'"
                )
            selection = "cyclic"
        elif selection is not None:
            raise TaichiRuntimeError(
                "selection is accepted only for variable_linear behavior"
            )
        self.target = target
        self.actions = actions
        self.action = actions[0] if behavior == "fixed_linear" else actions
        self.method = method.strip()
        self.behavior = behavior
        self.selection = selection
        self._program = target._program
        self._unsupported_reason = self._UNSUPPORTED_BEHAVIORS.get(behavior)
        self._handle = None
        self._handles = ()
        self._consumer_action = None
        self._consumer_actions = ()
        self._schedule_update_calls = 0
        self._schedule_update_successes = 0
        self._schedule_update_failures = 0
        if self._unsupported_reason is None:
            self._handles = tuple(
                _ti_core._make_experimental_preconditioner_plan(
                    self._program,
                    target._handle,
                    item._handle,
                    self.method,
                )
                for item in actions
            )
            self._handle = self._handles[0]
        get_runtime().register_runtime_object(self)

    def _ensure_valid(self, *, require_supported=True):
        if self._program is not _current_program() or self.target is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan cannot be used after ti.reset()"
            )
        self.target._ensure_valid()
        for action in self.actions:
            action._ensure_valid()
        if require_supported and self._handle is None:
            raise TaichiRuntimeError(
                "PreconditionerPlan behavior is unsupported: "
                f"{self._unsupported_reason}"
            )

    def _invalidate_runtime(self):
        self._consumer_action = None
        self._consumer_actions = ()
        self._handle = None
        self._handles = ()
        self.target = None
        self.action = None
        self.actions = ()
        self._program = None

    def _require_target(self, target):
        self._ensure_valid()
        if target is not self.target:
            raise TaichiRuntimeError(
                "PreconditionerPlan was built for a different target "
                "LinearOperator"
            )

    def _require_fixed_behavior(self, consumer):
        if self.behavior != "fixed_linear":
            raise TaichiRuntimeError(
                f"{consumer} requires behavior='fixed_linear'; "
                "variable_linear actions are consumed only by FGMRES"
            )

    @property
    def metadata(self):
        self._ensure_valid(require_supported=False)
        if self._handle is None:
            return _readonly_copy(
                {
                    "schema_version": 1,
                    "method": self.method,
                    "behavior": self.behavior,
                    "supported": False,
                    "unsupported_reason": self._unsupported_reason,
                    "is_setup": False,
                }
            )
        action_metadata = tuple(
            dict(handle._metadata()) for handle in self._handles
        )
        result = dict(action_metadata[0])
        result["behavior"] = self.behavior
        result["selection"] = self.selection
        result["period"] = len(self.actions)
        result["target_provider"] = self.target.provider
        result["action_providers"] = tuple(
            action.provider for action in self.actions
        )
        if self.behavior == "fixed_linear":
            result["action_provider"] = self.actions[0].provider
        else:
            result["is_setup"] = all(
                item["is_setup"] for item in action_metadata
            )
            result["built_from_operator_stamps"] = tuple(
                item["built_from_operator_stamp"] for item in action_metadata
            )
            result["accepted_target_stamps"] = tuple(
                item["accepted_target_stamp"] for item in action_metadata
            )
            result["accepted_action_stamps"] = tuple(
                item["accepted_action_stamp"] for item in action_metadata
            )
            result["actions"] = action_metadata
        return _readonly_copy(result)

    def setup(self):
        """Attests that current actions were built from the current target."""
        self._ensure_valid()
        for handle in self._handles:
            handle._setup(self._program)
        consumer_handles = tuple(
            _ti_core._make_experimental_preconditioner_action(
                self._program, handle
            )
            for handle in self._handles
        )
        self._consumer_actions = tuple(
            LinearOperator._from_handle(
                consumer_handle,
                provider_kind=action._provider_kind,
                retained=(self.target, action),
            )
            for consumer_handle, action in zip(consumer_handles, self.actions)
        )
        self._consumer_action = self._consumer_actions[0]
        return self

    def update(self, *, accept_reuse=False):
        """Approves current generations as a rebuild or explicit reuse.

        With the default ``accept_reuse=False``, a changed target requires
        every action generation to change. Variable-linear plans also accept
        one boolean per action, allowing an explicit mix of rebuilt and
        lagged actions. Reuse preserves each action's original
        ``built_from_operator_stamp`` provenance.
        """
        self._ensure_valid()
        if isinstance(accept_reuse, bool):
            reuse = (accept_reuse,) * len(self._handles)
        elif self.behavior == "variable_linear" and isinstance(
            accept_reuse, (tuple, list)
        ):
            reuse = tuple(accept_reuse)
            if len(reuse) != len(self._handles) or not all(
                isinstance(item, bool) for item in reuse
            ):
                raise TaichiRuntimeError(
                    "accept_reuse sequence must contain one bool per action"
                )
        else:
            raise TaichiRuntimeError(
                "accept_reuse must be bool or one bool per variable action"
            )
        self._schedule_update_calls += 1
        try:
            if self.behavior == "variable_linear":
                for handle, reuse_action in zip(self._handles, reuse):
                    handle._validate_update(self._program, reuse_action)
            for handle, reuse_action in zip(self._handles, reuse):
                handle._update(self._program, reuse_action)
        except Exception:
            self._schedule_update_failures += 1
            raise
        self._schedule_update_successes += 1
        return self

    def pin(self):
        """Pins the complete approved target/action-table snapshot."""
        self._ensure_valid()
        natives = tuple(handle._pin(self._program) for handle in self._handles)
        return PreconditionerSession(
            self, natives if self.behavior == "variable_linear" else natives[0]
        )

    def apply(self, residual, out=None, *, iteration=0):
        """Pins the approved snapshot and applies its selected action once."""
        return self.pin().apply(residual, out=out, iteration=iteration)

    def statistics(self):
        self._ensure_valid()
        action_statistics = tuple(
            dict(handle._debug_runtime_stats()) for handle in self._handles
        )
        if self.behavior == "fixed_linear":
            result = dict(action_statistics[0])
        else:
            summed_keys = (
                "setup_calls",
                "update_calls",
                "update_successes",
                "update_noops",
                "update_failures",
                "target_generation_changes",
                "action_generation_changes",
                "rebuild_attestations",
                "reuse_attestations",
                "pins",
                "apply_calls",
                "stale_rejections",
                "approved_generations_published",
                "approved_generations_retired",
                "approved_generations_released",
                "approved_generation_active_leases",
            )
            result = {
                key: sum(item[key] for item in action_statistics)
                for key in summed_keys
            }
            result.update(
                {
                    "schema_version": 1,
                    "behavior": self.behavior,
                    "selection": self.selection,
                    "period": len(self.actions),
                    "has_current_approved_generation": all(
                        item["has_current_approved_generation"]
                        for item in action_statistics
                    ),
                    "actions": action_statistics,
                }
            )
        result["behavior"] = self.behavior
        result["selection"] = self.selection
        result["period"] = len(self.actions)
        result["schedule_update_calls"] = self._schedule_update_calls
        result["schedule_update_successes"] = self._schedule_update_successes
        result["schedule_update_failures"] = self._schedule_update_failures
        return result


class _SolvePlanGraphExecutable(NativeGraphExecutable):
    def __init__(
        self,
        plan,
        solver,
        rhs_arg,
        output_arg,
        initial_arg,
        terminal,
        action_record,
        *,
        name,
    ):
        from taichi_forge.graph._graph import ArgKind

        vector_args = [(rhs_arg, "RHS"), (output_arg, "output")]
        if initial_arg is not None:
            vector_args.append((initial_arg, "initial_guess"))
        for value, role in vector_args:
            if (
                getattr(value, "tag", None) != ArgKind.NDARRAY
                or value.dtype() != f32
                or int(value.field_dim) != 1
                or tuple(value.element_shape) != ()
            ):
                raise TaichiRuntimeError(
                    f"SolvePlan Graph {role} must be a symbolic scalar f32 "
                    "1-D ndarray"
                )
        if rhs_arg.name == output_arg.name:
            raise TaichiRuntimeError(
                "SolvePlan Graph RHS and output must use distinct resources"
            )
        if initial_arg is not None and initial_arg.name == rhs_arg.name:
            raise TaichiRuntimeError(
                "SolvePlan Graph initial_guess and RHS must use distinct resources"
            )
        self._plan = plan
        self._solver = solver
        self._rhs_name = rhs_arg.name
        self._output_name = output_arg.name
        self._initial_name = (
            None if initial_arg is None else initial_arg.name
        )
        self._terminal = terminal
        self._action_record = action_record
        self._name = name
        # GraphKrylovSolver owns the exact operator used to materialize the
        # recorded dispatch sequence. Snapshot that provider and its schema so
        # later assignments to the parent plan cannot splice a different
        # operator into lifetime or public-binding validation.
        self._operator = solver._operator
        self._validate_operator_identity()
        self._operator_shape = tuple(self._operator.shape)
        self._expected_operator_stamp = tuple(self._operator._handle._resource_stamp())
        schema_names = [self._rhs_name, self._output_name]
        if self._initial_name not in (None, self._output_name):
            schema_names.append(self._initial_name)
        schema_names.extend((terminal.state.name, terminal.metrics.name))
        self._runtime_arg_schema = tuple(
            RuntimeBinding(name, "dense_vector")
            for name in schema_names[: 2 + int(
                self._initial_name not in (None, self._output_name)
            )]
        ) + (
            RuntimeBinding(terminal.state.name, "terminal_state"),
            RuntimeBinding(terminal.metrics.name, "terminal_metrics"),
        )
        effects = [
            ResourceEffect(self._rhs_name, GraphAccess.READ),
            ResourceEffect(self._output_name, GraphAccess.WRITE),
        ]
        if self._initial_name not in (None, self._output_name):
            effects.append(ResourceEffect(self._initial_name, GraphAccess.READ))
        effects.extend(
            (
                ResourceEffect(terminal.state.name, GraphAccess.WRITE),
                ResourceEffect(terminal.metrics.name, GraphAccess.WRITE),
            )
        )
        self._resource_effects = tuple(effects)
        self._graph_binding_views = {}
        private_prefix = f"__solve_plan_{id(self):x}"
        self._sequence = solver.recordable_sequence(
            rhs_arg,
            output_arg,
            terminal.state,
            terminal.metrics,
            initial_guess=initial_arg,
            private_prefix=private_prefix,
            name=name,
        )
        self._action_record.record_compiled()

    def _begin_graph_submission_observation(self):
        return self._action_record.begin_submission()

    def _record_synchronous_graph_execution(self):
        self._action_record.record_synchronous_execution()

    @property
    def runtime_arg_schema(self):
        return self._runtime_arg_schema

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def recordable_sequence(self):
        return self._sequence

    @property
    def debug_info(self):
        return {
            "kind": "solve_plan",
            "method": self._plan.method,
            "name": self._name,
            "recordable": True,
            "terminal_state": self._terminal.state.name,
            "terminal_metrics": self._terminal.metrics.name,
        }

    def _validate_operator_identity(self):
        if self._plan.operator is not self._operator:
            raise TaichiRuntimeError(
                "SolvePlan operator changed after its Graph action was created; "
                "rebuild the SolvePlan and Graph"
            )

    def validate_graph_lifetime(self):
        if self._plan._program is not _current_program():
            raise TaichiRuntimeError(
                "SolvePlan Graph action belongs to another runtime"
            )
        self._validate_operator_identity()
        self._operator._ensure_valid()
        if tuple(self._operator._handle._resource_stamp())[:4] != (
            self._expected_operator_stamp[:4]
        ):
            raise TaichiRuntimeError(
                "SolvePlan operator topology changed; rebuild the Graph"
            )

    def _terminal_array(self, runtime_args, symbol, dtype, role):
        value = runtime_args[symbol.name]
        value = _require_current_scalar_ndarray(value, role, dtype=dtype)
        if value.shape != (4,):
            raise TaichiRuntimeError(f"{role} must have shape (4,)")
        return value

    def bind_graph_arguments(self, runtime_args):
        replacements = {}
        names = [self._rhs_name, self._output_name]
        if self._initial_name not in (None, self._output_name):
            names.append(self._initial_name)
        for name in names:
            _, view = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                self._operator_shape[0],
            )
            if view is not None:
                replacements[name] = view
        return replacements

    def validate_graph_bindings(self, runtime_args):
        descriptions = {}
        names = [self._rhs_name, self._output_name]
        if self._initial_name not in (None, self._output_name):
            names.append(self._initial_name)
        for name in names:
            description, _ = _cached_graph_dense_vector_binding(
                self._graph_binding_views,
                name,
                runtime_args[name],
                self._operator_shape[0],
            )
            descriptions[name] = description
        _validate_graph_dense_vector_disjoint(
            (descriptions[self._rhs_name], descriptions[self._output_name]),
            "SolvePlan Graph RHS and output must be proven disjoint",
        )
        if self._initial_name not in (None, self._output_name):
            _validate_graph_dense_vector_disjoint(
                (
                    descriptions[self._rhs_name],
                    descriptions[self._initial_name],
                ),
                "SolvePlan Graph RHS and initial_guess must be proven disjoint",
            )
            initial_output_alias = analyze_storage_alias(
                descriptions[self._initial_name],
                descriptions[self._output_name],
            )
            initial_value = runtime_args[self._initial_name]
            output_value = runtime_args[self._output_name]
            exact_view = initial_value is output_value or (
                getattr(initial_value, "_exact_view_key", None) is not None
                and getattr(initial_value, "_exact_view_key", None)
                == getattr(output_value, "_exact_view_key", None)
            )
            if initial_output_alias != "kProvenDisjoint" and not exact_view:
                raise TaichiRuntimeError(
                    "SolvePlan Graph initial_guess and output overlap without "
                    f"being the same vector view ({initial_output_alias})"
                )
        self._terminal_array(
            runtime_args,
            self._terminal.state,
            i32,
            "SolvePlan Graph terminal state",
        )
        self._terminal_array(
            runtime_args,
            self._terminal.metrics,
            f32,
            "SolvePlan Graph terminal metrics",
        )

    def run(self, runtime_args):
        # Direct executable fallback does not pass through Graph's lifecycle
        # boundary. Keep it on the same exact-operator contract as the
        # recorded sequence before delegating to the parent plan.
        self._validate_operator_identity()
        result = self._plan.solve(
            runtime_args[self._rhs_name],
            initial_guess=(
                None
                if self._initial_name is None
                else runtime_args[self._initial_name]
            ),
            out=runtime_args[self._output_name],
        )
        terminal_state = self._terminal_array(
            runtime_args,
            self._terminal.state,
            i32,
            "SolvePlan Graph terminal state",
        )
        terminal_metrics = self._terminal_array(
            runtime_args,
            self._terminal.metrics,
            f32,
            "SolvePlan Graph terminal metrics",
        )
        terminal_state.from_numpy(
            np.asarray(
                [
                    result.status_code,
                    result.iterations,
                    int(result.breakdown),
                    1,
                ],
                dtype=np.int32,
            )
        )
        terminal_metrics.from_numpy(
            np.asarray(
                [
                    result.initial_residual_norm**2,
                    result.residual_norm**2,
                    result.relative_reference_norm**2,
                    result.effective_tolerance**2,
                ],
                dtype=np.float32,
            )
        )


class _SolvePlanGraphNode(NativeGraphNode):
    def __init__(
        self,
        plan,
        solver,
        rhs_arg,
        output_arg,
        initial_arg,
        terminal,
        action_record,
        *,
        name,
    ):
        self._plan = plan
        self._solver = solver
        self._rhs_arg = rhs_arg
        self._output_arg = output_arg
        self._initial_arg = initial_arg
        self.terminal = terminal
        self._action_record = action_record
        self.name = name

    def allocate_terminal(self, *, initialize=True):
        return self.terminal.allocate(initialize=initialize)

    def statistics(self):
        """Return execution attribution for this recorded solve action."""

        return self._action_record.statistics()

    def compile(self):
        return _SolvePlanGraphExecutable(
            self._plan,
            self._solver,
            self._rhs_arg,
            self._output_arg,
            self._initial_arg,
            self.terminal,
            self._action_record,
            name=self.name,
        )


class SolvePlan:
    """Persistent CG, PCG, MINRES, BiCGSTAB, GMRES, or FGMRES plan.

    ``pcg`` accepts fixed stored CSR/BSR providers with explicit ``"jacobi"``
    or ``"block_jacobi"`` selection. It also accepts a trusted SPD
    :class:`LinearOperator` or an explicitly versioned
    :class:`PreconditionerPlan` as a fixed-linear preconditioner. GPU
    device-convergent PCG requires recordable f32 A and M actions; qualified
    compiled-kernel, compiled-Graph, and composed providers share that
    capability contract.
    MINRES supports CPU ``f32``/``f64`` identity preconditioning and device-
    resident CUDA/Vulkan ``f32`` identity or trusted fixed-linear SPD
    preconditioning. BiCGSTAB supports CPU ``f32``/``f64`` host actions and
    device-resident CUDA/Vulkan ``f32`` actions with an optional fixed-linear
    right preconditioner. Vulkan supports bounded masked
    execution or chunked host convergence checks, including relative tolerance.
    """

    def __init__(
        self,
        operator,
        *,
        method="cg",
        preconditioner=None,
        max_iterations=50,
        atol=1e-6,
        rtol=0.0,
        execution_policy=None,
        check_interval=None,
        bounded_mode="auto",
        restart=None,
        submission_workspace_lanes=1,
        submission_workspace_saturation="wait",
    ):
        if not isinstance(operator, LinearOperator):
            raise TypeError("operator must be ti.linalg.LinearOperator")
        operator._ensure_valid()
        method = str(method).casefold()
        if method not in (
            "cg",
            "pcg",
            "minres",
            "bicgstab",
            "gmres",
            "fgmres",
        ):
            raise TaichiRuntimeError(
                "SolvePlan method must be 'cg', 'pcg', 'minres', "
                "'bicgstab', 'gmres', or 'fgmres'"
            )
        if operator.shape[0] != operator.shape[1]:
            raise TaichiRuntimeError(
                "Krylov SolvePlan requires a square operator"
            )
        max_iterations, atol, rtol = _validate_solve_controls(
            operator.dtype, max_iterations, atol, rtol
        )
        self.operator = operator
        self.method = method
        self.max_iterations = max_iterations
        self.atol = atol
        self.rtol = rtol
        self.preconditioner = preconditioner
        self._preconditioner_replay_qualified = preconditioner is None or (
            method in ("pcg", "minres") and isinstance(preconditioner, str)
        )
        if method == "pcg" and isinstance(preconditioner, LinearOperator):
            self._preconditioner_replay_qualified = bool(
                preconditioner.dtype == f32
                and preconditioner._supports_graph_action()
            )
        elif method == "pcg" and isinstance(
            preconditioner, PreconditionerPlan
        ):
            action = preconditioner._consumer_action
            self._preconditioner_replay_qualified = bool(
                preconditioner.behavior == "fixed_linear"
                and action is not None
                and action.dtype == f32
                and action._supports_graph_action()
            )
        if method in ("gmres", "fgmres"):
            if restart is None:
                restart = 16
            if isinstance(restart, bool):
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                )
            try:
                restart = _operator.index(restart)
            except TypeError as exc:
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                ) from exc
            if restart not in (8, 16, 32):
                raise TaichiRuntimeError(
                    f"{method.upper()} restart must be one of 8, 16, or 32"
                )
        elif restart is not None:
            raise TaichiRuntimeError(
                "restart is accepted only for method='gmres' or 'fgmres'"
            )
        self.restart = restart
        if isinstance(submission_workspace_lanes, bool):
            raise TaichiRuntimeError(
                "submission_workspace_lanes must be an integer"
            )
        try:
            submission_workspace_lanes = _operator.index(
                submission_workspace_lanes
            )
        except TypeError as exc:
            raise TaichiRuntimeError(
                "submission_workspace_lanes must be an integer"
            ) from exc
        if not 1 <= submission_workspace_lanes <= 64:
            raise TaichiRuntimeError(
                "submission_workspace_lanes must be between 1 and 64"
            )
        if submission_workspace_saturation not in ("wait", "raise"):
            raise TaichiRuntimeError(
                "submission_workspace_saturation must be 'wait' or 'raise'"
            )
        self.submission_workspace_lanes = submission_workspace_lanes
        self.submission_workspace_saturation = (
            submission_workspace_saturation
        )
        self._program = _current_program()
        if not isinstance(bounded_mode, str):
            raise TaichiRuntimeError("bounded_mode must be a string")
        self.bounded_mode = bounded_mode.casefold()
        if self.bounded_mode not in (
            "auto",
            "portable",
            "native_required",
        ):
            raise TaichiRuntimeError(
                "bounded_mode must be 'auto', 'portable', or "
                "'native_required'"
            )
        self.execution_policy, self.check_interval = (
            self._normalize_execution_policy(execution_policy, check_interval)
        )
        self._native_preconditioner = None
        self._vector_io = _VectorIOCache(
            allow_native_bulk=self._program.config().arch
            in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.vulkan,
            )
        )
        self._solver = self._build_solver()
        self._uses_graph_krylov = hasattr(self._solver, "solve_arrays")
        self._graph_krylov_direct_field_enabled = True
        self._graph_krylov_last_direct_field_boundary = False
        self._graph_krylov_binding_cache = {}
        self._graph_action_solver = None
        self._graph_action_serial = 0
        self._graph_action_records = []
        self._submission_lock = threading.RLock()
        self._submission_graphs = {}
        self._submission_graph_builds = 0
        self._submission_calls = 0
        self._submission_successes = 0
        self._submission_failures = 0
        self._submission_telemetry_requests = 0
        self._submission_terminal_materializations = 0
        self._submission_native_completed_results = 0
        get_runtime().register_runtime_object(self)

    def _invalidate_runtime(self):
        # Solver plans own backend workspaces and must release them before the
        # Program allocator/backend teardown.
        self._solver = None
        self._uses_graph_krylov = False
        self._graph_krylov_direct_field_enabled = False
        self._graph_krylov_last_direct_field_boundary = False
        self._graph_krylov_binding_cache = None
        self._graph_action_solver = None
        self._graph_action_records = None
        self._submission_graphs = None
        self._native_preconditioner = None
        self._vector_io = None
        self.operator = None
        self._program = None

    def _execution_policy_capabilities(self):
        return _solver_execution_capabilities(
            self._program,
            self.operator._provider_kind,
            batched=False,
            method=self.method,
            dtype=self.operator.dtype,
            preconditioner_replay_qualified=(
                self._preconditioner_replay_qualified
            ),
            provider_recordable=self.operator._supports_graph_action(),
        )

    def _normalize_execution_policy(self, policy, check_interval):
        arch = self._program.config().arch
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        capabilities = self._execution_policy_capabilities()
        if policy is None:
            policy = capabilities["default_execution_policy"]
        if not isinstance(policy, str):
            raise TaichiRuntimeError("execution_policy must be a string")
        policy = policy.casefold()
        if (
            self.operator._provider_kind == "composition"
            and arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
            and not capabilities["execution_policies"].get(policy, False)
        ):
            raise TaichiRuntimeError(
                "GPU composed operators currently require the qualified "
                "device_convergent policy; no host-readback fallback was "
                "performed"
            )
        self.requested_execution_policy = policy
        self._native_execution_policy = policy
        self._require_native_device_convergent = False
        if policy != "bounded_convergent" and self.bounded_mode != "auto":
            raise TaichiRuntimeError(
                "bounded_mode is configurable only with "
                "execution_policy='bounded_convergent'"
            )
        if policy == "bounded_convergent":
            capabilities = self._execution_policy_capabilities()
            bounded = capabilities["bounded_convergent"]
            if not bounded["supported"]:
                raise TaichiRuntimeError(
                    "SolvePlan execution_policy='bounded_convergent' is "
                    "not qualified for this method/provider/dtype; no "
                    "fallback was performed"
                )
            if (
                self.bounded_mode == "native_required"
                and not bounded["native_upgrade_available"]
            ):
                raise TaichiRuntimeError(
                    "SolvePlan bounded_mode='native_required' is "
                    "unsupported; no fallback was performed: "
                    f"{bounded['native_upgrade_unavailable_reason']}"
                )
            if (
                arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                and self.bounded_mode != "portable"
                and (
                    (
                        arch == _ti_core.Arch.cuda
                        and self.operator._provider_kind == "stored"
                    )
                    or self.bounded_mode == "native_required"
                )
                and bounded["native_upgrade_available"]
            ):
                self._native_execution_policy = "device_convergent"
                self._require_native_device_convergent = (
                    self.bounded_mode == "native_required"
                )
        if policy == "device_convergent":
            capability = self._execution_policy_capabilities()[
                "device_convergent"
            ]
            if not capability["supported"]:
                raise TaichiRuntimeError(
                    "SolvePlan execution_policy='device_convergent' is "
                    "unsupported; no fallback was performed: "
                    f"{capability['unsupported_reason']}"
                )
            self._require_native_device_convergent = True
        if arch in cpu_arches:
            if policy not in ("host_each_iteration", "bounded_convergent"):
                raise TaichiRuntimeError(
                    "CPU SolvePlan supports host_each_iteration or "
                    "bounded_convergent"
                )
            expected_interval = 1
        elif arch == _ti_core.Arch.cuda:
            supported = (
                ("host_check_every_k",)
                if self.method in ("gmres", "fgmres")
                else (
                    "host_each_iteration",
                    "host_check_every_k",
                    "bounded_convergent",
                    "device_convergent",
                )
            )
            if policy not in supported:
                raise TaichiRuntimeError(
                    "CUDA GMRES/FGMRES supports host_check_every_k only"
                    if self.method in ("gmres", "fgmres")
                    else "CUDA SolvePlan supports host_each_iteration or "
                    "host_check_every_k"
                )
            expected_interval = (
                self.restart
                if self.method in ("gmres", "fgmres")
                else (
                    16
                    if policy
                    in (
                        "bounded_convergent",
                        "device_convergent",
                    )
                    else (4 if policy == "host_check_every_k" else 1)
                )
            )
        elif arch == _ti_core.Arch.vulkan:
            if policy not in (
                "fixed_budget_masked",
                "host_check_every_k",
                "bounded_convergent",
                "device_convergent",
            ):
                raise TaichiRuntimeError(
                    "Vulkan SolvePlan supports fixed_budget_masked or "
                    "host_check_every_k, bounded_convergent, or "
                    "device_convergent"
                )
            expected_interval = (
                self.restart
                if self.method in ("gmres", "fgmres")
                and policy == "host_check_every_k"
                else (
                    16
                    if policy in ("bounded_convergent", "device_convergent")
                    else (
                        4
                        if policy == "host_check_every_k"
                        else self.max_iterations
                    )
                )
            )
        else:
            raise TaichiRuntimeError("unsupported SolvePlan backend")
        if check_interval is None:
            check_interval = expected_interval
        if isinstance(check_interval, bool):
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            )
        try:
            check_interval = _operator.index(check_interval)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            ) from exc
        if check_interval <= 0:
            raise TaichiRuntimeError(
                "check_interval must be a positive integer"
            )
        if policy not in (
            "host_check_every_k",
            "bounded_convergent",
            "device_convergent",
        ) and (check_interval != expected_interval):
            raise TaichiRuntimeError(
                "check_interval is configurable only for "
                "host_check_every_k, bounded_convergent, or "
                "device_convergent"
            )
        if (
            self.method in ("gmres", "fgmres")
            and policy == "host_check_every_k"
            and check_interval != self.restart
        ):
            raise TaichiRuntimeError(
                "GMRES/FGMRES host_check_every_k requires "
                "check_interval == restart"
            )
        if (
            self.method not in ("gmres", "fgmres")
            and policy == "host_check_every_k"
            and check_interval not in (4, 8)
        ):
            raise TaichiRuntimeError(
                "host_check_every_k currently supports K=4 or K=8"
            )
        if policy in (
            "bounded_convergent",
            "device_convergent",
        ) and check_interval not in (
            1,
            2,
            4,
            8,
            16,
        ):
            raise TaichiRuntimeError(
                "bounded/device convergent check_interval is the portable "
                "fallback chunk limit "
                "and must be one of 1, 2, 4, 8, or 16"
            )
        return policy, check_interval

    def _configure_cuda_solver(self, solver):
        arguments = (
            self._native_execution_policy,
            self.check_interval,
        )
        if self.method in ("cg", "pcg"):
            arguments += (self._require_native_device_convergent,)
        solver._configure_execution_policy(*arguments)
        return solver

    def _configure_vulkan_solver(self, solver):
        if self.execution_policy in (
            "host_check_every_k",
            "bounded_convergent",
        ):
            solver._configure_execution_policy(
                self.execution_policy, self.check_interval
            )
        return solver

    def _require_spd(self):
        traits = self.operator._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        positive_definite = dict(traits["positive_definite"])
        singular = dict(traits["singular"])
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                "CG/PCG requires an explicit or structurally derived "
                "self_adjoint=True trait"
            )
        if (
            not positive_definite["known"]
            or positive_definite["value"] is not True
        ):
            raise TaichiRuntimeError(
                "CG/PCG requires an explicit or structurally derived "
                "positive_definite=True trait"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError("CG/PCG rejects singular operators")

    def _require_self_adjoint(self):
        traits = self.operator._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        singular = dict(traits["singular"])
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                "MINRES requires an explicit or structurally derived "
                "self_adjoint=True trait"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                "MINRES does not provide singular minimum-length semantics "
                "and rejects operators declared singular"
            )

    def _require_fixed_linear_preconditioner(self, preconditioner):
        preconditioner._ensure_valid()
        if preconditioner._program is not self._program:
            raise TaichiRuntimeError(
                "preconditioner must belong to the SolvePlan runtime"
            )
        expected_shape = (self.operator.shape[1], self.operator.shape[0])
        if preconditioner.shape != expected_shape:
            raise TaichiRuntimeError(
                "preconditioner must map the operator range back to its "
                "domain"
            )
        if preconditioner.dtype != self.operator.dtype:
            raise TaichiRuntimeError(
                "operator and preconditioner must have the same dtype"
            )
        traits = preconditioner._metadata_snapshot["traits"]
        self_adjoint = dict(traits["self_adjoint"])
        positive_definite = dict(traits["positive_definite"])
        singular = dict(traits["singular"])
        solver_name = self.method.upper()
        if not self_adjoint["known"] or self_adjoint["value"] is not True:
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} requires preconditioner "
                "self_adjoint=True"
            )
        if (
            not positive_definite["known"]
            or positive_definite["value"] is not True
        ):
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} requires preconditioner "
                "positive_definite=True"
            )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                f"fixed-linear {solver_name} rejects singular preconditioners"
            )

    def _require_fixed_linear_right_preconditioner(self, preconditioner):
        preconditioner._ensure_valid()
        if preconditioner._program is not self._program:
            raise TaichiRuntimeError(
                "preconditioner must belong to the SolvePlan runtime"
            )
        expected_shape = (self.operator.shape[1], self.operator.shape[0])
        if preconditioner.shape != expected_shape:
            raise TaichiRuntimeError(
                "right preconditioner must map the operator range back "
                "to its domain"
            )
        if preconditioner.dtype != self.operator.dtype:
            raise TaichiRuntimeError(
                "operator and preconditioner must have the same dtype"
            )
        singular = dict(
            preconditioner._metadata_snapshot["traits"]["singular"]
        )
        if singular["known"] and singular["value"] is True:
            raise TaichiRuntimeError(
                f"fixed-linear right {self.method.upper()} rejects singular "
                "preconditioners"
            )

    def _build_solver(self):
        arch = self._program.config().arch
        core = self.operator._provider_core
        kind = self.operator._provider_kind
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        gpu_arches = (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
        if self.operator.dtype == f64 and arch not in cpu_arches:
            raise TaichiRuntimeError("GPU SolvePlan currently requires f32")

        if self.method in ("cg", "pcg"):
            self._require_spd()

        if self.method == "minres":
            self._require_self_adjoint()
            if arch in cpu_arches:
                if self.preconditioner is not None:
                    raise TaichiRuntimeError(
                        "CPU experimental MINRES currently uses identity "
                        "preconditioning only"
                    )
                factory = (
                    _ti_core._make_float_cpu_experimental_minres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_experimental_minres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported MINRES backend")

            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if self.preconditioner is None:
                factory = (
                    _ti_core._make_device_fixed_sparse_minres_solver
                    if kind == "stored"
                    else _ti_core._make_device_experimental_minres_solver
                )
                arguments = [self._program, self.operator._handle]
                if kind == "stored":
                    arguments.append(core)
                arguments.extend([self.max_iterations, self.atol, self.rtol])
                return configure(factory(*arguments))

            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("MINRES")
                    preconditioner_scope = self.preconditioner.pin()
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_preconditioner(
                    preconditioner_action
                )
                return configure(
                    _ti_core._make_device_operator_preconditioned_minres_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )

            if not isinstance(self.preconditioner, str):
                raise TaichiRuntimeError(
                    "MINRES requires a fixed LinearOperator, "
                    "PreconditionerPlan, or "
                    "preconditioner='jacobi'/'block_jacobi'"
                )
            preconditioner = self.preconditioner.casefold()
            if preconditioner not in ("jacobi", "block_jacobi"):
                raise TaichiRuntimeError(
                    "MINRES requires preconditioner='jacobi' or "
                    "'block_jacobi'"
                )
            if kind != "stored":
                raise TaichiRuntimeError(
                    "built-in MINRES preconditioning requires a fixed "
                    "stored CSR/BSR provider"
                )
            contract = self.operator._source._get_format_contract()
            storage = contract["identity"]["storage_format"]
            required = "csr" if preconditioner == "jacobi" else "bsr"
            if storage != required:
                raise TaichiRuntimeError(
                    f"{preconditioner} MINRES requires "
                    f"{required.upper()} storage, got {storage.upper()}"
                )
            if preconditioner == "jacobi":
                self._native_preconditioner = (
                    _ti_core._make_sparse_jacobi_preconditioner_plan(
                        self._program, core
                    )
                )
                factory = _ti_core._make_device_jacobi_minres_solver
            else:
                self._native_preconditioner = (
                    _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                        self._program, core
                    )
                )
                factory = _ti_core._make_device_block_jacobi_minres_solver
            return configure(
                factory(
                    self._program,
                    self.operator._handle,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "fgmres":
            if not isinstance(self.preconditioner, PreconditionerPlan):
                raise TaichiRuntimeError(
                    "FGMRES requires PreconditionerPlan("
                    "behavior='variable_linear')"
                )
            self.preconditioner._require_target(self.operator)
            if self.preconditioner.behavior != "variable_linear":
                raise TaichiRuntimeError(
                    "FGMRES requires behavior='variable_linear'; use GMRES "
                    "for fixed-linear right preconditioning"
                )
            preconditioner_actions = self.preconditioner._consumer_actions
            if len(preconditioner_actions) != len(self.preconditioner.actions):
                raise TaichiRuntimeError(
                    "variable_linear PreconditionerPlan must be setup before "
                    "constructing FGMRES"
                )
            for action in preconditioner_actions:
                self._require_fixed_linear_right_preconditioner(action)
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or any(
                    action._provider_kind in ("kernel", "graph")
                    for action in preconditioner_actions
                ):
                    raise TaichiRuntimeError(
                        "CPU FGMRES does not consume compiled ndarray "
                        "operator actions"
                    )
                factory = (
                    _ti_core._make_float_cpu_variable_preconditioned_experimental_gmres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_variable_preconditioned_experimental_gmres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    [action._handle for action in preconditioner_actions],
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported FGMRES backend")
            if self.operator.dtype != f32:
                raise TaichiRuntimeError(
                    "GPU FGMRES currently supports f32 only"
                )
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            return configure(
                _ti_core._make_device_variable_preconditioned_gmres_solver(
                    self._program,
                    self.operator._handle,
                    [action._handle for action in preconditioner_actions],
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "gmres":
            preconditioner_action = None
            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("GMRES")
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_right_preconditioner(
                    preconditioner_action
                )
            elif self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "GMRES requires a fixed LinearOperator or "
                    "PreconditionerPlan right preconditioner"
                )
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or (
                    preconditioner_action is not None
                    and preconditioner_action._provider_kind
                    in ("kernel", "graph")
                ):
                    raise TaichiRuntimeError(
                        "CPU GMRES does not consume compiled ndarray "
                        "operator actions"
                    )
                if preconditioner_action is None:
                    factory = (
                        _ti_core._make_float_cpu_experimental_gmres_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_experimental_gmres_solver
                    )
                    return factory(
                        self._program,
                        self.operator._handle,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                factory = (
                    _ti_core._make_float_cpu_preconditioned_experimental_gmres_solver
                    if self.operator.dtype == f32
                    else _ti_core._make_double_cpu_preconditioned_experimental_gmres_solver
                )
                return factory(
                    self._program,
                    self.operator._handle,
                    preconditioner_action._handle,
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported GMRES backend")
            if self.operator.dtype != f32:
                raise TaichiRuntimeError(
                    "GPU GMRES currently supports f32 only"
                )
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if preconditioner_action is not None:
                return configure(
                    _ti_core._make_device_operator_preconditioned_gmres_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                )
            if kind == "stored":
                return configure(
                    _ti_core._make_device_fixed_sparse_gmres_solver(
                        self._program,
                        self.operator._handle,
                        core,
                        self.max_iterations,
                        self.restart,
                        self.atol,
                        self.rtol,
                    )
                )
            return configure(
                _ti_core._make_device_experimental_gmres_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.restart,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "bicgstab":
            preconditioner_action = None
            if isinstance(
                self.preconditioner, (LinearOperator, PreconditionerPlan)
            ):
                if isinstance(self.preconditioner, PreconditionerPlan):
                    self.preconditioner._require_target(self.operator)
                    self.preconditioner._require_fixed_behavior("BiCGSTAB")
                    preconditioner_scope = self.preconditioner.pin()
                    preconditioner_action = (
                        self.preconditioner._consumer_action
                    )
                else:
                    preconditioner_scope = None
                    preconditioner_action = self.preconditioner
                self._require_fixed_linear_right_preconditioner(
                    preconditioner_action
                )
            elif self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "BiCGSTAB requires a fixed LinearOperator or "
                    "PreconditionerPlan right preconditioner"
                )
            if arch in cpu_arches:
                if kind in ("kernel", "graph") or (
                    preconditioner_action is not None
                    and preconditioner_action._provider_kind
                    in ("kernel", "graph")
                ):
                    raise TaichiRuntimeError(
                        "CPU BiCGSTAB does not consume compiled ndarray "
                        "operator actions"
                    )
                factory = (
                    (
                        _ti_core._make_float_cpu_preconditioned_experimental_bicgstab_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_preconditioned_experimental_bicgstab_solver
                    )
                    if preconditioner_action is not None
                    else (
                        _ti_core._make_float_cpu_experimental_bicgstab_solver
                        if self.operator.dtype == f32
                        else _ti_core._make_double_cpu_experimental_bicgstab_solver
                    )
                )
                arguments = [self._program, self.operator._handle]
                if preconditioner_action is not None:
                    arguments.append(preconditioner_action._handle)
                arguments.extend([self.max_iterations, self.atol, self.rtol])
                return factory(*arguments)

            if arch not in gpu_arches:
                raise TaichiRuntimeError("unsupported BiCGSTAB backend")
            configure = (
                self._configure_cuda_solver
                if arch == _ti_core.Arch.cuda
                else self._configure_vulkan_solver
            )
            if preconditioner_action is not None:
                return configure(
                    _ti_core._make_device_operator_preconditioned_bicgstab_solver(
                        self._program,
                        self.operator._handle,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            if kind == "stored":
                return configure(
                    _ti_core._make_device_fixed_sparse_bicgstab_solver(
                        self._program,
                        self.operator._handle,
                        core,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            return configure(
                _ti_core._make_device_experimental_bicgstab_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )

        if self.method == "cg":
            if self.preconditioner is not None:
                raise TaichiRuntimeError(
                    "CG does not accept a preconditioner; use method='pcg'"
                )
            if arch in cpu_arches:
                return _ti_core._make_cpu_experimental_cg_solver(
                    self._program,
                    self.operator._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch == _ti_core.Arch.cuda:
                if (
                    self.operator._supports_graph_action()
                    and self._native_execution_policy == "device_convergent"
                ):
                    from taichi_forge.linalg._graph_krylov import (
                        GraphKrylovSolver,
                    )

                    return GraphKrylovSolver(
                        self.operator,
                        None,
                        max_iterations=self.max_iterations,
                        absolute_tolerance=self.atol,
                        relative_tolerance=self.rtol,
                    )
                if kind == "kernel":
                    factory = _ti_core._make_cuda_compiled_kernel_cg_solver
                elif kind == "graph":
                    factory = _ti_core._make_cuda_compiled_graph_cg_solver
                elif kind == "stored":
                    contract = self.operator._source._get_format_contract()
                    if contract["identity"]["storage_format"] != "csr":
                        raise TaichiRuntimeError(
                            "CUDA identity CG supports CSR, compiled-kernel, "
                            "and compiled-Graph providers"
                        )
                    return self._configure_cuda_solver(
                        _ti_core.make_cucg_solver(
                            core,
                            self.max_iterations,
                            self.atol,
                            False,
                            self.rtol,
                        )
                    )
                else:
                    raise TaichiRuntimeError(
                        "GPU SolvePlan does not lower composed operators"
                    )
                return self._configure_cuda_solver(
                    factory(
                        self._program,
                        core,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                if (
                    self.operator._supports_graph_action()
                    and self._native_execution_policy == "device_convergent"
                ):
                    from taichi_forge.linalg._graph_krylov import (
                        GraphKrylovSolver,
                    )

                    return GraphKrylovSolver(
                        self.operator,
                        None,
                        max_iterations=self.max_iterations,
                        absolute_tolerance=self.atol,
                        relative_tolerance=self.rtol,
                    )
                if kind == "kernel":
                    factory = (
                        _ti_core._make_vulkan_compiled_kernel_cg_convergence_plan
                    )
                elif kind == "graph":
                    factory = (
                        _ti_core._make_vulkan_compiled_graph_cg_convergence_plan
                    )
                elif kind == "stored":
                    factory = _ti_core._make_vulkan_cg_convergence_plan
                else:
                    raise TaichiRuntimeError(
                        "GPU SolvePlan does not lower composed operators"
                    )
                return self._configure_vulkan_solver(
                    factory(
                        self._program,
                        core,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported SolvePlan backend")

        if isinstance(
            self.preconditioner, (LinearOperator, PreconditionerPlan)
        ):
            if isinstance(self.preconditioner, PreconditionerPlan):
                self.preconditioner._require_target(self.operator)
                self.preconditioner._require_fixed_behavior("PCG")
                # Validate and pin the exact approved pair across native
                # solver construction. The solver owns its own generation
                # pins after the factory returns.
                preconditioner_scope = self.preconditioner.pin()
                preconditioner_action = self.preconditioner._consumer_action
            else:
                preconditioner_scope = None
                preconditioner_action = self.preconditioner
            self._require_fixed_linear_preconditioner(preconditioner_action)
            if (
                arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
                and self.operator._supports_graph_action()
                and preconditioner_action._supports_graph_action()
                and self._native_execution_policy == "device_convergent"
            ):
                from taichi_forge.linalg._graph_krylov import (
                    GraphKrylovSolver,
                )

                return GraphKrylovSolver(
                    self.operator,
                    preconditioner_action,
                    max_iterations=self.max_iterations,
                    absolute_tolerance=self.atol,
                    relative_tolerance=self.rtol,
                )
            if arch in cpu_arches:
                factory = _ti_core._make_cpu_experimental_pcg_solver
                return factory(
                    self._program,
                    self.operator._handle,
                    preconditioner_action._handle,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if (
                kind != "kernel"
                or preconditioner_action._provider_kind != "kernel"
            ):
                raise TaichiRuntimeError(
                    "GPU fixed-linear PCG requires recordable A and M providers "
                    "with execution_policy='device_convergent', or legacy "
                    "compiled-kernel A and M providers"
                )
            if arch == _ti_core.Arch.cuda:
                factory = _ti_core._make_cuda_experimental_pcg_solver
                return self._configure_cuda_solver(
                    factory(
                        self._program,
                        core,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                factory = (
                    _ti_core._make_vulkan_experimental_pcg_convergence_plan
                )
                return self._configure_vulkan_solver(
                    factory(
                        self._program,
                        core,
                        preconditioner_action._handle,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported PCG backend")

        if not isinstance(self.preconditioner, str):
            raise TaichiRuntimeError(
                "PCG requires a fixed LinearOperator, PreconditionerPlan, or "
                "preconditioner='jacobi'/'block_jacobi'"
            )
        preconditioner = self.preconditioner.casefold()
        if preconditioner not in ("jacobi", "block_jacobi"):
            raise TaichiRuntimeError(
                "PCG requires preconditioner='jacobi' or 'block_jacobi'"
            )
        if kind != "stored":
            raise TaichiRuntimeError(
                "experimental PCG supports fixed stored CSR/BSR providers only"
            )
        contract = self.operator._source._get_format_contract()
        storage = contract["identity"]["storage_format"]
        required_storage = "csr" if preconditioner == "jacobi" else "bsr"
        if storage != required_storage:
            raise TaichiRuntimeError(
                f"{preconditioner} PCG requires {required_storage.upper()} "
                f"storage, got {storage.upper()}"
            )
        if preconditioner == "jacobi":
            self._native_preconditioner = (
                _ti_core._make_sparse_jacobi_preconditioner_plan(
                    self._program, core
                )
            )
            if arch in cpu_arches:
                return _ti_core._make_cpu_jacobi_pcg_solver(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            if arch == _ti_core.Arch.cuda:
                return self._configure_cuda_solver(
                    _ti_core._make_cuda_jacobi_pcg_solver(
                        self._program,
                        core,
                        self._native_preconditioner,
                        self.max_iterations,
                        self.atol,
                        False,
                        self.rtol,
                    )
                )
            if arch == _ti_core.Arch.vulkan:
                return self._configure_vulkan_solver(
                    _ti_core._make_vulkan_jacobi_pcg_convergence_plan(
                        self._program,
                        core,
                        self._native_preconditioner,
                        self.max_iterations,
                        self.atol,
                        self.rtol,
                    )
                )
            raise TaichiRuntimeError("unsupported PCG backend")

        self._native_preconditioner = (
            _ti_core._make_sparse_block_jacobi_preconditioner_plan(
                self._program, core
            )
        )
        if arch in cpu_arches:
            return _ti_core._make_cpu_block_jacobi_pcg_solver(
                self._program,
                core,
                self._native_preconditioner,
                self.max_iterations,
                self.atol,
                self.rtol,
            )
        if arch == _ti_core.Arch.cuda:
            return self._configure_cuda_solver(
                _ti_core._make_cuda_block_jacobi_pcg_solver(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    False,
                    self.rtol,
                )
            )
        if arch == _ti_core.Arch.vulkan:
            return self._configure_vulkan_solver(
                _ti_core._make_vulkan_block_jacobi_pcg_convergence_plan(
                    self._program,
                    core,
                    self._native_preconditioner,
                    self.max_iterations,
                    self.atol,
                    self.rtol,
                )
            )
        raise TaichiRuntimeError("unsupported PCG backend")

    def _prepare_graph_krylov_input_operand(self, value, role, size, staging_role):
        direct = None
        if self._graph_krylov_direct_field_enabled:
            direct = _try_direct_graph_krylov_operand(
                value,
                role,
                size,
                self.operator.dtype,
                self._vector_io,
                self._graph_krylov_binding_cache,
            )
        if direct is not None:
            return direct
        return _graph_krylov_staged_operand(
            _prepare_vector_input(
                value,
                role,
                size,
                self.operator.dtype,
                self._vector_io,
                staging_role,
            )
        )

    def _prepare_graph_krylov_output_operand(self, value, role, size, staging_role):
        direct = None
        if self._graph_krylov_direct_field_enabled:
            direct = _try_direct_graph_krylov_operand(
                value,
                role,
                size,
                self.operator.dtype,
                self._vector_io,
                self._graph_krylov_binding_cache,
            )
        if direct is not None:
            return direct
        return _graph_krylov_staged_operand(
            _prepare_vector_output(
                value,
                role,
                size,
                self.operator.dtype,
                self._vector_io,
                staging_role,
            )
        )

    def _prepare_graph_krylov_operands(self, rhs, initial_guess, out, size):
        rhs_operand = self._prepare_graph_krylov_input_operand(
            rhs,
            "SolvePlan RHS",
            size,
            "solve_rhs",
        )
        if out is None:
            out = ScalarNdarray(self.operator.dtype, (size,))
        output_operand = self._prepare_graph_krylov_output_operand(
            out,
            "SolvePlan output",
            size,
            "solve_output",
        )
        if _graph_krylov_operands_overlap(rhs_operand, output_operand):
            raise TaichiRuntimeError("SolvePlan RHS and output may not alias")

        initial_operand = None
        if initial_guess is not None:
            output_view = self._vector_io.view(out, "SolvePlan output")
            initial_view = self._vector_io.view(
                initial_guess, "SolvePlan initial_guess"
            )
            shares_output = (
                initial_view is not None
                and output_view is not None
                and initial_view._exact_view_key == output_view._exact_view_key
            )
            initial_operand = self._prepare_graph_krylov_input_operand(
                (initial_view if initial_view is not None else initial_guess),
                "SolvePlan initial_guess",
                size,
                "solve_output" if shares_output else "solve_initial",
            )
            if _graph_krylov_operands_overlap(initial_operand, rhs_operand):
                raise TaichiRuntimeError(
                    "SolvePlan RHS and initial_guess may not alias"
                )
            if (
                _graph_krylov_operands_overlap(initial_operand, output_operand)
                and not _graph_krylov_operands_exact(initial_operand, output_operand)
            ):
                raise TaichiRuntimeError(
                    "SolvePlan initial_guess and output overlap without "
                    "being the same vector view"
                )
        return out, rhs_operand, output_operand, initial_operand

    def graph_action(
        self,
        rhs,
        output,
        *,
        initial_guess=None,
        name=None,
    ):
        """Record this complete solve into an enclosing Graph.

        The returned native node owns symbolic device terminal resources via
        ``node.terminal``. Allocate one runtime packet with
        ``node.allocate_terminal()`` and include ``packet.arguments`` in the
        enclosing Graph arguments. No terminal state is read back before the
        caller waits for the enclosing ``SubmissionTicket``.
        """

        if self.operator is None or self._program is not _current_program():
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        if self.method not in ("cg", "pcg"):
            raise TaichiRuntimeError(
                "SolvePlan Graph actions currently support CG and PCG"
            )
        if self.operator.dtype != f32:
            raise TaichiRuntimeError(
                "SolvePlan Graph actions currently require f32"
            )
        if not self.operator._supports_graph_action():
            raise TaichiRuntimeError(
                "SolvePlan Graph action requires a recordable operator"
            )
        preconditioner = None
        if self.method == "pcg":
            if isinstance(self.preconditioner, LinearOperator):
                preconditioner = self.preconditioner
            elif isinstance(self.preconditioner, PreconditionerPlan):
                self.preconditioner._require_target(self.operator)
                self.preconditioner._require_fixed_behavior("PCG Graph action")
                preconditioner = self.preconditioner._consumer_action
            else:
                raise TaichiRuntimeError(
                    "PCG Graph action requires a fixed recordable "
                    "LinearOperator or PreconditionerPlan"
                )
            self._require_fixed_linear_preconditioner(preconditioner)
            if not preconditioner._supports_graph_action():
                raise TaichiRuntimeError(
                    "PCG Graph action requires a recordable preconditioner"
                )
        if name is None:
            name = f"{self.method}_solve_{self._graph_action_serial}"
        if not isinstance(name, str) or not name:
            raise TaichiRuntimeError(
                "SolvePlan Graph action name must be a nonempty string"
            )
        from taichi_forge.linalg._graph_krylov import GraphKrylovSolver

        if self._graph_action_solver is None:
            if isinstance(self._solver, GraphKrylovSolver):
                self._graph_action_solver = self._solver
            else:
                self._graph_action_solver = GraphKrylovSolver(
                    self.operator,
                    preconditioner,
                    max_iterations=self.max_iterations,
                    absolute_tolerance=self.atol,
                    relative_tolerance=self.rtol,
                    recordable_only=True,
                )
        serial = self._graph_action_serial
        self._graph_action_serial += 1
        action_record = _SolvePlanGraphActionRecord(
            serial=serial,
            name=name,
            method=self.method,
        )
        self._graph_action_records.append(action_record)
        terminal_prefix = f"__solve_terminal_{id(self):x}_{serial}"
        terminal = SolveGraphTerminal(
            f"{terminal_prefix}_state",
            f"{terminal_prefix}_metrics",
            action_record=action_record,
        )
        return _SolvePlanGraphNode(
            self,
            self._graph_action_solver,
            rhs,
            output,
            initial_guess,
            terminal,
            action_record,
            name=name,
        )

    def _submission_graph(self, with_initial_guess):
        if self.operator is None or self._submission_graphs is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        capability = self._submission_capability()
        if not capability["qualified"]:
            raise TaichiRuntimeError(
                "SolvePlan.submit is unsupported; no fallback was performed: "
                f"{capability['unsupported_reason']}"
            )
        key = bool(with_initial_guess)
        with self._submission_lock:
            cached = self._submission_graphs.get(key)
            if cached is not None:
                return cached

            from taichi_forge.graph._graph import Arg, ArgKind, GraphBuilder

            variant = "initial" if key else "zero"
            prefix = f"__solve_submit_{id(self):x}_{variant}"
            rhs_arg = Arg(ArgKind.NDARRAY, f"{prefix}_rhs", f32, ndim=1)
            output_arg = Arg(
                ArgKind.NDARRAY, f"{prefix}_output", f32, ndim=1
            )
            initial_arg = (
                Arg(
                    ArgKind.NDARRAY,
                    f"{prefix}_initial",
                    f32,
                    ndim=1,
                )
                if key
                else None
            )
            action = self.graph_action(
                rhs_arg,
                output_arg,
                initial_guess=initial_arg,
                name=f"{self.method}_submit_{variant}",
            )
            builder = GraphBuilder()
            builder.append_native(action)
            graph = builder.compile(
                workspace_lanes=self.submission_workspace_lanes,
                workspace_saturation=(
                    self.submission_workspace_saturation
                ),
            )
            cached = {
                "graph": graph,
                "action": action,
                "rhs_name": rhs_arg.name,
                "output_name": output_arg.name,
                "initial_name": (
                    None if initial_arg is None else initial_arg.name
                ),
            }
            self._submission_graphs[key] = cached
            self._submission_graph_builds += 1
            return cached

    def _submission_capability(self):
        arch = self._program.config().arch
        if arch in (_ti_core.Arch.x64, _ti_core.Arch.arm64):
            return {
                "qualified": True,
                "asynchronous": False,
                "unsupported_reason": "none",
            }
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan):
            return {
                "qualified": False,
                "asynchronous": False,
                "unsupported_reason": "unsupported_backend",
            }
        if self.method not in ("cg", "pcg") or self.operator.dtype != f32:
            return {
                "qualified": False,
                "asynchronous": False,
                "unsupported_reason": "recordable_f32_cg_pcg_required",
            }
        if self._native_execution_policy != "device_convergent":
            return {
                "qualified": False,
                "asynchronous": False,
                "unsupported_reason": "device_convergent_policy_required",
            }
        if not self.operator._supports_graph_action():
            return {
                "qualified": False,
                "asynchronous": False,
                "unsupported_reason": "operator_action_not_recordable",
            }
        if self.method == "pcg":
            if isinstance(self.preconditioner, LinearOperator):
                preconditioner = self.preconditioner
            elif isinstance(self.preconditioner, PreconditionerPlan):
                preconditioner = self.preconditioner._consumer_action
            else:
                preconditioner = None
            if (
                preconditioner is None
                or not preconditioner._supports_graph_action()
            ):
                return {
                    "qualified": False,
                    "asynchronous": False,
                    "unsupported_reason": (
                        "preconditioner_action_not_recordable"
                    ),
                }
        return {
            "qualified": True,
            "asynchronous": True,
            "unsupported_reason": "none",
        }

    def submit(
        self,
        rhs,
        *,
        initial_guess=None,
        out=None,
        pacer=None,
        lane=None,
        on_saturation="wait",
        telemetry=False,
        workspace_lane=None,
    ):
        """Submit one complete solve and return one ticket-shaped wrapper.

        CUDA/Vulkan record :meth:`graph_action` into one lazily cached Graph
        and delegate admission, workspace-lane ownership, completion, and
        telemetry to :meth:`Graph.submit`. No terminal state is read on that
        path; ``result()`` performs the one terminal-packet materialization.
        ``telemetry="summary"`` omits backend timestamp markers;
        ``telemetry="timestamps"`` and compatibility ``True`` request them.
        CPU uses the existing synchronous native solve and returns an already-
        complete lane-0 wrapper without Graph telemetry.
        """

        telemetry = _normalize_submission_telemetry_mode(
            telemetry, "SolvePlan.submit()"
        )

        with self._submission_lock:
            self._submission_calls += 1
            if telemetry:
                self._submission_telemetry_requests += 1
        try:
            if self._program.config().arch in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
            ):
                if workspace_lane not in (None, 0):
                    raise TaichiRuntimeError(
                        "CPU SolvePlan.submit completes synchronously and "
                        "supports workspace_lane 0 only"
                    )
                result = self.solve(
                    rhs, initial_guess=initial_guess, out=out
                )
                submission = SolvePlanSubmission(
                    self,
                    None,
                    _CompletedSolvePlanTicket(),
                    None,
                    rhs,
                    result.solution,
                    initial_guess,
                    completed_result=result,
                )
                with self._submission_lock:
                    self._submission_successes += 1
                    self._submission_native_completed_results += 1
                return submission
            cached = self._submission_graph(initial_guess is not None)
            if out is None:
                out = ScalarNdarray(
                    self.operator.dtype, (self.operator.shape[0],)
                )
            packet = cached["action"].allocate_terminal(initialize=False)
            arguments = {
                cached["rhs_name"]: rhs,
                cached["output_name"]: out,
                **packet.arguments,
            }
            if initial_guess is not None:
                arguments[cached["initial_name"]] = initial_guess
            graph = cached["graph"]
            ticket = graph.submit(
                arguments,
                pacer=pacer,
                lane=lane,
                on_saturation=on_saturation,
                telemetry=telemetry,
                workspace_lane=workspace_lane,
            )
            packet._attach_submission(ticket)
            submission = SolvePlanSubmission(
                self,
                graph,
                ticket,
                packet,
                rhs,
                out,
                initial_guess,
            )
            if ticket._has_backend_work:
                get_runtime().transfer_runtime_submission_owner(
                    ticket._completion, submission
                )
        except BaseException:
            with self._submission_lock:
                self._submission_failures += 1
            raise
        with self._submission_lock:
            self._submission_successes += 1
        return submission

    def prepare_telemetry(self, mode, *, with_initial_guess=False):
        """Prepare one cached GPU submission variant for telemetry sampling.

        The call builds the selected SolvePlan Graph variant and delegates its
        allocation/JIT/timestamp preparation to ``Graph.prepare_telemetry``.
        It does not submit a solve or allocate a caller output/terminal packet.
        """

        mode = _normalize_submission_telemetry_mode(
            mode, "SolvePlan.prepare_telemetry()"
        )
        if mode is False:
            return self
        if not isinstance(with_initial_guess, bool):
            raise TaichiRuntimeError(
                "SolvePlan.prepare_telemetry() with_initial_guess must be a bool"
            )
        if self._program is None or self._program.config().arch not in (
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
        ):
            raise TaichiRuntimeError(
                "SolvePlan.prepare_telemetry() requires a CUDA or Vulkan "
                "device-convergent submission plan"
            )
        cached = self._submission_graph(with_initial_guess)
        cached["graph"].prepare_telemetry(mode)
        return self

    def solve(self, rhs, *, initial_guess=None, out=None):
        """Solves one RHS with persistent plan-owned workspace."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        if self._program is not _current_program():
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        size = self.operator.shape[0]
        graph_rhs_operand = None
        graph_output_operand = None
        graph_initial_operand = None
        if self._uses_graph_krylov:
            (
                out,
                graph_rhs_operand,
                graph_output_operand,
                graph_initial_operand,
            ) = self._prepare_graph_krylov_operands(
                rhs,
                initial_guess,
                out,
                size,
            )
            self._graph_krylov_last_direct_field_boundary = bool(
                graph_rhs_operand.direct_dense
                and graph_output_operand.direct_dense
            )
        else:
            self._graph_krylov_last_direct_field_boundary = False
            rhs_operand = _prepare_vector_input(
                rhs,
                "SolvePlan RHS",
                size,
                self.operator.dtype,
                self._vector_io,
                "solve_rhs",
            )
            if out is None:
                out = ScalarNdarray(self.operator.dtype, (size,))
            output_operand = _prepare_vector_output(
                out,
                "SolvePlan output",
                size,
                self.operator.dtype,
                self._vector_io,
                "solve_output",
            )
            if _vector_operands_overlap(rhs_operand, output_operand):
                raise TaichiRuntimeError("SolvePlan RHS and output may not alias")
            if initial_guess is None:
                output_operand.array.fill(0)
            else:
                initial_view = self._vector_io.view(
                    initial_guess, "SolvePlan initial_guess"
                )
                shares_output = (
                    initial_view is not None
                    and output_operand.view is not None
                    and initial_view._exact_view_key
                    == output_operand.view._exact_view_key
                )
                initial_operand = _prepare_vector_input(
                    (initial_view if initial_view is not None else initial_guess),
                    "SolvePlan initial_guess",
                    size,
                    self.operator.dtype,
                    self._vector_io,
                    "solve_output" if shares_output else "solve_initial",
                )
                if _vector_operands_overlap(initial_operand, rhs_operand):
                    raise TaichiRuntimeError(
                        "SolvePlan RHS and initial_guess may not alias"
                    )
                if _vector_operands_overlap(
                    initial_operand, output_operand
                ) and not _vector_operands_exact(initial_operand, output_operand):
                    raise TaichiRuntimeError(
                        "SolvePlan initial_guess and output overlap without "
                        "being the same vector view"
                    )
                if not _vector_operands_exact(initial_operand, output_operand):
                    output_operand.array.copy_from(initial_operand.array)
        if self._native_preconditioner is not None:
            preconditioner_stats = dict(
                self._native_preconditioner._debug_runtime_stats()
            )
            identity = dict(preconditioner_stats["identity"])
            if identity["operator_stale"]:
                self._native_preconditioner._refresh_numeric(self._program)
        preconditioner_scope = None
        if isinstance(self.preconditioner, PreconditionerPlan):
            self.preconditioner._require_target(self.operator)
            preconditioner_scope = self.preconditioner.pin()
        cpu_arches = (_ti_core.Arch.x64, _ti_core.Arch.arm64)
        if self._uses_graph_krylov:
            self._solver.solve_arrays(
                graph_output_operand.runtime_value,
                graph_rhs_operand.runtime_value,
                (
                    None
                    if graph_initial_operand is None
                    else graph_initial_operand.runtime_value
                ),
                direct_output=graph_output_operand.direct_dense,
            )
            self._vector_io.record_direct_graph_solve(
                graph_rhs_operand,
                graph_output_operand,
                graph_initial_operand,
            )
        elif (
            self.method in ("bicgstab", "gmres", "fgmres", "minres")
            and self._program.config().arch in cpu_arches
        ):
            self._solver.solve_ndarray(
                self._program,
                output_operand.array.arr,
                rhs_operand.array.arr,
            )
        else:
            self._solver.solve(
                self._program,
                output_operand.array.arr,
                rhs_operand.array.arr,
            )
        if self._uses_graph_krylov:
            if graph_output_operand.staged is not None:
                _finish_vector_output(
                    self._vector_io,
                    graph_output_operand.staged,
                    self.operator.dtype,
                    size,
                )
        else:
            _finish_vector_output(
                self._vector_io,
                output_operand,
                self.operator.dtype,
                size,
            )
        snapshot = dict(self._solver._get_last_result())
        return SolveResult(solution=out, **snapshot)

    def submission_statistics(self):
        """Return ownership, lane, memory, and materialization counters."""

        if self.operator is None or self._submission_graphs is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        with self._submission_lock:
            graphs = tuple(
                ("with_initial_guess" if key else "zero_initial_guess", value)
                for key, value in self._submission_graphs.items()
            )
            result = {
                "execution_path": (
                    "native_cpu_completed"
                    if self._program.config().arch
                    in (_ti_core.Arch.x64, _ti_core.Arch.arm64)
                    else "cached_graph_submission"
                ),
                "configured_workspace_lane_capacity": (
                    self.submission_workspace_lanes
                ),
                "workspace_lane_capacity": (
                    1
                    if self._program.config().arch
                    in (_ti_core.Arch.x64, _ti_core.Arch.arm64)
                    else self.submission_workspace_lanes
                ),
                "workspace_saturation_policy": (
                    self.submission_workspace_saturation
                ),
                "graphs_materialized": self._submission_graph_builds,
                "submit_calls": self._submission_calls,
                "submit_successes": self._submission_successes,
                "submit_failures": self._submission_failures,
                "telemetry_requests": self._submission_telemetry_requests,
                "terminal_materializations": (
                    self._submission_terminal_materializations
                ),
                "native_completed_results": (
                    self._submission_native_completed_results
                ),
            }
            result.update(self._submission_capability())
        persistent_bytes = 0
        transient_bytes = 0
        lanes_materialized = 0
        lanes_busy = 0
        variants = {}
        for name, cached in graphs:
            report = cached["graph"].execution_stats()
            memory = report.memory
            persistent_bytes += memory.persistent_internal_storage_bytes
            transient_bytes += memory.transient_temporary_bytes
            lanes_materialized += memory.workspace_lanes_materialized
            lanes_busy += memory.workspace_lanes_busy
            variants[name] = {
                "persistent_internal_storage_bytes": (
                    memory.persistent_internal_storage_bytes
                ),
                "transient_temporary_bytes": (
                    memory.transient_temporary_bytes
                ),
                "workspace_lanes_materialized": (
                    memory.workspace_lanes_materialized
                ),
                "workspace_lanes_busy": memory.workspace_lanes_busy,
            }
        result.update(
            {
                "persistent_internal_storage_bytes": persistent_bytes,
                "transient_temporary_bytes": transient_bytes,
                "workspace_lanes_materialized": lanes_materialized,
                "workspace_lanes_busy": lanes_busy,
                "variants": variants,
            }
        )
        return result

    def graph_action_statistics(self):
        """Return telemetry owned by complete recorded SolvePlan actions.

        Submission counters describe enclosing Graph invocations. Completion
        counters advance only when a ticket completion is observed, and
        terminal iteration counters advance only at explicit packet snapshot
        boundaries. No terminal readback is performed by this method.
        """

        if self.operator is None or self._graph_action_records is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        actions = tuple(
            record.statistics() for record in self._graph_action_records
        )
        return {
            "schema_version": 1,
            "actions_created": len(actions),
            "compiled_executables": sum(
                item["compiled_executables"] for item in actions
            ),
            "enclosing_graph_submissions": sum(
                item["enclosing_graph_submissions"] for item in actions
            ),
            "observed_completions": sum(
                item["observed_completions"] for item in actions
            ),
            "terminal_snapshots": sum(
                item["terminal_snapshots"] for item in actions
            ),
            "terminal_iteration_sum": sum(
                item["terminal_iteration_sum"] for item in actions
            ),
            "completion_semantics": "ticket_observed_without_terminal_readback",
            "terminal_scope": "last_action_invocation_in_packet",
            "actions": actions,
        }

    def statistics(self):
        """Returns backend-neutral plan resource and operation telemetry."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        result = dict(self._solver._debug_runtime_stats())
        identity = result["identity"]
        identity["requested_solver_execution_policy"] = (
            self.requested_execution_policy
        )
        identity["bounded_mode"] = self.bounded_mode
        if self.execution_policy == "bounded_convergent":
            native_used = (
                identity["solver_execution_policy"] == "device_convergent"
            )
            identity["bounded_native_upgrade_used"] = native_used
            bounded = self.execution_capabilities()["bounded_convergent"]
            if native_used:
                native_reason = "none"
            elif self.bounded_mode == "portable":
                native_reason = "portable_mode_selected"
            elif bounded["native_upgrade_available"]:
                native_reason = identity["solver_replay_unavailable_reason"]
            else:
                native_reason = bounded["native_upgrade_unavailable_reason"]
            identity["bounded_native_upgrade_unavailable_reason"] = (
                native_reason
            )
            identity["bounded_internal_execution_policy"] = (
                self._native_execution_policy
            )
            if self._program.config().arch in (
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
            ):
                identity["solver_execution_policy"] = "host_each_iteration"
                identity["bounded_chunk_limit"] = 1
                identity["bounded_control_path"] = "native_cpu_solver_loop"
                identity["bounded_chunk_schedule"] = "every_iteration"
        else:
            identity["bounded_native_upgrade_used"] = False
            identity["bounded_native_upgrade_unavailable_reason"] = (
                "not_requested"
            )
            identity["bounded_internal_execution_policy"] = (
                self._native_execution_policy
            )
        if isinstance(self.preconditioner, PreconditionerPlan):
            result["preconditioner_lifecycle"] = (
                self.preconditioner.statistics()
            )
        result["default_solution_binding"] = {
            "enabled": False,
            "workspace_allocated": False,
            "workspace_builds": 0,
            "workspace_reuses": 0,
            "result_copies": 0,
            "return_ownership": "independent_result",
            "disabled_reason": "independent_result_requires_allocation",
            "fast_path": "pass_explicit_out",
        }
        result["vector_io"] = self._vector_io.statistics()
        result["submission"] = self.submission_statistics()
        result["graph_actions"] = self.graph_action_statistics()
        result["execution_capabilities"] = self.execution_capabilities()
        return result

    def execution_capabilities(self):
        """Returns qualified execution policies and explicit failure reasons."""
        if self.operator is None or self._solver is None:
            raise TaichiRuntimeError(
                "SolvePlan cannot be used after ti.reset()"
            )
        self.operator._ensure_valid()
        result = self._execution_policy_capabilities()
        result["vector_io"] = _vector_io_capabilities()
        result["direct_dense_field_solve"] = {
            "supported": bool(self._uses_graph_krylov),
            "enabled": bool(
                self._uses_graph_krylov
                and self._graph_krylov_direct_field_enabled
            ),
            "selected": bool(
                self._uses_graph_krylov
                and self._graph_krylov_last_direct_field_boundary
            ),
            "primitive": (
                "graph_fused_runtime_storage_boundary"
                if self._uses_graph_krylov
                else "device_staged"
            ),
            "qualified_methods": ("cg", "pcg"),
            "qualified_dtypes": ("f32",),
            "qualified_layouts": (
                "root_dense_scalar_contiguous",
                "root_dense_packed_vector_matrix_contiguous",
            ),
            "initialization": (
                "graph_preamble"
                if self._uses_graph_krylov
                else "boundary_copy_or_fill"
            ),
            "iterative_storage": (
                "plan_owned_ndarray"
                if self._uses_graph_krylov
                else "backend_solver_owned"
            ),
            "boundary_copy": (
                "graph_preamble_epilogue"
                if self._uses_graph_krylov
                else "separate_transfer_submission"
            ),
            "unsupported_layout_fallback": "device_staged",
            "value_host_transfer": False,
        }
        return result


# Imported last because the batched implementation deliberately reuses the
# validated LinearOperator helpers above without extending OperatorSpaceDesc.
from taichi_forge.linalg._batched_solver import (  # noqa: E402
    BatchedSolvePlan,
    BatchedSolveResult,
    BatchedSolveWorkspacePool,
    BatchedSubmissionTelemetry,
    SolveSubmission,
)
from taichi_forge.linalg._solve_qualification import (  # noqa: E402
    SolveQualificationReport,
    qualify_solve_plan,
    summarize_solve_qualifications,
)


__all__ = [
    "BatchedSolvePlan",
    "BatchedSolveResult",
    "BatchedSolveWorkspacePool",
    "BatchedSubmissionTelemetry",
    "LinearOperator",
    "OperatorCapabilities",
    "OperatorTraits",
    "SmallBlockInverseBuilder",
    "SmallBlockInverseResult",
    "PreconditionerPlan",
    "PreconditionerSession",
    "SolveGraphTerminal",
    "SolveGraphTerminalPacket",
    "SolveGraphTerminalSnapshot",
    "SolvePlan",
    "SolvePlanSubmission",
    "SolveQualificationReport",
    "SolveResult",
    "SolveSubmission",
    "VectorView",
    "qualify_solve_plan",
    "summarize_operator_qualifications",
    "summarize_solve_qualifications",
    "aslinearoperator",
    "block_diagonal",
    "identity",
    "inverse_block_diagonal",
    "vector_io_capabilities",
    "vector_view",
]
