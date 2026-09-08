"""Cold, provider-owned matmul configuration discovery and reconstruction.

Opaque algorithm bytes are process-local execution data. Only documented
configuration attributes and observed workspace become persistent facts.
"""

import ctypes
from dataclasses import dataclass

from taichi_forge.hardware._cublaslt import (
    CublasLtMatmulPlan,
    _HeuristicResult,
    _MatmulAlgo,
    _set_attribute,
)
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import u8


# cuBLASLt public AlgoConfig ABI. The two shape attributes use uint16, not
# the uint32 representation used by most of the earlier attributes.
_CONFIG = (
    ("id", ctypes.c_int32),
    ("tile", ctypes.c_uint32),
    ("split_k", ctypes.c_int32),
    ("reduction", ctypes.c_uint32),
    ("swizzle", ctypes.c_uint32),
    ("custom", ctypes.c_uint32),
    ("stages", ctypes.c_uint32),
    ("inner_shape", ctypes.c_uint16),
    ("cluster_shape", ctypes.c_uint16),
)


@dataclass(frozen=True)
class _AlgorithmChoice:
    configuration: tuple
    workspace_bytes: int

    def __post_init__(self):
        values = tuple(self.configuration)
        if len(values) != len(_CONFIG):
            raise ValueError("Incomplete cuBLASLt algorithm configuration")
        for index, (value, (_, dtype)) in enumerate(zip(values, _CONFIG)):
            if value is None and index >= 7:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or dtype(value).value != value
            ):
                raise ValueError("Invalid cuBLASLt algorithm configuration value")
        if (
            isinstance(self.workspace_bytes, bool)
            or not isinstance(self.workspace_bytes, int)
            or self.workspace_bytes < 0
        ):
            raise ValueError("Invalid cuBLASLt algorithm workspace")
        object.__setattr__(self, "configuration", values)

    def to_dict(self):
        return {
            "configuration": {
                name: value for (name, _), value in zip(_CONFIG, self.configuration)
            },
            "workspace_bytes": self.workspace_bytes,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {
            "configuration",
            "workspace_bytes",
        }:
            raise ValueError("Invalid cuBLASLt algorithm facts")
        values = data["configuration"]
        if not isinstance(values, dict) or set(values) != {name for name, _ in _CONFIG}:
            raise ValueError("Incomplete cuBLASLt algorithm configuration")
        return cls(tuple(values[name] for name, _ in _CONFIG), data["workspace_bytes"])


class _AlgorithmApi:
    def __init__(self, library):
        self.library = library
        pointer = ctypes.c_void_p
        integer = ctypes.c_int
        size = ctypes.c_size_t
        algo = ctypes.POINTER(_MatmulAlgo)
        signatures = {
            "init": (
                "cublasLtMatmulAlgoInit",
                (
                    pointer,
                    integer,
                    integer,
                    integer,
                    integer,
                    integer,
                    integer,
                    integer,
                    algo,
                ),
            ),
            "get": (
                "cublasLtMatmulAlgoConfigGetAttribute",
                (algo, integer, pointer, size, ctypes.POINTER(size)),
            ),
            "set": (
                "cublasLtMatmulAlgoConfigSetAttribute",
                (algo, integer, pointer, size),
            ),
            "check": (
                "cublasLtMatmulAlgoCheck",
                (
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    algo,
                    ctypes.POINTER(_HeuristicResult),
                ),
            ),
        }
        for name, (symbol, arguments) in signatures.items():
            function = getattr(library.library, symbol, None)
            if function is None:
                raise TaichiRuntimeError(
                    f"cuBLASLt recipe preparation requires {symbol}"
                )
            function.argtypes, function.restype = arguments, integer
            setattr(self, name, function)

    def describe(self, algorithm, workspace_bytes):
        values = []
        for attribute, (_, dtype) in enumerate(_CONFIG):
            value, written = dtype(), ctypes.c_size_t()
            status = self.get(
                ctypes.byref(algorithm),
                attribute,
                ctypes.byref(value),
                ctypes.sizeof(value),
                ctypes.byref(written),
            )
            if attribute >= 7 and status in (
                7,
                15,
            ):  # Older vendor API lacks this optional attribute.
                values.append(None)
                continue
            self.library.require(status, "algorithm configuration query")
            if written.value != ctypes.sizeof(value):
                raise TaichiRuntimeError(
                    "cuBLASLt algorithm configuration ABI size differs"
                )
            values.append(int(value.value))
        return _AlgorithmChoice(tuple(values), int(workspace_bytes))

    def restore(self, plan, choice):
        algorithm = _MatmulAlgo()
        self.library.require(
            self.init(
                plan.provider._handle,
                68,
                0,
                0,
                0,
                0,
                0,
                choice.configuration[0],
                ctypes.byref(algorithm),
            ),
            "algorithm configuration initialization",
        )
        for attribute, ((_, dtype), value) in enumerate(
            zip(_CONFIG, choice.configuration)
        ):
            if attribute == 0 or value is None:
                continue
            native = dtype(value)
            self.library.require(
                self.set(
                    ctypes.byref(algorithm),
                    attribute,
                    ctypes.byref(native),
                    ctypes.sizeof(native),
                ),
                "algorithm configuration restoration",
            )
        result = _HeuristicResult()
        self.library.require(
            self.check(
                plan.provider._handle,
                plan._matmul_desc,
                *plan._layouts,
                ctypes.byref(algorithm),
                ctypes.byref(result),
            ),
            "restored algorithm support",
        )
        self.library.require(result.state, "restored algorithm state")
        observed = self.describe(algorithm, result.workspace_size)
        if observed != choice:
            raise TaichiRuntimeError(
                "cuBLASLt restored algorithm configuration or workspace drifted"
            )
        result.algo = algorithm  # AlgoCheck deliberately does not populate this field.
        return result

    def shortlist(self, plan, count):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 32
        ):
            raise ValueError("cuBLASLt preparation shortlist count must be in [1, 32]")
        library = self.library
        preference = ctypes.c_void_p()
        library.require(
            library.preference_create(ctypes.byref(preference)),
            "recipe preference creation",
        )
        try:
            _set_attribute(
                library,
                library.preference_set_attribute,
                preference,
                1,
                ctypes.c_size_t(plan.workspace_limit_bytes),
                "recipe workspace limit",
            )
            results, returned = (_HeuristicResult * count)(), ctypes.c_int()
            library.require(
                library.heuristic(
                    plan.provider._handle,
                    plan._matmul_desc,
                    *plan._layouts,
                    preference,
                    count,
                    results,
                    ctypes.byref(returned),
                ),
                "recipe heuristic shortlist",
            )
            if not 0 <= returned.value <= count:
                raise TaichiRuntimeError("cuBLASLt returned an invalid shortlist count")
            choices = {}
            for result in results[: returned.value]:
                if (
                    result.state != 0
                    or result.workspace_size > plan.workspace_limit_bytes
                ):
                    continue
                choice = self.describe(result.algo, result.workspace_size)
                if choice not in choices:
                    choices[choice] = _HeuristicResult.from_buffer_copy(result)
            if not choices:
                raise TaichiRuntimeError(
                    "cuBLASLt found no compatible physical matmul strategy"
                )
            return choices
        finally:
            library.require(
                library.preference_destroy(preference), "recipe preference destruction"
            )


class _MatmulRecipePlan(CublasLtMatmulPlan):
    """Graph-only physical plan, or host-only preparation descriptors."""

    def __init__(
        self,
        provider,
        semantics,
        *,
        layout,
        epilogue,
        workspace_limit_bytes,
        choice=None,
        shortlist_size=8,
        preparation_only=False,
    ):
        self._requested_layout, self._requested_epilogue = layout, epilogue
        self._selected_choice = choice
        self._shortlist_size = shortlist_size
        self._preparation_only = preparation_only
        with provider._lock:
            provider._validate_lifetime()
            if not hasattr(provider._library, "_recipe_algorithm_api"):
                provider._library._recipe_algorithm_api = _AlgorithmApi(
                    provider._library
                )
            self._algorithm_api = provider._library._recipe_algorithm_api
            super().__init__(
                provider, **semantics, workspace_limit_bytes=workspace_limit_bytes
            )
            provider._plans.add(self)

    def _create_native_plan(self):
        self._create_descriptors(
            layout=self._requested_layout, epilogue=self._requested_epilogue
        )
        if self._selected_choice is None:
            found = self._algorithm_api.shortlist(self, self._shortlist_size)
            self.choices = tuple(found)
            self._heuristic = next(iter(found.values()))
        else:
            self._heuristic = self._algorithm_api.restore(self, self._selected_choice)
            self.choices = (self._selected_choice,)
        self.workspace_bytes = int(self._heuristic.workspace_size)
        if self.workspace_bytes > self.workspace_limit_bytes:
            raise TaichiRuntimeError(
                "Selected cuBLASLt workspace exceeds the preparation limit"
            )
        if self.workspace_bytes and not self._preparation_only:
            self.workspace = impl.ndarray(u8, shape=self.workspace_bytes)

    def execute(self, bindings):
        raise TaichiRuntimeError("cuBLASLt recipe plans require retained Graph capture")

    def close(self):
        if self._preparation_only:
            # Descriptors never submitted mathematics or owned GPU scratch.
            # No unrelated device synchronization is needed to retire them.
            with self.provider._lock, self._lock:
                self._close_native()
            return None
        return super().close()
