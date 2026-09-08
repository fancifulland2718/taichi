"""Private, immutable algorithm facts for complete cuSPARSELt materializers."""

import ctypes

from taichi_forge.hardware._cusparselt import CusparseLtMatmulPlan, _PlanDesc, _PlanInfo
from taichi_forge.lang.exception import TaichiRuntimeError


_DEFAULT = -(2**31)
_VERSION = 2


class _Configuration(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("algorithm_id", ctypes.c_int32),
        ("split_k", ctypes.c_int32),
        ("split_k_mode", ctypes.c_int32),
        ("split_k_buffers", ctypes.c_int32),
        ("relu", ctypes.c_uint32),
    ]


_Create = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_PlanDesc),
    ctypes.POINTER(_Configuration),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(_PlanInfo),
)
_Get = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(_Configuration),
    ctypes.POINTER(ctypes.c_int32),
)


class _ConfiguredApi(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("execution_abi_version", ctypes.c_uint32),
        ("create_matmul_plan", _Create),
        ("get_configuration", _Get),
    ]


def _api(provider):
    api = getattr(provider, "_configured_execution_api", None)
    if api is None:
        api = provider._runtime.query_execution_api(_ConfiguredApi, version=_VERSION)
        provider._configured_execution_api = api
    return api


def _encode(facts):
    allowed = {
        "algorithm_id",
        "split_k",
        "split_k_mode",
        "split_k_buffers",
        "activation",
    }
    if not isinstance(facts, dict) or not set(facts) <= allowed:
        raise ValueError("cuSPARSELt physical configuration fields are invalid")
    values = [
        facts.get(name, -1 if name == "algorithm_id" else _DEFAULT)
        for name in ("algorithm_id", "split_k", "split_k_mode", "split_k_buffers")
    ]
    if any(
        isinstance(x, bool) or not isinstance(x, int) or not _DEFAULT <= x < 2**31
        for x in values
    ):
        raise ValueError("cuSPARSELt physical configuration requires int32 attributes")
    activation = facts.get("activation", "identity")
    if activation not in ("identity", "relu") or values[0] < -1:
        raise ValueError("cuSPARSELt physical algorithm/activation is invalid")
    return _Configuration(
        ctypes.sizeof(_Configuration), *values, int(activation == "relu")
    )


def _read(provider, handle):
    value, count = _Configuration(), ctypes.c_int32()
    value.struct_size = ctypes.sizeof(value)
    provider._runtime.check_result(
        _api(provider).get_configuration(
            handle, ctypes.byref(value), ctypes.byref(count)
        )
    )
    return (
        dict(
            algorithm_id=int(value.algorithm_id),
            split_k=int(value.split_k),
            split_k_mode=int(value.split_k_mode),
            split_k_buffers=int(value.split_k_buffers),
            activation="relu" if value.relu else "identity",
        ),
        count.value,
    )


class _PreparationPlan(CusparseLtMatmulPlan):
    def execute(self, *args, **kwargs):
        raise TaichiRuntimeError("cuSPARSELt preparation descriptions cannot execute")

    compress = execute
    record = execute


def configured_plan(
    provider,
    m,
    n,
    k,
    *,
    configuration=None,
    preparation_only=False,
    expected_resources=None,
    expected_configuration=None
):
    """Create one physical plan. No data inspection, pruning or vendor search.

    Preparation owns only native descriptors. The selected materialized plan
    gets new compressed/workspace buffers; configurations never mutate in place.
    """
    cls = _PreparationPlan if preparation_only else CusparseLtMatmulPlan
    with provider._lock:
        provider._validate_lifetime()
        return cls(
            provider,
            m,
            n,
            k,
            alignment_bytes=16,
            _configuration={} if configuration is None else configuration,
            _preparation_only=preparation_only,
            _expected_resources=expected_resources,
            _expected_configuration=expected_configuration,
        )
