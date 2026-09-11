"""Prepared stable compaction with one device prefix shared across columns."""

import hashlib
import json

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._prepared_primitive import _PreparedNativePlan
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError


class PreparedCompactPlan(_PreparedNativePlan):
    """Compact records with nonzero i32 flags, preserving input order.

    Bindings must be compact, naturally aligned CUDA/Vulkan dense ranges.
    Values/output are matching scalar, vector or matrix records with 32/64-bit
    scalar lanes. Output capacity must cover the input capacity. Count is a
    scalar i32 field or an i32 vector; the result stays in count[0] (or count[None]
    for scalar fields). Only output[:count] is defined; input data is unchanged.

    Preparation performs no mathematics or GPU submission. In-place content
    updates reuse the plan; new storage/layout requires preparation again.
    Native scratch remains lazy and Program-owned. This is an explicit
    runtime-ordered primitive, not a capture-safe command or a raw search axis.
    """

    def __init__(self, values, flags, output, count):
        self._prepare_columns(((values, output),), flags, count)

    @classmethod
    def _from_columns(cls, columns, flags, count):
        plan = cls.__new__(cls)
        plan._prepare_columns(tuple(columns), flags, count)
        return plan

    def _prepare_columns(self, columns, flags, count):
        owners = (flags, count, *(value for column in columns for value in column))
        access = ("read", "write", *(mode for _ in columns for mode in ("read", "write")))
        descriptions = tuple(
            describe_storage(value, access="read" if mode == "read" else "readwrite")
            for value, mode in zip(owners, access)
        )
        for description in descriptions:
            if not description.supported:
                raise TaichiRuntimeError(f"Prepared compact storage is unsupported: {description.failure_reason}")
        descriptors = tuple(description.descriptor for description in descriptions)
        program = impl.get_runtime().prog
        prepare = getattr(program, "_prepare_primitive_compact", None)
        if prepare is None:
            raise TaichiRuntimeError("Prepared compact requires native prepared-compaction support")
        command = prepare(descriptors[2::2], descriptors[0], descriptors[3::2], descriptors[1])
        self._program, self._command = program, command
        self._execute = program._execute_primitive_compact
        self._owners, self._access = owners, access
        self._workspace_bytes = None
        self._workspace_effect = "forge-native-compact-workspace"
        self._backend = "cuda" if impl.current_cfg().arch == _ti_core.Arch.cuda else "vulkan"
        self._description = {
            "kind": "stable_compact",
            "backend": self._backend,
            "capacity": int(descriptors[0].index_shape[0]),
            "columns": tuple(
                {"dtype": str(descriptor.scalar_type), "element_shape": tuple(descriptor.element_shape)}
                for descriptor in descriptors[2::2]
            ),
            "predicate": "nonzero_i32",
            "prefix_evaluations": 1,
            "count_location": "device",
        }
        canonical = json.dumps(self._description, sort_keys=True, separators=(",", ":"))
        self._physical_id = "prepared-native-compact-v1:" + hashlib.sha256(canonical.encode()).hexdigest()


def prepare_compact(values, flags, output, count):
    """Prepare stable fixed-binding compaction; see :class:`PreparedCompactPlan`."""
    return PreparedCompactPlan(values, flags, output, count)


__all__ = ["PreparedCompactPlan", "prepare_compact"]
