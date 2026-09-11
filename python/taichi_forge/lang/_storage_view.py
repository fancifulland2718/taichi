"""Internal unified storage description and shadow-validation helpers.

This module does not define a new owning tensor. It adapts existing Forge
objects to the backend-neutral C++ storage descriptor without resolving a raw
pointer, allocating temporary storage, or submitting backend work.
"""

from dataclasses import dataclass
import os
from types import MappingProxyType
import warnings

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import (
    Ndarray,
    StructNdarrayScalarMemberView,
    StructNdarrayTensorMemberView,
)
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.types.ndarray_type import NdarrayTypeMetadata


_SHADOW_ENV = "TI_STORAGE_VIEW_SHADOW"
_SHADOW_MODES = {"off", "warn", "error"}


def _shadow_mode():
    mode = os.environ.get(_SHADOW_ENV, "off").strip().lower()
    if mode in ("", "0", "false", "no"):
        return "off"
    if mode in ("1", "true", "yes"):
        return "error"
    if mode not in _SHADOW_MODES:
        warnings.warn(
            f"{_SHADOW_ENV}={mode!r} is invalid; storage shadow validation " "is disabled",
            RuntimeWarning,
            stacklevel=2,
        )
        return "off"
    return mode


STORAGE_VIEW_SHADOW_MODE = _shadow_mode()


@dataclass(frozen=True)
class StorageRequirement:
    """Consumer capability contract for a dense storage description."""

    scalar_type: object = None
    min_index_rank: int = 0
    max_index_rank: int = 12
    max_element_rank: int = 12
    require_ndarray_abi: bool = False
    accept_compact_subrange: bool = True
    accept_single_record_stride: bool = False
    accept_general_affine: bool = False
    require_unique_mapping: bool = False
    require_writable: bool = False
    accept_external_owner: bool = False
    allow_materialization: bool = False


@dataclass(frozen=True)
class StorageDescription:
    """Immutable result of adapting one existing storage object."""

    _result: object = None
    _failure_reason: str = "kNone"

    @property
    def supported(self):
        return bool(self._result is not None and self._result.ok)

    @property
    def failure_reason(self):
        if self._result is not None:
            return self._result.reason
        return self._failure_reason

    @property
    def descriptor(self):
        if not self.supported:
            return None
        return self._result.descriptor

    @property
    def properties(self):
        descriptor = self.descriptor
        if descriptor is None:
            return MappingProxyType({})
        return MappingProxyType(dict(descriptor.properties))


class DenseNdarrayView:
    """Explicit non-owning ndarray-ABI view over existing dense storage."""

    __slots__ = (
        "_source",
        "_description",
        "_descriptor",
        "_runtime_prog",
        "_runtime_storage_arguments",
        "_element_type",
        "_type_metadata",
        "shape",
        "grad",
        "__weakref__",
    )
    _is_dense_ndarray_view = True

    def __init__(self, source, description):
        descriptor = description.descriptor
        if descriptor is None:
            raise ValueError(f"cannot view unsupported storage: {description.failure_reason}")
        element_shape = tuple(int(extent) for extent in descriptor.element_shape)
        if len(element_shape) == 0:
            element_type = descriptor.scalar_type
        elif len(element_shape) <= 2:
            element_type = _ti_core.get_type_factory_instance().get_tensor_type(
                element_shape, descriptor.scalar_type
            )
        else:
            raise ValueError("ndarray_view supports scalar, vector, or matrix elements")
        self._source = source
        self._description = description
        self._descriptor = descriptor
        self._runtime_prog = _current_program()
        self._runtime_storage_arguments = {}
        self._runtime_storage_argument("ordinary_kernel", "ordinary")
        self._element_type = element_type
        self.shape = tuple(int(extent) for extent in descriptor.index_shape)
        self.grad = None
        self._type_metadata = NdarrayTypeMetadata(element_type, self.shape, False)

    @property
    def description(self):
        return self._description

    @property
    def descriptor(self):
        return self._descriptor

    def _runtime_storage_argument(self, consumer, mode):
        program = _current_program()
        if program is not self._runtime_prog:
            raise RuntimeError(
                "dense storage view belongs to another Taichi runtime"
            )
        key = (impl.current_cfg().arch, consumer, mode)
        cached = self._runtime_storage_arguments.get(key)
        if cached is not None:
            return cached
        argument = _ti_core._make_runtime_storage_argument(
            program, self._descriptor, consumer, mode
        )
        qualification = argument.qualification
        if (
            not qualification["bindable"]
            or not qualification["zero_copy_qualified"]
        ):
            raise ValueError(
                "storage cannot be bound as a runtime argument: "
                f"{qualification['reason']}"
            )
        if mode in ("replay", "capture") and not qualification["replayable"]:
            raise ValueError(
                "storage cannot be replayed as a Graph argument: "
                f"{qualification['reason']}"
            )
        if mode == "capture" and not qualification["capturable"]:
            raise ValueError(
                "storage cannot be captured as a Graph argument: "
                f"{qualification['reason']}"
            )
        self._runtime_storage_arguments[key] = argument
        return argument

    @property
    def runtime_argument(self):
        return self._runtime_storage_argument("ordinary_kernel", "ordinary")

    def get_type(self):
        return self._type_metadata

    def __repr__(self):
        return (
            f"DenseNdarrayView(shape={self.shape}, element_type={self._element_type}, "
            f"source={self.descriptor.source_kind})"
        )


def _shape_tuple(value):
    shape = value.shape
    if shape is None:
        raise RuntimeError("storage object must be placed before description")
    if isinstance(shape, tuple):
        return tuple(int(extent) for extent in shape)
    if isinstance(shape, list):
        return tuple(int(extent) for extent in shape)
    return (int(shape),)


def _current_program():
    program = impl.get_runtime().prog
    if program is None:
        raise RuntimeError("ti.init() must be called before storage description")
    return program


def _describe_field(field, access):
    impl.get_runtime().materialize()
    program = _current_program()
    if isinstance(field, ScalarField):
        components = (field.snode.ptr,)
        return _ti_core._describe_dense_field_storage(
            program,
            components[0],
            components,
            field.dtype,
            _shape_tuple(field),
            (),
            access,
        )
    if isinstance(field, MatrixField):
        indices = tuple(field._native_dense_component_indices())
        scalar_fields = tuple(field.get_scalar_field(*index) for index in indices)
        if not scalar_fields:
            return None
        element_shape = (int(field.n),) if int(field.ndim) == 1 else (int(field.n), int(field.m))
        components = tuple(item.snode.ptr for item in scalar_fields)
        return _ti_core._describe_dense_field_storage(
            program,
            components[0],
            components,
            field.dtype,
            _shape_tuple(field),
            element_shape,
            access,
        )
    return None


def describe_storage(obj, *, access="readwrite"):
    """Describe an existing dense storage object without resolving a pointer."""

    if isinstance(obj, DenseNdarrayView):
        requested = {"read": "readonly", "write": "writeonly"}.get(access, access)
        if requested not in ("readonly", "writeonly", "readwrite"):
            raise ValueError("storage access must be read, write, or readwrite")
        available = obj.descriptor.access.removeprefix("k").lower()
        # Preserve the source descriptor's actual permission. A consumer may
        # require only reads from a writable view; that does not widen access.
        if requested != available and available != "readwrite":
            return StorageDescription(_failure_reason="kReadOnlySource")
        return obj.description
    if isinstance(obj, StructNdarrayScalarMemberView):
        result = _ti_core._describe_struct_member_storage(
            obj.base.arr,
            obj.dtype,
            _shape_tuple(obj),
            (),
            int(obj.offset),
            int(obj.stride),
            False,
            access,
        )
        return StorageDescription(result)
    if isinstance(obj, StructNdarrayTensorMemberView):
        result = _ti_core._describe_struct_member_storage(
            obj.base.arr,
            obj.scalar_dtype,
            _shape_tuple(obj),
            tuple(int(extent) for extent in obj.element_shape),
            int(obj.offset),
            int(obj.stride),
            True,
            access,
        )
        return StorageDescription(result)
    if isinstance(obj, Ndarray):
        if obj.arr is None:
            return StorageDescription(_failure_reason="kInvalidOwner")
        return StorageDescription(_ti_core._describe_ndarray_storage(obj.arr, access))
    field_result = _describe_field(obj, access)
    if field_result is not None:
        return StorageDescription(field_result)
    return StorageDescription(_failure_reason="kUnsupportedStorageKind")


def _flatten_storage_to_scalar_vector(description):
    """Return a no-copy scalar-flat description for compact dense storage."""

    if not isinstance(description, StorageDescription):
        raise TypeError("description must be a StorageDescription")
    descriptor = description.descriptor
    if descriptor is None:
        return StorageDescription(_failure_reason=description.failure_reason)
    return StorageDescription(
        _ti_core._flatten_dense_storage_to_scalar_vector(descriptor)
    )


def _slice_storage_description(description, slices):
    descriptor = description.descriptor
    rank = len(tuple(descriptor.index_shape))
    if isinstance(slices, slice):
        slices = (slices,)
    else:
        slices = tuple(slices)
    if len(slices) != rank:
        raise ValueError(f"ndarray_view requires exactly {rank} index slices")

    starts = []
    lengths = []
    steps = []
    for extent, item in zip(descriptor.index_shape, slices):
        if not isinstance(item, slice):
            raise TypeError(
                "ndarray_view accepts one slice per index axis; "
                "integer indexing, ellipsis, and axis permutation are unsupported"
            )
        try:
            start, stop, step = item.indices(int(extent))
        except TypeError as exc:
            raise TypeError("ndarray_view slice bounds must be integers") from exc
        except ValueError as exc:
            raise ValueError(
                "ndarray_view requires strictly positive slice steps"
            ) from exc
        if step <= 0:
            raise ValueError("ndarray_view requires strictly positive slice steps")
        starts.append(start)
        lengths.append(len(range(start, stop, step)))
        steps.append(step)

    result = _ti_core._slice_dense_storage(
        descriptor, starts, lengths, steps
    )
    if not result.ok:
        raise ValueError(
            "storage cannot form a positive-stride affine view: "
            f"{result.reason}"
        )
    return StorageDescription(result)


def ndarray_view(obj, *, slices=None, access="readwrite"):
    """Create an explicit zero-copy dense view over existing storage.

    ``slices`` may contain one positive-step :class:`slice` per logical index
    axis. The view preserves rank and element layout. Negative steps, integer
    indexing, broadcast/overlap, axis permutation, and arbitrary element
    strides are intentionally outside the current execution contract.
    """

    if access != "readwrite":
        raise ValueError("executable ndarray_view currently requires access='readwrite'")
    description = describe_storage(obj, access=access)
    if not description.supported:
        raise ValueError(f"storage cannot be described: {description.failure_reason}")
    if slices is not None:
        description = _slice_storage_description(description, slices)
    qualification = qualify_storage(
        description,
        StorageRequirement(
            max_element_rank=2,
            accept_general_affine=True,
            require_unique_mapping=True,
            require_writable=True,
            allow_materialization=False,
        ),
    )
    if not qualification["supported"] or qualification["requires_materialization"]:
        raise ValueError(
            "storage cannot be bound as a zero-copy dense view: "
            f"{qualification['reason']}"
        )
    return DenseNdarrayView(obj, description)

def validate_storage_owner(description):
    """Return a stable owner-status code for the current Program."""

    if not isinstance(description, StorageDescription):
        raise TypeError("description must be a StorageDescription")
    descriptor = description.descriptor
    if descriptor is None:
        return description.failure_reason
    program = impl.get_runtime().prog
    if program is None:
        return "kDifferentProgram"
    return _ti_core._validate_storage_owner(program, descriptor)


def qualify_storage(description, requirement):
    """Qualify one description against an explicit consumer contract."""

    if not isinstance(description, StorageDescription):
        raise TypeError("description must be a StorageDescription")
    if not isinstance(requirement, StorageRequirement):
        raise TypeError("requirement must be a StorageRequirement")
    descriptor = description.descriptor
    if descriptor is None:
        return MappingProxyType(
            {
                "supported": False,
                "execution_mode": "kUnsupported",
                "reason": description.failure_reason,
                "requires_materialization": False,
                "estimated_copy_bytes": 0,
            }
        )
    result = _ti_core._qualify_dense_storage(
        descriptor,
        requirement.scalar_type,
        int(requirement.min_index_rank),
        int(requirement.max_index_rank),
        int(requirement.max_element_rank),
        bool(requirement.require_ndarray_abi),
        bool(requirement.accept_compact_subrange),
        bool(requirement.accept_single_record_stride),
        bool(requirement.accept_general_affine),
        bool(requirement.require_unique_mapping),
        bool(requirement.require_writable),
        bool(requirement.accept_external_owner),
        bool(requirement.allow_materialization),
    )
    return MappingProxyType(dict(result))


def analyze_storage_alias(lhs, rhs):
    """Return ProvenDisjoint, ProvenOverlap, or Unknown."""

    if not isinstance(lhs, StorageDescription) or not isinstance(rhs, StorageDescription):
        raise TypeError("alias operands must be StorageDescription objects")
    if lhs.descriptor is None or rhs.descriptor is None:
        return "kUnknown"
    return _ti_core._analyze_storage_alias(lhs.descriptor, rhs.descriptor)


def _report_shadow_mismatch(kind, mismatches):
    if not mismatches:
        return
    message = f"unified storage shadow mismatch for {kind}: " + "; ".join(mismatches)
    if STORAGE_VIEW_SHADOW_MODE == "error":
        raise RuntimeError(message)
    if STORAGE_VIEW_SHADOW_MODE == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def shadow_validate_primitive_view(obj, legacy):
    """Compare a legacy algorithms view with the unified descriptor."""

    description = describe_storage(obj)
    if not description.supported:
        _report_shadow_mismatch(
            legacy.storage,
            [f"new descriptor rejected: {description.failure_reason}"],
        )
        return description

    descriptor = description.descriptor
    properties = description.properties
    mismatches = []
    if str(descriptor.scalar_type) != str(legacy.dtype):
        mismatches.append(f"dtype {descriptor.scalar_type} != {legacy.dtype}")
    if tuple(descriptor.index_shape) != tuple(legacy.shape):
        mismatches.append(f"index shape {tuple(descriptor.index_shape)} " f"!= {tuple(legacy.shape)}")
    if tuple(descriptor.element_shape) != tuple(legacy.element_shape):
        mismatches.append(f"element shape {tuple(descriptor.element_shape)} " f"!= {tuple(legacy.element_shape)}")
    if legacy.storage.startswith("struct_"):
        if int(descriptor.byte_offset) != int(legacy.offset):
            mismatches.append(f"byte offset {descriptor.byte_offset} != {legacy.offset}")
        if int(properties["record_stride"]) != int(legacy.stride):
            mismatches.append(f"record stride {properties['record_stride']} " f"!= {legacy.stride}")
    elif legacy.storage == "dense_field":
        stride = int(properties["record_stride"])
        if stride != int(legacy.stride):
            mismatches.append(f"record stride {stride} != {legacy.stride}")
        if stride > 0 and (int(descriptor.byte_offset) - int(legacy.offset)) % stride:
            mismatches.append("root-relative byte offset is inconsistent with the " "legacy record offset")
    _report_shadow_mismatch(legacy.storage, mismatches)
    return description


def shadow_validate_dense_field_descriptor(field, legacy):
    """Compare the LinearOperator field descriptor with unified metadata."""

    description = describe_storage(field)
    if not description.supported:
        _report_shadow_mismatch(
            legacy.storage_kind,
            [f"new descriptor rejected: {description.failure_reason}"],
        )
        return description

    descriptor = description.descriptor
    properties = description.properties
    mismatches = []
    expected = {
        "dtype": (str(descriptor.scalar_type), str(legacy.dtype)),
        "index_shape": (
            tuple(descriptor.index_shape),
            tuple(legacy.index_shape),
        ),
        "element_shape": (
            tuple(descriptor.element_shape),
            tuple(legacy.element_shape),
        ),
        "scalar_extent": (
            int(properties["scalar_count"]),
            int(legacy.scalar_extent),
        ),
        "item_stride": (
            int(properties["record_stride"]),
            int(legacy.item_stride),
        ),
        "tree_id": (
            int(descriptor.tree_identity[0]),
            int(legacy.tree_id),
        ),
        "node_ids": (
            tuple(descriptor.component_snode_ids),
            tuple(legacy.node_ids),
        ),
    }
    for name, (current, previous) in expected.items():
        if current != previous:
            mismatches.append(f"{name} {current!r} != {previous!r}")
    stride = int(properties["record_stride"])
    if stride > 0 and (int(descriptor.byte_offset) - int(legacy.byte_offset)) % stride:
        mismatches.append("root-relative byte offset is inconsistent with the " "legacy field offset")
    _report_shadow_mismatch(legacy.storage_kind, mismatches)
    return description


__all__ = [
    "DenseNdarrayView",
    "StorageDescription",
    "StorageRequirement",
    "analyze_storage_alias",
    "describe_storage",
    "ndarray_view",
    "qualify_storage",
    "validate_storage_owner",
]
