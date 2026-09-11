from dataclasses import dataclass
from typing import Optional, Tuple

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import util
from taichi_forge.types import primitive_types


@dataclass(frozen=True)
class ElementTypeDescriptor:
    """Canonical internal description of one scalar/aggregate array element."""

    category: str
    logical_type: object
    scalar_type: object
    shape: Optional[Tuple[Optional[int], ...]]

    @property
    def is_complete(self):
        return (
            self.logical_type is not None
            and self.shape is not None
            and None not in self.shape
        )

    def matches(self, actual):
        if not isinstance(actual, ElementTypeDescriptor):
            actual = describe_element_type(actual)
        if self.category == "any":
            return True
        if self.category != actual.category:
            return False
        if self.scalar_type is not None and self.scalar_type != actual.scalar_type:
            return False
        if self.shape is not None:
            if actual.shape is None or len(self.shape) != len(actual.shape):
                return False
            if any(
                expected is not None and expected != observed
                for expected, observed in zip(self.shape, actual.shape)
            ):
                return False
        if self.category == "struct" and self.logical_type is not None:
            return self.logical_type == actual.logical_type
        return True

    def display_name(self):
        if self.category == "any":
            return "any"
        if self.logical_type is not None:
            return self.logical_type.to_string()
        scalar = (
            self.scalar_type.to_string()
            if self.scalar_type is not None
            else "any"
        )
        return f"tensor{self.shape}<{scalar}>"

    def structural_key(self):
        scalar_key = (
            self.scalar_type.to_string()
            if self.scalar_type is not None
            else None
        )
        logical_key = (
            self.logical_type.to_string()
            if self.category == "struct" and self.logical_type is not None
            else None
        )
        return (self.category, scalar_key, self.shape, logical_key)


@dataclass(frozen=True)
class ArgumentTypeDescriptor:
    """Internal structural contract shared by kernel, Graph, and AOT paths."""

    kind: str
    element: Optional[ElementTypeDescriptor] = None
    ndim: Optional[int] = None
    needs_grad: Optional[bool] = None
    fmt: object = None

    def structural_key(self):
        return (
            self.kind,
            self.element.structural_key() if self.element is not None else None,
            self.ndim,
            self.needs_grad,
            str(self.fmt) if self.fmt is not None else None,
        )


def describe_element_type(dtype):
    if dtype is None:
        return ElementTypeDescriptor("any", None, None, None)

    if hasattr(dtype, "tensor_type") and hasattr(dtype, "get_shape"):
        shape = tuple(dtype.get_shape())
        scalar_type = (
            util.cook_dtype(dtype.dtype) if dtype.dtype is not None else None
        )
        logical_type = (
            util.cook_dtype(dtype.tensor_type)
            if dtype.tensor_type is not None
            else None
        )
        return ElementTypeDescriptor(
            "tensor", logical_type, scalar_type, shape
        )

    if hasattr(dtype, "members") and hasattr(dtype, "dtype"):
        return ElementTypeDescriptor("struct", dtype.dtype, dtype.dtype, ())

    logical_type = util.cook_dtype(dtype)
    shape = tuple(logical_type.shape())
    if shape:
        return ElementTypeDescriptor(
            "tensor", logical_type, logical_type.element_type(), shape
        )
    category = (
        "scalar"
        if logical_type in primitive_types.all_types
        else "struct"
    )
    return ElementTypeDescriptor(
        category, logical_type, logical_type, ()
    )


def describe_annotation(annotation):
    # Imports stay local because the annotation modules participate in the
    # frontend import cycle.
    from taichi_forge.lang.matrix import MatrixType
    from taichi_forge.lang.struct import StructType
    from taichi_forge.types.ndarray_type import NdarrayType
    from taichi_forge.types.ray_type import AccelerationStructureType
    from taichi_forge.types.texture_type import RWTextureType, TextureType

    if isinstance(annotation, NdarrayType):
        return ArgumentTypeDescriptor(
            "ndarray",
            describe_element_type(annotation.dtype),
            annotation.ndim,
            annotation.needs_grad,
        )
    if isinstance(annotation, RWTextureType):
        return ArgumentTypeDescriptor(
            "rw_texture", ndim=annotation.num_dimensions, fmt=annotation.fmt
        )
    if isinstance(annotation, TextureType):
        return ArgumentTypeDescriptor(
            "texture", ndim=annotation.num_dimensions
        )
    if isinstance(annotation, AccelerationStructureType):
        return ArgumentTypeDescriptor("acceleration_structure")
    if isinstance(annotation, MatrixType):
        return ArgumentTypeDescriptor(
            "matrix", describe_element_type(annotation)
        )
    if isinstance(annotation, StructType):
        return ArgumentTypeDescriptor(
            "struct", describe_element_type(annotation)
        )
    return ArgumentTypeDescriptor(
        "scalar", describe_element_type(annotation)
    )


def describe_symbolic_arg(symbolic_arg):
    tag = symbolic_arg.tag
    if tag == _ti_core.ArgKind.NDARRAY:
        element_type = symbolic_arg.element_dtype()
        return ArgumentTypeDescriptor(
            "ndarray",
            describe_element_type(element_type),
            symbolic_arg.field_dim,
        )
    if tag == _ti_core.ArgKind.MATRIX:
        return ArgumentTypeDescriptor(
            "matrix", describe_element_type(symbolic_arg.element_dtype())
        )
    if tag == getattr(_ti_core.ArgKind, "ACCELERATION_STRUCTURE", None):
        return ArgumentTypeDescriptor("acceleration_structure")
    if tag == _ti_core.ArgKind.TEXTURE:
        return ArgumentTypeDescriptor(
            "texture", ndim=len(symbolic_arg.texture_shape)
        )
    if tag == _ti_core.ArgKind.RWTEXTURE:
        from taichi_forge.types.texture_type import TY_CH2FORMAT

        return ArgumentTypeDescriptor(
            "rw_texture",
            ndim=len(symbolic_arg.texture_shape),
            fmt=TY_CH2FORMAT[
                (symbolic_arg.channel_format(), symbolic_arg.num_channels)
            ],
        )
    return ArgumentTypeDescriptor(
        "scalar", describe_element_type(symbolic_arg.element_dtype())
    )


def python_compound_type(descriptor):
    if descriptor.category != "tensor":
        return descriptor.logical_type
    if descriptor.shape is None or None in descriptor.shape:
        raise ValueError("Cannot materialize an incomplete tensor descriptor")
    from taichi_forge.lang.matrix import MatrixType, VectorType

    if len(descriptor.shape) == 1:
        return VectorType(descriptor.shape[0], descriptor.scalar_type)
    if len(descriptor.shape) == 2:
        return MatrixType(
            descriptor.shape[0],
            descriptor.shape[1],
            2,
            descriptor.scalar_type,
        )
    raise ValueError(
        "Only scalar, vector, and matrix element descriptors are supported"
    )
