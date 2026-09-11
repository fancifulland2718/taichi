from typing import Any

from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.enums import Format
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang.matrix import (
    Matrix,
    MatrixNdarray,
    MatrixType,
    VectorNdarray,
    VectorType,
)
from taichi_forge.types.annotations import template
from taichi_forge.types.ndarray_type import NdarrayType
from taichi_forge.types._argument_descriptor import (
    describe_annotation,
    describe_element_type,
    describe_symbolic_arg,
    python_compound_type,
)
from taichi_forge.types.texture_type import RWTextureType, TextureType
from taichi_forge.types.ray_type import AccelerationStructureType

template_types = (NdarrayType, TextureType, template)


def reject_acceleration_structure_arguments(kernel, integration):
    """Fail closed where no serialized acceleration-structure ABI exists."""

    from taichi_forge.types.ray_type import AccelerationStructureType

    names = [
        arg.name
        for arg in kernel.arguments
        if isinstance(arg.annotation, AccelerationStructureType)
    ]
    if names:
        raise TaichiCompilationError(
            "Vulkan acceleration-structure kernel arguments are JIT-only; "
            f"{integration} does not support them (arguments: "
            + ", ".join(names)
            + ")"
        )


def check_type_match(lhs, rhs):
    return describe_element_type(lhs).matches(describe_element_type(rhs))


def produce_injected_args_from_template(kernel, template_args):
    injected_args = []
    num_template_args = len([arg.annotation for arg in kernel.arguments if isinstance(arg.annotation, template_types)])
    assert num_template_args == len(
        template_args
    ), f"Need {num_template_args} inputs to instantiate the template parameters, got {len(template_args)}"
    for arg in kernel.arguments:
        anno = arg.annotation
        if isinstance(anno, template_types):
            injected_args.append(template_args[arg.name])
        elif isinstance(anno, RWTextureType):
            texture_shape = (2,) * anno.num_dimensions
            fmt = anno.fmt
            injected_args.append(Texture(fmt, texture_shape))
        else:
            injected_args.append(0)
    return injected_args


def _produce_injected_arg(arg, symbolic_arg=None, has_symbolic_arg=False):
    anno = arg.annotation
    if isinstance(anno, AccelerationStructureType):
        if has_symbolic_arg and (
            symbolic_arg is None
            or describe_symbolic_arg(symbolic_arg).kind != "acceleration_structure"
        ):
            raise TaichiCompilationError(
                f"Graph argument {arg.name} requires ArgKind.ACCELERATION_STRUCTURE"
            )
        # AS descriptors do not specialize codegen. Never allocate a dummy
        # BLAS/TLAS to compile a symbolic Graph dispatch.
        return None
    if isinstance(anno, NdarrayType):
        if has_symbolic_arg:
            symbolic_descriptor = describe_symbolic_arg(symbolic_arg)
            dtype = python_compound_type(symbolic_descriptor.element)
            ndim = symbolic_descriptor.ndim
        else:
            ndim = anno.ndim
            dtype = anno.dtype

        if anno.ndim is not None and ndim != anno.ndim:
            raise TaichiCompilationError(
                f"{ndim} from Arg {arg.name} doesn't match kernel's " f"annotated ndim={anno.ndim}"
            )

        expected_descriptor = describe_annotation(anno)
        if anno.dtype is not None and not expected_descriptor.element.matches(dtype):
            raise TaichiCompilationError(
                f" Arg {arg.name}'s dtype "
                f"{describe_element_type(dtype).display_name()} doesn't match "
                f"kernel's annotated dtype="
                f"{expected_descriptor.element.display_name()}"
            )

        shape = (2,) * ndim
        if isinstance(dtype, VectorType):
            return VectorNdarray(dtype.n, dtype=dtype.dtype, shape=shape)
        if isinstance(dtype, MatrixType):
            return MatrixNdarray(dtype.n, dtype.m, dtype=dtype.dtype, shape=shape)
        return ScalarNdarray(dtype, shape)

    if isinstance(anno, RWTextureType):
        if has_symbolic_arg:
            expected = describe_annotation(anno)
            actual = describe_symbolic_arg(symbolic_arg)
            if actual.kind != "rw_texture" or actual.ndim != expected.ndim:
                raise TaichiCompilationError(
                    f"RWTexture descriptor mismatch for argument {arg.name}: "
                    f"expected ndim={expected.ndim}, got {actual.kind} "
                    f"ndim={actual.ndim}."
                )
            if actual.fmt != expected.fmt:
                raise TaichiCompilationError(
                    f"RWTexture format mismatch for argument {arg.name}: " f"expected {expected.fmt}, got {actual.fmt}."
                )
        return Texture(anno.fmt, (2,) * anno.num_dimensions)
    if isinstance(anno, TextureType):
        if has_symbolic_arg:
            expected = describe_annotation(anno)
            actual = describe_symbolic_arg(symbolic_arg)
            if actual.kind not in ("texture", "rw_texture") or actual.ndim != expected.ndim:
                raise TaichiCompilationError(
                    f"Texture descriptor mismatch for argument {arg.name}: "
                    f"expected ndim={expected.ndim}, got {actual.kind} "
                    f"ndim={actual.ndim}."
                )
        return Texture(Format.rgba8, (2,) * anno.num_dimensions)
    if isinstance(anno, MatrixType):
        if has_symbolic_arg:
            expected = describe_annotation(anno).element
            actual = describe_symbolic_arg(symbolic_arg).element
            if not expected.matches(actual):
                raise RuntimeError(
                    f"Matrix descriptor mismatch, expected "
                    f"{expected.display_name()} but dispatched "
                    f"{actual.display_name()}."
                )
        return Matrix([0] * anno.n * anno.m, dt=anno.dtype)

    dtype = describe_symbolic_arg(symbolic_arg).element if has_symbolic_arg else describe_element_type(anno)
    expected = describe_element_type(anno)
    if not expected.matches(dtype):
        raise TaichiCompilationError(
            f" Arg {arg.name}'s dtype {dtype.display_name()} doesn't match "
            f"kernel's annotated dtype={expected.display_name()}"
        )
    # For primitive types, we can just inject a dummy value.
    return 0


def produce_injected_args(kernel, symbolic_args=None):
    has_symbolic_args = symbolic_args is not None
    return [
        _produce_injected_arg(
            arg,
            symbolic_args[index] if has_symbolic_args else None,
            has_symbolic_args,
        )
        for index, arg in enumerate(kernel.arguments)
    ]


def _validate_graph_template_exemplar(arg, symbolic_arg, exemplar):
    anno = arg.annotation
    if isinstance(anno, NdarrayType):
        if not isinstance(exemplar, Ndarray):
            raise TaichiCompilationError(f"Graph template exemplar {arg.name} must be a Taichi ndarray")
        symbolic_descriptor = describe_symbolic_arg(symbolic_arg)
        exemplar_descriptor = describe_element_type(exemplar.element_type)
        if len(exemplar.shape) != symbolic_descriptor.ndim or not symbolic_descriptor.element.matches(
            exemplar_descriptor
        ):
            raise TaichiCompilationError(
                f"Graph template exemplar {arg.name} does not match its "
                "symbolic ndarray dtype, ndim, or element shape"
            )
    elif isinstance(anno, TextureType) and not isinstance(exemplar, Texture):
        raise TaichiCompilationError(f"Graph template exemplar {arg.name} must be a Taichi texture")


def produce_injected_args_for_graph(kernel, symbolic_args, template_args=None):
    """Inject compile-time values while preserving Graph runtime arguments."""

    if template_args is None:
        has_required_template = any(isinstance(arg.annotation, template) for arg in kernel.arguments)
        if not has_required_template:
            injected = produce_injected_args(kernel, symbolic_args=symbolic_args)
            for argument, value in zip(kernel.arguments, injected):
                if isinstance(argument.annotation, NdarrayType):
                    value._graph_runtime_affine_exemplar = True
            return injected
        template_args = {}
    if not isinstance(template_args, dict):
        raise TaichiCompilationError("Graph template_args must be a dict keyed by kernel argument name")

    template_keys = frozenset(template_args)
    symbolic_key = tuple(describe_symbolic_arg(arg).structural_key() for arg in symbolic_args)
    cache = getattr(kernel, "_graph_template_injection_cache", None)
    if cache is not None and cache[0] == template_keys and cache[1] == symbolic_key:
        return [template_args[value] if kind == "template" else value for kind, value in cache[2]]

    arguments_by_name = {arg.name: arg for arg in kernel.arguments}
    unknown = sorted(set(template_args) - set(arguments_by_name))
    if unknown:
        raise TaichiCompilationError("Unknown Graph template arguments: " + ", ".join(unknown))

    invalid = sorted(
        name for name in template_args if not isinstance(arguments_by_name[name].annotation, template_types)
    )
    if invalid:
        raise TaichiCompilationError(
            "Graph template_args can only bind ti.template, ndarray, or "
            "texture parameters; invalid: " + ", ".join(invalid)
        )

    required = {arg.name for arg in kernel.arguments if isinstance(arg.annotation, template)}
    missing = sorted(required - set(template_args))
    if missing:
        raise TaichiCompilationError("Missing required Graph template arguments: " + ", ".join(missing))

    injected_args = []
    symbolic_index = 0
    for arg in kernel.arguments:
        anno = arg.annotation
        if isinstance(anno, template):
            injected_args.append(template_args[arg.name])
            continue

        if symbolic_index >= len(symbolic_args):
            raise TaichiCompilationError(f"Missing symbolic Graph argument for kernel parameter " f"{arg.name}")
        symbolic_arg = symbolic_args[symbolic_index]
        symbolic_index += 1
        reconstructed = _produce_injected_arg(arg, symbolic_arg, True)
        if arg.name in template_args:
            _validate_graph_template_exemplar(arg, symbolic_arg, template_args[arg.name])
        if isinstance(anno, NdarrayType):
            # A symbolic Graph ndarray may be rebound to a qualified affine
            # dense storage view at run time. Compile the Graph specialization
            # with runtime strides even though its temporary compile exemplar
            # is itself canonical. Ordinary direct ndarray calls retain their
            # branch-free canonical specialization.
            reconstructed._graph_runtime_affine_exemplar = True
            injected_args.append(reconstructed)
        else:
            injected_args.append(template_args.get(arg.name, reconstructed))

    if symbolic_index != len(symbolic_args):
        raise TaichiCompilationError(
            f"Graph dispatch received {len(symbolic_args)} symbolic "
            f"arguments but kernel expects {symbolic_index} after "
            "compile-time template binding"
        )

    cacheable = all(
        isinstance(arg.annotation, template)
        or not isinstance(arg.annotation, (NdarrayType, TextureType, RWTextureType))
        for arg in kernel.arguments
    )
    if cacheable:
        actions = tuple(
            ("template", arg.name) if isinstance(arg.annotation, template) else ("static", injected)
            for arg, injected in zip(kernel.arguments, injected_args)
        )
        kernel._graph_template_injection_cache = (
            template_keys,
            symbolic_key,
            actions,
        )
    return injected_args


def json_data_model(f):
    """
    Decorates a JSON data model. A JSON data model MUST NOT have any member
    functions and it MUST be constructible from a JSON object.

    This is merely a marker.
    """
    f._is_json_data_model = True
    return f


def is_json_data_model(cls) -> bool:
    return hasattr(cls, "_is_json_data_model")


def dump_json_data_model(x: object) -> Any:
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    if isinstance(x, (list, tuple)):
        return [dump_json_data_model(e) for e in x]
    if isinstance(x, dict):
        return {k: dump_json_data_model(v) for k, v in x.items()}
    if is_json_data_model(x):
        return {k: dump_json_data_model(v) for k, v in x.__dict__.items() if k != "_is_json_data_model"}
    return x
