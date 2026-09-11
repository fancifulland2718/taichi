import operator

import numpy as np
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.expr import Expr, make_expr_group
from taichi_forge.lang.matrix import Matrix
from taichi_forge.lang.util import taichi_scope
from taichi_forge.types import vector
from taichi_forge.types.primitive_types import f32
from taichi_forge.types.texture_type import rw_texture_sampled_type


def _get_entries(mat):
    if isinstance(mat, Matrix):
        return mat.entries
    return [mat]


class TextureSampler:
    def __init__(self, ptr_expr, num_dims) -> None:
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims

    @taichi_scope
    def sample_lod(self, uv, lod):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        args_group = make_expr_group(*_get_entries(uv), lod)
        v = ast_builder.make_texture_op_expr(_ti_core.TextureOpType.kSampleLod, self.ptr_expr, args_group, dbg_info)
        r = impl.call_internal("composite_extract_0", v, with_runtime_context=False)
        g = impl.call_internal("composite_extract_1", v, with_runtime_context=False)
        b = impl.call_internal("composite_extract_2", v, with_runtime_context=False)
        a = impl.call_internal("composite_extract_3", v, with_runtime_context=False)
        return vector(4, f32)([r, g, b, a])

    @taichi_scope
    def fetch(self, index, lod):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        args_group = make_expr_group(*_get_entries(index), lod)
        v = ast_builder.make_texture_op_expr(_ti_core.TextureOpType.kFetchTexel, self.ptr_expr, args_group, dbg_info)
        r = impl.call_internal("composite_extract_0", v, with_runtime_context=False)
        g = impl.call_internal("composite_extract_1", v, with_runtime_context=False)
        b = impl.call_internal("composite_extract_2", v, with_runtime_context=False)
        a = impl.call_internal("composite_extract_3", v, with_runtime_context=False)
        return vector(4, f32)([r, g, b, a])


class RWTextureAccessor:
    def __init__(self, ptr_expr, num_dims, buffer_format) -> None:
        # taichi_python.TexturePtrExpression.
        self.ptr_expr = ptr_expr
        self.num_dims = num_dims
        self.sampled_type = rw_texture_sampled_type(buffer_format)

    def _extract_component(self, value, index):
        suffix = f"_{index}"
        if self.sampled_type != f32:
            suffix = f"_{self.sampled_type}_{index}"
        return impl.call_internal(
            f"composite_extract{suffix}", value, with_runtime_context=False
        )

    @taichi_scope
    def load(self, index):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        args_group = make_expr_group(*_get_entries(index))
        v = ast_builder.make_texture_op_expr(_ti_core.TextureOpType.kLoad, self.ptr_expr, args_group, dbg_info)
        values = [self._extract_component(v, i) for i in range(4)]
        return vector(4, self.sampled_type)(values)

    @taichi_scope
    def store(self, index, value):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        args_group = make_expr_group(*_get_entries(index), *_get_entries(value))
        # A storage-image write has no result. Binding it to a temporary would
        # leave an unknown-typed alloca when optional optimizations are disabled.
        ast_builder.insert_expr_stmt(
            ast_builder.make_texture_op_expr(_ti_core.TextureOpType.kStore, self.ptr_expr, args_group, dbg_info)
        )

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr_expr)
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        ret = [Expr(_ti_core.get_external_tensor_shape_along_axis(self.ptr_expr, i, dbg_info)) for i in range(dim)]
        return ret

    @taichi_scope
    def _loop_range(self):
        """Gets the corresponding taichi_python.Expr to serve as loop range.

        Returns:
            taichi_python.Expr: See above.
        """
        return self.ptr_expr


class Texture:
    """Taichi Texture class.

    Args:
        fmt (ti.Format): Color format of the texture.
        shape (Tuple[int]): Shape of the Texture.
    """

    def __init__(self, fmt, arr_shape, *, sampler=None):
        try:
            arr_shape = tuple(arr_shape)
        except TypeError as exc:
            raise TypeError("Texture shape must be an iterable of dimensions") from exc
        if not 1 <= len(arr_shape) <= 3:
            raise ValueError("Texture shape must have one, two, or three dimensions")
        normalized_shape = []
        for axis, extent in enumerate(arr_shape):
            if isinstance(extent, bool):
                raise TypeError(f"Texture shape axis {axis} must be an integer")
            try:
                extent = operator.index(extent)
            except TypeError as exc:
                raise TypeError(
                    f"Texture shape axis {axis} must be an integer"
                ) from exc
            if not 0 < extent <= 0x7FFFFFFF:
                raise ValueError(
                    f"Texture shape axis {axis} must be in [1, 2^31 - 1]"
                )
            normalized_shape.append(extent)
        arr_shape = tuple(normalized_shape)
        runtime = impl.get_runtime()
        self._runtime_prog = runtime.prog
        arch = impl.current_cfg().arch
        graphics_texture_arches = (
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
            _ti_core.Arch.opengl,
            _ti_core.Arch.gles,
            _ti_core.Arch.metal,
            _ti_core.Arch.dx11,
        )
        if arch not in graphics_texture_arches:
            self._runtime_prog = None
            raise RuntimeError(
                f"Texture resources are unavailable on the {arch.name} backend; "
                "this backend does not implement graphics texture allocation "
                "and TextureOp lowering"
            )
        if sampler is None:
            sampler_config = _ti_core.ImageSamplerConfig()
        else:
            from taichi_forge.hardware._sampling import SamplerConfig

            if not isinstance(sampler, SamplerConfig):
                raise TypeError(
                    "sampler must be a ti.hardware.sampling.SamplerConfig"
                )
            sampler_config = sampler._as_core_config()
        self.tex = self._runtime_prog.create_texture(
            fmt, arr_shape, sampler_config
        )
        self.fmt = fmt
        self.num_dims = len(arr_shape)
        self.shape = arr_shape
        runtime.register_runtime_object(self)

    def _invalidate_runtime(self):
        self.tex = None
        self._runtime_prog = None

    def _delete_runtime_texture(self):
        prog = self._runtime_prog
        tex = self.tex
        self._invalidate_runtime()
        if prog is not None and tex is not None:
            prog.delete_texture(tex)

    def __del__(self):
        if impl is not None:
            try:
                self._delete_runtime_texture()
            except Exception:
                pass

    def from_ndarray(self, ndarray):
        """Loads an ndarray to texture.

        Args:
            ndarray (ti.Ndarray): Source ndarray to load from.
        """
        self.tex.from_ndarray(ndarray.arr)

    def from_field(self, field):
        """Loads a field to texture.

        Args:
            field (ti.Field): Source field to load from.
        """
        self.tex.from_snode(field.snode.ptr)

    def _device_allocation_ptr(self):
        return self.tex.device_allocation_ptr()

    def from_image(self, image):
        """Loads a PIL image to texture. This method is only allowed a 2D texture with `ti.Format.rgba8`.

        Args:
            image (PIL.Image.Image): Source PIL image to load from.

        """
        from PIL import Image  # pylint: disable=import-outside-toplevel

        assert isinstance(image, Image.Image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        assert image.size == tuple(self.shape)

        assert self.num_dims == 2
        # Don't use transpose method since its enums are too new
        image = image.rotate(90, expand=True)
        arr = np.asarray(image)
        from taichi_forge._kernels import (  # pylint: disable=import-outside-toplevel
            load_texture_from_numpy,
        )

        load_texture_from_numpy(self, arr)

    def to_image(self):
        """Saves a texture to a PIL image in RGB mode. This method is only allowed a 2D texture with `ti.Format.rgba8`.

        Returns:
            img (PIL.Image.Image): a PIL image in RGB mode, with the same size as source texture.
        """
        assert self.num_dims == 2
        from PIL import Image  # pylint: disable=import-outside-toplevel

        res = np.zeros(self.shape + (3,), np.uint8)
        from taichi_forge._kernels import (  # pylint: disable=import-outside-toplevel
            save_texture_to_numpy,
        )

        save_texture_to_numpy(self, res)
        return Image.fromarray(res).rotate(270, expand=True)
