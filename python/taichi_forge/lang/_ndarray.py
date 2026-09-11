import numpy as np
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.enums import Layout
from taichi_forge.lang.exception import TaichiIndexError, TaichiRuntimeError
from taichi_forge.lang.util import cook_dtype, get_traceback, python_scope, to_numpy_type
from taichi_forge.types import primitive_types
from taichi_forge.types.ndarray_type import NdarrayTypeMetadata
from taichi_forge.types.utils import is_real, is_signed


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _is_matrix_type(dtype):
    return hasattr(dtype, "tensor_type") and hasattr(dtype, "get_shape") and hasattr(dtype, "dtype")


def _is_struct_type(dtype):
    return hasattr(dtype, "members") and hasattr(dtype, "dtype")


def _native_copy_ad_required(src, dst):
    if getattr(impl.get_runtime(), "target_tape", None) is None:
        return False
    return (
        getattr(src, "grad", None) is not None
        and getattr(dst, "grad", None) is not None
    )


def _native_copy_ad_supported(src, dst):
    if not _native_copy_ad_required(src, dst):
        return True
    from taichi_forge.algorithms._algorithms import (  # pylint: disable=C0415
        _can_native_ad_copy,
    )

    return _can_native_ad_copy(src, dst)


def _record_native_copy_ad(src, dst):
    if not _native_copy_ad_required(src, dst):
        return True
    from taichi_forge.algorithms._algorithms import (  # pylint: disable=C0415
        _record_native_copy_ad as record_native_copy_ad,
    )

    return record_native_copy_ad(src, dst)


def _native_bulk_copy_needs_sync():
    arch = impl.current_cfg().arch
    return arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)


_STRUCT_MEMBER_NATIVE_HOST_COPY_MIN_PAYLOAD_BYTES = 1 << 20
_STRUCT_MEMBER_NATIVE_HOST_COPY_DTYPES = (
    primitive_types.u32,
    primitive_types.i32,
    primitive_types.f32,
    primitive_types.u64,
    primitive_types.i64,
    primitive_types.f64,
)
_NDARRAY_NATIVE_HOST_STAGING_FILL_DTYPES = (
    primitive_types.u64,
    primitive_types.i64,
    primitive_types.f64,
)


def _shape_numel(shape):
    return int(np.prod(shape, dtype=np.int64))


def _struct_scalar_member_native_host_copy_worthwhile(view):
    if view.dtype not in _STRUCT_MEMBER_NATIVE_HOST_COPY_DTYPES:
        return False
    arch = impl.current_cfg().arch
    if arch not in (_ti_core.Arch.x64, _ti_core.Arch.cuda, _ti_core.Arch.vulkan):
        return False
    n = _shape_numel(view.shape)
    payload_bytes = n * view.stride
    member_bytes = n * view.element_size
    if payload_bytes < _STRUCT_MEMBER_NATIVE_HOST_COPY_MIN_PAYLOAD_BYTES:
        return False
    return member_bytes < payload_bytes


def _struct_tensor_member_native_host_copy_worthwhile(view):
    if view.scalar_dtype not in _STRUCT_MEMBER_NATIVE_HOST_COPY_DTYPES:
        return False
    arch = impl.current_cfg().arch
    if arch not in (_ti_core.Arch.x64, _ti_core.Arch.cuda, _ti_core.Arch.vulkan):
        return False
    n = _shape_numel(view.shape)
    payload_bytes = n * view.stride
    member_bytes = n * view.element_size
    if payload_bytes < _STRUCT_MEMBER_NATIVE_HOST_COPY_MIN_PAYLOAD_BYTES:
        return False
    return member_bytes < payload_bytes


def _struct_member_numpy_layout(dtype):
    if dtype in primitive_types.all_types:
        cooked_dtype = cook_dtype(dtype)
        np_dtype = np.dtype(to_numpy_type(cooked_dtype))
        return np_dtype, _ti_core.data_type_alignment(cooked_dtype), _ti_core.data_type_size(cooked_dtype)
    if _is_matrix_type(dtype):
        base_np_dtype = np.dtype(to_numpy_type(dtype.dtype))
        shape = tuple(dtype.get_shape())
        return (
            np.dtype((base_np_dtype, shape)),
            _ti_core.data_type_alignment(dtype.tensor_type),
            _ti_core.data_type_size(dtype.tensor_type),
        )
    if _is_struct_type(dtype):
        return (
            _struct_numpy_dtype(dtype),
            _ti_core.data_type_alignment(dtype.dtype),
            _ti_core.data_type_size(dtype.dtype),
        )
    raise TaichiRuntimeError(f"{dtype} is not supported in StructNdarray")


def _struct_numpy_dtype(struct_type):
    names = []
    formats = []
    offsets = []
    offset = 0
    for name, dtype in struct_type.members.items():
        np_dtype, alignment, size = _struct_member_numpy_layout(dtype)
        offset = _align_up(offset, alignment)
        names.append(name)
        formats.append(np_dtype)
        offsets.append(offset)
        offset += size
    itemsize = _align_up(offset, _ti_core.data_type_alignment(struct_type.dtype))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})


def _normalize_struct_member_path(name):
    if isinstance(name, str):
        if len(name) == 0:
            raise ValueError("StructNdarray.field() expects a non-empty member path")
        return tuple(name.split("."))
    if isinstance(name, (tuple, list)):
        if len(name) == 0:
            raise ValueError("StructNdarray.field() expects a non-empty member path")
        for part in name:
            if not isinstance(part, str) or len(part) == 0:
                raise TypeError("StructNdarray.field() expects a string member path or a tuple/list of strings")
        return tuple(name)
    raise TypeError(f"StructNdarray.field() expects a string member path or a tuple/list of strings, got {type(name)}")


def _format_struct_member_path(path):
    return ".".join(path)


def _normalize_component_index(component):
    if component is None:
        return None
    if isinstance(component, (int, np.integer)):
        return (int(component),)
    if isinstance(component, (tuple, list)):
        if len(component) == 0:
            raise ValueError("StructNdarray.field(component=...) expects a non-empty component index")
        result = []
        for item in component:
            if not isinstance(item, (int, np.integer)):
                raise TypeError("StructNdarray.field(component=...) expects integer component indices")
            result.append(int(item))
        return tuple(result)
    raise TypeError("StructNdarray.field(component=...) expects an int or a tuple/list of ints")


def _extract_struct_numpy_member(struct_arr, path):
    current = struct_arr
    for part in path:
        current = current[part]
    return current


def _resolve_struct_member_path(struct_type, numpy_dtype, path):
    current_type = struct_type
    current_np_dtype = numpy_dtype
    offset = 0
    resolved = []
    for part in path:
        resolved.append(part)
        if not _is_struct_type(current_type):
            raise TypeError(
                f"StructNdarray member path '{_format_struct_member_path(tuple(resolved))}' "
                f"continues through non-struct member type {current_type}."
            )
        if part not in current_type.members:
            raise KeyError(f"StructNdarray has no member '{_format_struct_member_path(tuple(resolved))}'.")
        field_info = current_np_dtype.fields[part]
        offset += field_info[1]
        current_np_dtype = field_info[0]
        current_type = current_type.members[part]
    return current_type, current_np_dtype, offset


def _resolve_matrix_component_offset(member_dtype, member_np_dtype, component):
    if not _is_matrix_type(member_dtype):
        raise TypeError("StructNdarray.field(component=...) is only valid for vector/matrix members.")
    subdtype = member_np_dtype.subdtype
    if subdtype is None:
        raise TypeError(f"StructNdarray member type {member_dtype} does not expose component storage.")
    scalar_np_dtype, shape = subdtype
    if len(component) != len(shape):
        raise ValueError(
            f"StructNdarray.field(component=...) expects {len(shape)} indices for member shape {shape}, "
            f"got {len(component)}."
        )
    for index, extent in zip(component, shape):
        if index < 0 or index >= extent:
            raise IndexError(
                f"StructNdarray.field(component=...) index {index} is out of bounds for member shape {shape}."
            )
    flat_index = np.ravel_multi_index(component, shape, order="C")
    return int(flat_index * scalar_np_dtype.itemsize)


class Ndarray:
    """Taichi ndarray class.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the Ndarray.
    """

    def __init__(self):
        self.host_accessor = None
        self.shape = None
        self.element_type = None
        self.dtype = None
        self.arr = None
        self.layout = Layout.AOS
        self.grad = None
        self._runtime_prog = None
        self._runtime_allocation_identity = None
        self._runtime_storage_arguments = {}
        self._runtime_affine_exemplar_value = False
        self._graph_runtime_affine_exemplar_value = False
        self._runtime_affine_exemplar_token = 0

    @property
    def _runtime_affine_exemplar(self):
        return self._runtime_affine_exemplar_value

    @_runtime_affine_exemplar.setter
    def _runtime_affine_exemplar(self, value):
        self._runtime_affine_exemplar_value = bool(value)
        self._runtime_affine_exemplar_token = int(
            self._runtime_affine_exemplar_value
            or self._graph_runtime_affine_exemplar_value
        )

    @property
    def _graph_runtime_affine_exemplar(self):
        return self._graph_runtime_affine_exemplar_value

    @_graph_runtime_affine_exemplar.setter
    def _graph_runtime_affine_exemplar(self, value):
        self._graph_runtime_affine_exemplar_value = bool(value)
        self._runtime_affine_exemplar_token = int(
            self._runtime_affine_exemplar_value
            or self._graph_runtime_affine_exemplar_value
        )

    def _register_runtime_object(self):
        runtime = impl.get_runtime()
        self._runtime_prog = runtime.prog
        self._runtime_allocation_identity = self.arr.device_allocation().alloc_id
        runtime.register_runtime_object(self)

    def _invalidate_runtime(self):
        self.host_accessor = None
        self.arr = None
        self._runtime_prog = None
        self._runtime_allocation_identity = None
        self.grad = None
        self._runtime_storage_arguments.clear()

    def _runtime_storage_argument(self, consumer, mode):
        if self.arr is None:
            raise RuntimeError(
                "Cannot prepare storage metadata after the Ndarray runtime "
                "has been reset"
            )
        program = impl.get_runtime().prog
        if program is None or program is not self._runtime_prog:
            raise RuntimeError("Ndarray belongs to another Taichi runtime")
        key = (id(self.arr), impl.current_cfg().arch, consumer, mode)
        cached = self._runtime_storage_arguments.get(key)
        if cached is not None:
            return cached
        described = _ti_core._describe_ndarray_storage(self.arr, "readwrite")
        if not described.ok:
            raise RuntimeError(
                "Cannot describe Ndarray runtime storage: "
                f"{described.reason}"
            )
        argument = _ti_core._make_runtime_storage_argument(
            program, described.descriptor, consumer, mode
        )
        qualification = argument.qualification
        if not qualification["bindable"] or not qualification["replayable"]:
            raise RuntimeError(
                "Ndarray runtime storage is not Graph eligible: "
                f"{qualification['reason']}"
            )
        if mode == "capture" and not qualification["capturable"]:
            raise RuntimeError(
                "Ndarray runtime storage is not Graph-capturable: "
                f"{qualification['reason']}"
            )
        self._runtime_storage_arguments[key] = argument
        return argument

    def _delete_runtime_ndarray(self):
        prog = self._runtime_prog
        arr = self.arr
        self._invalidate_runtime()
        if prog is not None and arr is not None:
            prog.delete_ndarray(arr)

    def __del__(self):
        if impl is not None:
            try:
                self._delete_runtime_ndarray()
            except Exception:
                pass

    def get_type(self):
        return NdarrayTypeMetadata(self.element_type, self.shape, self.grad is not None)

    @property
    def element_shape(self):
        """Gets ndarray element shape.

        Returns:
            Tuple[Int]: Ndarray element shape.
        """
        raise NotImplementedError()

    @python_scope
    def __setitem__(self, key, value):
        """Sets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.
            value (element type): Value to set.
        """
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key):
        """Gets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.

        Returns:
            element type: Value retrieved.
        """
        raise NotImplementedError()

    @python_scope
    def fill(self, val):
        """Fills ndarray with a specific scalar value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        fast_fill_archs = (
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
            _ti_core.Arch.x64,
        )
        if impl.current_cfg().arch not in fast_fill_archs:
            self._fill_by_kernel(val)
        elif self._can_fast_zero_fill(val):
            impl.get_runtime().prog.fill_uint(self.arr, 0)
        elif _ti_core.is_tensor(self.element_type) and not self._can_fast_scalar_fill(val):
            self._fill_by_kernel(val)
        elif self.dtype == primitive_types.f32:
            impl.get_runtime().prog.fill_float(self.arr, val)
        elif self.dtype == primitive_types.i32:
            impl.get_runtime().prog.fill_int(self.arr, val)
        elif self.dtype == primitive_types.u32:
            impl.get_runtime().prog.fill_uint(self.arr, val)
        elif self._try_fast_host_staging_fill(val):
            return
        else:
            self._fill_by_kernel(val)

    @python_scope
    def _can_fast_zero_fill(self, val):
        try:
            is_scalar = np.isscalar(val)
        except TypeError:
            return False
        if not is_scalar:
            return False
        if not isinstance(val, (bool, int, float, np.integer, np.floating)):
            return False
        try:
            if val != 0:
                return False
        except (TypeError, ValueError):
            return False
        if isinstance(val, (float, np.floating)) and np.signbit(val):
            return False
        return (self._get_nelement() * self._get_element_size()) % 4 == 0

    @python_scope
    def _can_fast_scalar_fill(self, val):
        try:
            is_scalar = np.isscalar(val)
        except TypeError:
            return False
        return is_scalar and isinstance(val, (bool, int, float, np.integer, np.floating))

    @python_scope
    def _try_fast_host_staging_fill(self, val):
        if self.dtype not in _NDARRAY_NATIVE_HOST_STAGING_FILL_DTYPES:
            return False
        if not self._can_fast_scalar_fill(val):
            return False
        try:
            arr = np.empty(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
            arr.fill(np.array(val, dtype=arr.dtype).item())
        except (TypeError, ValueError, OverflowError):
            return False
        if not self._can_fast_host_copy(arr, is_from_host=True):
            return False
        if _native_bulk_copy_needs_sync():
            impl.get_runtime().sync()
        impl.get_runtime().prog.copy_ndarray_from_host(self.arr, arr)
        return True

    @python_scope
    def _can_fast_host_copy(self, arr, is_from_host=False):
        arch = impl.current_cfg().arch
        fast_host_copy_archs = (
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
            _ti_core.Arch.x64,
        )
        if arch not in fast_host_copy_archs:
            return False
        if not isinstance(arr, np.ndarray) or not arr.flags.c_contiguous:
            return False
        if self.layout != Layout.AOS:
            return False
        return (
            arr.dtype == np.dtype(to_numpy_type(self.dtype))
            and tuple(self.arr.total_shape()) == tuple(arr.shape)
            and arr.nbytes == self._get_nelement() * self._get_element_size()
        )

    @python_scope
    def _ndarray_to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        arr = np.empty(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
        if self._can_fast_host_copy(arr):
            if _native_bulk_copy_needs_sync():
                impl.get_runtime().sync()
            impl.get_runtime().prog.copy_ndarray_to_host(self.arr, arr)
            return arr

        from taichi_forge._kernels import ndarray_to_ext_arr  # pylint: disable=C0415

        ndarray_to_ext_arr(self, arr)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_matrix_to_numpy(self, as_vector):
        """Converts matrix ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        arr = np.empty(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
        if self._can_fast_host_copy(arr):
            if _native_bulk_copy_needs_sync():
                impl.get_runtime().sync()
            impl.get_runtime().prog.copy_ndarray_to_host(self.arr, arr)
            return arr

        from taichi_forge._kernels import ndarray_matrix_to_ext_arr  # pylint: disable=C0415

        layout_is_aos = 1
        ndarray_matrix_to_ext_arr(self, arr, layout_is_aos, as_vector)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_from_numpy(self, arr):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.total_shape()) != tuple(arr.shape):
            raise ValueError(f"Mismatch shape: {tuple(self.arr.shape)} expected, but {tuple(arr.shape)} provided")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        if self._can_fast_host_copy(arr, is_from_host=True):
            if _native_bulk_copy_needs_sync():
                impl.get_runtime().sync()
            impl.get_runtime().prog.copy_ndarray_from_host(self.arr, arr)
            return

        from taichi_forge._kernels import ext_arr_to_ndarray  # pylint: disable=C0415

        ext_arr_to_ndarray(arr, self)
        impl.get_runtime().sync()

    @python_scope
    def _ndarray_matrix_from_numpy(self, arr, as_vector):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.total_shape()) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.total_shape())} expected, but {tuple(arr.shape)} provided"
            )
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        if self._can_fast_host_copy(arr, is_from_host=True):
            if _native_bulk_copy_needs_sync():
                impl.get_runtime().sync()
            impl.get_runtime().prog.copy_ndarray_from_host(self.arr, arr)
            return

        from taichi_forge._kernels import ext_arr_to_ndarray_matrix  # pylint: disable=C0415

        layout_is_aos = 1
        ext_arr_to_ndarray_matrix(arr, self, layout_is_aos, as_vector)
        impl.get_runtime().sync()

    @python_scope
    def _get_element_size(self):
        """Returns the size of one element in bytes.

        Returns:
            Size in bytes.
        """
        return self.arr.element_size()

    @python_scope
    def _get_nelement(self):
        """Returns the total number of elements.

        Returns:
            Total number of elements.
        """
        return self.arr.nelement()

    @python_scope
    def copy_from(self, other):
        """Copies all elements from another ndarray.

        The shape of the other ndarray needs to be the same as `self`.

        Args:
            other (Ndarray): The source ndarray.
        """
        assert isinstance(other, Ndarray)
        assert tuple(self.arr.shape) == tuple(other.arr.shape)
        if self._can_fast_copy_from(other) and _native_copy_ad_supported(other, self):
            impl.get_runtime().prog.copy_ndarray(self.arr, other.arr)
            if _native_bulk_copy_needs_sync():
                impl.get_runtime().sync()
            if not _record_native_copy_ad(other, self):
                raise RuntimeError(
                    "Native ndarray copy could not record an autodiff "
                    "backward launcher after a successful native forward copy."
                )
            return

        from taichi_forge._kernels import ndarray_to_ndarray  # pylint: disable=C0415

        ndarray_to_ndarray(self, other)
        impl.get_runtime().sync()

    @python_scope
    def _can_fast_copy_from(self, other):
        arch = impl.current_cfg().arch
        fast_copy_archs = (
            _ti_core.Arch.cuda,
            _ti_core.Arch.vulkan,
            _ti_core.Arch.x64,
        )
        if arch not in fast_copy_archs:
            return False
        return (
            self.dtype == other.dtype
            and self.element_shape == other.element_shape
            and self.layout == other.layout
            and tuple(self.arr.total_shape()) == tuple(other.arr.total_shape())
            and self._get_element_size() == other._get_element_size()
        )

    def _set_grad(self, grad):
        """Sets the gradient ndarray.

        Args:
            grad (Ndarray): The gradient ndarray.
        """
        self.grad = grad

    def __deepcopy__(self, memo=None):
        """Copies all elements to a new ndarray.

        Returns:
            Ndarray: The result ndarray.
        """
        raise NotImplementedError()

    def _fill_by_kernel(self, val):
        """Fills ndarray with a specific scalar value using a ti.kernel.

        Args:
            val (Union[int, float]): Value to fill.
        """
        raise NotImplementedError()

    @python_scope
    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key,)
        if len(key) != len(self.arr.total_shape()):
            raise TaichiIndexError(f"{len(self.arr.total_shape())}d ndarray indexed with {len(key)}d indices: {key}")
        return key

    @python_scope
    def _initialize_host_accessor(self):
        if self.host_accessor:
            return
        impl.get_runtime().materialize()
        self.host_accessor = NdarrayHostAccessor(self.arr)


class ScalarNdarray(Ndarray):
    """Taichi ndarray with scalar elements.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.
    """

    def __init__(self, dtype, arr_shape):
        super().__init__()
        self.dtype = cook_dtype(dtype)
        self.arr = impl.get_runtime().prog.create_ndarray(
            self.dtype, arr_shape, layout=Layout.NULL, zero_fill=True, dbg_info=_ti_core.DebugInfo(get_traceback())
        )
        self._register_runtime_object()
        self.shape = tuple(self.arr.shape)
        self.element_type = dtype

    @classmethod
    def _private_scratch_storage(cls, dtype, arr_shape):
        """Uninitialized provider scratch with ordinary runtime ownership.

        Internal callers must overwrite the full range before its first read.
        Public ndarray construction keeps its zero-initialization contract.
        """
        value = cls.__new__(cls)
        Ndarray.__init__(value)
        value.dtype = cook_dtype(dtype)
        value.arr = impl.get_runtime().prog.create_ndarray(
            value.dtype, arr_shape, layout=Layout.NULL, zero_fill=False
        )
        value._register_runtime_object()
        value.shape = tuple(value.arr.shape)
        value.element_type = dtype
        return value

    @classmethod
    def _graph_pool_storage(cls, dtype, arr_shape, pool):
        """Materialize private Graph storage in an owned native CUDA pool."""

        value = cls.__new__(cls)
        Ndarray.__init__(value)
        value.dtype = cook_dtype(dtype)
        value.arr = pool.create_ndarray(value.dtype, arr_shape)
        value._register_runtime_object()
        value.shape = tuple(value.arr.shape)
        value.element_type = dtype
        return value

    @classmethod
    def _graph_observation_storage(cls, dtype, arr_shape):
        """Allocate tiny completion-attached storage for Graph snapshots."""

        value = cls.__new__(cls)
        Ndarray.__init__(value)
        value.dtype = cook_dtype(dtype)
        value.arr = impl.get_runtime().prog._create_graph_observation_ndarray(
            value.dtype, arr_shape, layout=Layout.NULL
        )
        value._register_runtime_object()
        value.shape = tuple(value.arr.shape)
        value.element_type = dtype
        return value

    @property
    def element_shape(self):
        return ()

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessor()
        self.host_accessor.setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessor()
        return self.host_accessor.getter(*self._pad_key(key))

    @python_scope
    def to_numpy(self):
        return self._ndarray_to_numpy()

    @python_scope
    def from_numpy(self, arr):
        self._ndarray_from_numpy(arr)

    def __deepcopy__(self, memo=None):
        ret_arr = ScalarNdarray(self.dtype, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    def _fill_by_kernel(self, val):
        from taichi_forge._kernels import fill_ndarray  # pylint: disable=C0415

        fill_ndarray(self, val)

    def __repr__(self):
        return "<ti.ndarray>"


class StructNdarrayScalarMemberView:
    """A primitive scalar member view of a StructNdarray.

    The view is strided inside the parent AOS payload. It is intentionally not
    a normal Ndarray because existing native primitives assume contiguous
    element storage.
    """

    def __init__(self, base, name, dtype, offset, path=None, component=None):
        self.base = base
        self.path = tuple(path) if path is not None else (name,)
        self.component = _normalize_component_index(component)
        self.name = name
        self.dtype = cook_dtype(dtype)
        self.element_type = self.dtype
        self.shape = base.shape
        self.offset = int(offset)
        self.stride = _ti_core.data_type_size(base.dtype)
        self.element_size = _ti_core.data_type_size(self.dtype)
        self._native_host_copy_tmp = None
        self._native_host_copy_enabled = _struct_scalar_member_native_host_copy_worthwhile(self)

    @property
    def element_shape(self):
        return ()

    @python_scope
    def to_numpy(self):
        native = self._try_native_to_numpy()
        if native is not None:
            return native
        member = _extract_struct_numpy_member(self.base.to_numpy(), self.path)
        if self.component is not None:
            member = member[(...,) + self.component]
        return np.ascontiguousarray(member)

    @python_scope
    def from_numpy(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(arr.shape) != self.shape:
            raise ValueError(f"Mismatch shape: {self.shape} expected, but {tuple(arr.shape)} provided")
        expected_dtype = np.dtype(to_numpy_type(self.dtype))
        if arr.dtype != expected_dtype:
            raise TypeError(f"Mismatch dtype: {expected_dtype} expected, but {arr.dtype} provided")
        if self._try_native_from_numpy(arr):
            return
        struct_arr = self.base.to_numpy()
        member = _extract_struct_numpy_member(struct_arr, self.path)
        if self.component is not None:
            member[(...,) + self.component] = np.ascontiguousarray(arr)
        else:
            member[...] = np.ascontiguousarray(arr)
        self.base.from_numpy(struct_arr)

    def _native_host_copy_temp(self):
        if self._native_host_copy_tmp is None:
            self._native_host_copy_tmp = ScalarNdarray(self.dtype, self.shape)
        return self._native_host_copy_tmp

    def _try_native_to_numpy(self):
        if not self._native_host_copy_enabled:
            return None
        try:
            from taichi_forge.algorithms import experimental_transform  # pylint: disable=C0415

            tmp = self._native_host_copy_temp()
            experimental_transform(self, tmp)
            return tmp.to_numpy()
        except (RuntimeError, TypeError, ValueError):
            return None

    def _try_native_from_numpy(self, arr):
        if not self._native_host_copy_enabled:
            return False
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        try:
            from taichi_forge.algorithms import experimental_transform  # pylint: disable=C0415

            tmp = self._native_host_copy_temp()
            tmp.from_numpy(arr)
            experimental_transform(tmp, self)
            return True
        except (RuntimeError, TypeError, ValueError):
            return False

    def get_type(self):
        return NdarrayTypeMetadata(self.element_type, self.shape, False)

    def __repr__(self):
        return f"<ti.StructNdarrayScalarMemberView {self.name}: {self.dtype}>"


class StructNdarrayTensorMemberView:
    """A vector/matrix member view of a StructNdarray.

    The view exposes a full Taichi tensor element while preserving the parent
    AOS byte stride. It is intended for kernel arguments and host bulk IO;
    scalar-lane primitive calls should still use component=....
    """

    def __init__(self, base, name, dtype, offset, path=None):
        self.base = base
        self.path = tuple(path) if path is not None else (name,)
        self.component = None
        self.name = name
        self.dtype = dtype
        self.scalar_dtype = cook_dtype(dtype.dtype)
        self.element_type = dtype.tensor_type
        self.shape = base.shape
        self.offset = int(offset)
        self.stride = _ti_core.data_type_size(base.dtype)
        self.element_size = _ti_core.data_type_size(self.element_type)
        self._native_host_copy_tmp = None
        self._native_host_copy_tmp_view = None
        self._native_host_copy_enabled = _struct_tensor_member_native_host_copy_worthwhile(self)

    @property
    def element_shape(self):
        return tuple(self.dtype.get_shape())

    @python_scope
    def to_numpy(self):
        native = self._try_native_to_numpy()
        if native is not None:
            return native
        return np.ascontiguousarray(_extract_struct_numpy_member(self.base.to_numpy(), self.path))

    @python_scope
    def from_numpy(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        expected_shape = self.shape + self.element_shape
        if tuple(arr.shape) != expected_shape:
            raise ValueError(f"Mismatch shape: {expected_shape} expected, but {tuple(arr.shape)} provided")
        expected_dtype = np.dtype(to_numpy_type(self.dtype.dtype))
        if arr.dtype != expected_dtype:
            raise TypeError(f"Mismatch dtype: {expected_dtype} expected, but {arr.dtype} provided")
        if self._try_native_from_numpy(arr):
            return
        struct_arr = self.base.to_numpy()
        member = _extract_struct_numpy_member(struct_arr, self.path)
        member[...] = np.ascontiguousarray(arr)
        self.base.from_numpy(struct_arr)

    def _native_host_copy_temp_view(self):
        if self._native_host_copy_tmp is None:
            from taichi_forge.types import compound_types  # pylint: disable=C0415

            temp_type = compound_types.struct(value=self.dtype)
            self._native_host_copy_tmp = StructNdarray(temp_type, self.shape)
            self._native_host_copy_tmp_view = self._native_host_copy_tmp.field("value")
        return self._native_host_copy_tmp, self._native_host_copy_tmp_view

    def _try_native_to_numpy(self):
        if not self._native_host_copy_enabled:
            return None
        try:
            from taichi_forge.algorithms import experimental_transform  # pylint: disable=C0415

            tmp, tmp_view = self._native_host_copy_temp_view()
            experimental_transform(self, tmp_view)
            return np.ascontiguousarray(tmp.to_numpy()["value"])
        except (RuntimeError, TypeError, ValueError):
            return None

    def _try_native_from_numpy(self, arr):
        if not self._native_host_copy_enabled:
            return False
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        try:
            from taichi_forge.algorithms import experimental_transform  # pylint: disable=C0415

            tmp, tmp_view = self._native_host_copy_temp_view()
            struct_arr = np.empty(shape=self.shape, dtype=tmp.numpy_dtype)
            struct_arr["value"] = arr
            tmp.from_numpy(struct_arr)
            experimental_transform(tmp_view, self)
            return True
        except (RuntimeError, TypeError, ValueError):
            return False

    def get_type(self):
        return NdarrayTypeMetadata(self.element_type, self.shape, False)

    def __repr__(self):
        return f"<ti.StructNdarrayTensorMemberView {self.name}: {self.dtype}>"


class StructNdarray(Ndarray):
    """Taichi ndarray with structured AOS elements.

    This is intentionally limited to raw host/device copies for now. Kernel
    field access and field-wise arithmetic primitives are enabled separately.
    """

    def __init__(self, struct_type, arr_shape):
        super().__init__()
        self.struct_type = struct_type
        self.dtype = struct_type.dtype
        self.layout = Layout.AOS
        self.numpy_dtype = _struct_numpy_dtype(struct_type)
        nelement = int(np.prod(arr_shape, dtype=np.int64))
        byte_size = nelement * _ti_core.data_type_size(self.dtype)
        zero_fill_on_create = byte_size % 4 == 0
        self.arr = impl.get_runtime().prog.create_ndarray(
            self.dtype,
            arr_shape,
            layout=Layout.AOS,
            zero_fill=zero_fill_on_create,
            dbg_info=_ti_core.DebugInfo(get_traceback()),
        )
        self._register_runtime_object()
        self.shape = tuple(self.arr.shape)
        self.element_type = self.dtype
        self._device_write_pending = zero_fill_on_create and impl.current_cfg().arch == _ti_core.Arch.cuda
        self._member_view_cache = {}
        if not zero_fill_on_create:
            self.fill(0)

    @property
    def element_shape(self):
        return ()

    def get_type(self):
        return NdarrayTypeMetadata(self.struct_type, self.shape, False)

    @python_scope
    def __setitem__(self, key, value):
        raise TaichiRuntimeError("StructNdarray Python item assignment is not supported yet; use from_numpy().")

    @python_scope
    def __getitem__(self, key):
        raise TaichiRuntimeError("StructNdarray Python item access is not supported yet; use to_numpy().")

    @python_scope
    def field(self, name, component=None):
        path = _normalize_struct_member_path(name)
        component = _normalize_component_index(component)
        cache_key = (path, component)
        cached = self._member_view_cache.get(cache_key)
        if cached is not None:
            return cached
        member_dtype, member_np_dtype, offset = _resolve_struct_member_path(self.struct_type, self.numpy_dtype, path)
        member_name = _format_struct_member_path(path)
        if component is not None:
            component_offset = _resolve_matrix_component_offset(member_dtype, member_np_dtype, component)
            component_name = f"{member_name}[{','.join(str(i) for i in component)}]"
            view = StructNdarrayScalarMemberView(
                self, component_name, member_dtype.dtype, offset + component_offset, path=path, component=component
            )
            self._member_view_cache[cache_key] = view
            return view
        if _is_matrix_type(member_dtype):
            view = StructNdarrayTensorMemberView(self, member_name, member_dtype, offset, path=path)
            self._member_view_cache[cache_key] = view
            return view
        if member_dtype not in primitive_types.all_types:
            raise TypeError(
                "StructNdarray.field() currently supports primitive scalar leaves and vector/matrix members. "
                f"Member '{member_name}' has type {member_dtype}."
            )
        view = StructNdarrayScalarMemberView(self, member_name, member_dtype, offset, path=path)
        self._member_view_cache[cache_key] = view
        return view

    def _sync_pending_device_write(self):
        if self._device_write_pending:
            impl.get_runtime().sync()
            self._device_write_pending = False
            return True
        return False

    @python_scope
    def fill(self, val):
        try:
            is_zero = np.isscalar(val) and val == 0
        except (TypeError, ValueError):
            is_zero = False
        if not is_zero:
            raise TaichiRuntimeError("StructNdarray only supports zero fill for now.")
        byte_size = self._get_nelement() * self._get_element_size()
        if byte_size % 4 == 0:
            impl.get_runtime().prog.fill_uint(self.arr, 0)
            self._device_write_pending = impl.current_cfg().arch in (_ti_core.Arch.cuda, _ti_core.Arch.vulkan)
            return
        zeros = np.zeros(self.shape, dtype=self.numpy_dtype)
        self._sync_pending_device_write()
        impl.get_runtime().prog.copy_ndarray_from_host(self.arr, zeros)

    @python_scope
    def to_numpy(self):
        arr = np.empty(shape=self.shape, dtype=self.numpy_dtype)
        if not self._sync_pending_device_write() and _native_bulk_copy_needs_sync():
            impl.get_runtime().sync()
        impl.get_runtime().prog.copy_ndarray_to_host(self.arr, arr)
        return arr

    @python_scope
    def from_numpy(self, arr):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(arr.shape) != self.shape:
            raise ValueError(f"Mismatch shape: {self.shape} expected, but {tuple(arr.shape)} provided")
        if arr.dtype != self.numpy_dtype:
            raise TypeError(f"Mismatch dtype: {self.numpy_dtype} expected, but {arr.dtype} provided")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        self._sync_pending_device_write()
        impl.get_runtime().prog.copy_ndarray_from_host(self.arr, arr)

    @python_scope
    def to_numpy_fields(self, *names):
        if len(names) == 1:
            path = _normalize_struct_member_path(names[0])
            try:
                return {
                    _format_struct_member_path(path): self.field(path).to_numpy()
                }
            except TypeError:
                pass
        struct_arr = self.to_numpy()
        if len(names) == 0:
            names = tuple(self.struct_type.members.keys())
        result = {}
        for name in names:
            path = _normalize_struct_member_path(name)
            _resolve_struct_member_path(self.struct_type, self.numpy_dtype, path)
            result[_format_struct_member_path(path)] = np.ascontiguousarray(_extract_struct_numpy_member(struct_arr, path))
        return result

    @python_scope
    def from_numpy_fields(self, fields=None, **kwargs):
        updates = {}
        if fields is not None:
            if not isinstance(fields, dict):
                raise TypeError("StructNdarray.from_numpy_fields() expects a dict or keyword fields")
            updates.update(fields)
        updates.update(kwargs)
        if len(updates) == 0:
            return
        if len(updates) == 1:
            name, value = next(iter(updates.items()))
            path = _normalize_struct_member_path(name)
            try:
                self.field(path).from_numpy(np.asarray(value))
                return
            except TypeError:
                pass
        struct_arr = self.to_numpy()
        for name, value in updates.items():
            path = _normalize_struct_member_path(name)
            _resolve_struct_member_path(self.struct_type, self.numpy_dtype, path)
            target = _extract_struct_numpy_member(struct_arr, path)
            value = np.asarray(value)
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Mismatch shape for member '{_format_struct_member_path(path)}': "
                    f"{tuple(target.shape)} expected, but {tuple(value.shape)} provided"
                )
            if value.dtype != target.dtype:
                raise TypeError(
                    f"Mismatch dtype for member '{_format_struct_member_path(path)}': "
                    f"{target.dtype} expected, but {value.dtype} provided"
                )
            target[...] = np.ascontiguousarray(value)
        self.from_numpy(struct_arr)

    @python_scope
    def debug_getitem(self, key):
        return self.to_numpy()[key]

    @python_scope
    def debug_setitem(self, key, value):
        struct_arr = self.to_numpy()
        struct_arr[key] = value
        self.from_numpy(struct_arr)

    @python_scope
    def copy_from(self, other):
        if not isinstance(other, StructNdarray):
            raise TaichiRuntimeError("StructNdarray can only copy from another StructNdarray.")
        if self.dtype != other.dtype or self.shape != other.shape or self.numpy_dtype != other.numpy_dtype:
            raise TaichiRuntimeError("StructNdarray copy_from requires matching dtype and shape.")
        self._sync_pending_device_write()
        other._sync_pending_device_write()
        impl.get_runtime().prog.copy_ndarray(self.arr, other.arr)
        if _native_bulk_copy_needs_sync():
            impl.get_runtime().sync()
        self._device_write_pending = False

    def __deepcopy__(self, memo=None):
        ret_arr = StructNdarray(self.struct_type, self.shape)
        ret_arr.copy_from(self)
        return ret_arr

    def _fill_by_kernel(self, val):
        raise TaichiRuntimeError("StructNdarray kernel fill is not supported yet.")

    def __repr__(self):
        return "<ti.StructNdarray>"


class NdarrayHostAccessor:
    def __init__(self, ndarray):
        dtype = ndarray.element_data_type()
        if is_real(dtype):

            def getter(*key):
                return ndarray.read_float(key)

            def setter(value, *key):
                ndarray.write_float(key, value)

        else:
            if is_signed(dtype):

                def getter(*key):
                    return ndarray.read_int(key)

            else:

                def getter(*key):
                    return ndarray.read_uint(key)

            def setter(value, *key):
                ndarray.write_int(key, value)

        self.getter = getter
        self.setter = setter


class NdarrayHostAccess:
    """Class for accessing VectorNdarray/MatrixNdarray in Python scope.
    Args:
        arr (Union[VectorNdarray, MatrixNdarray]): See above.
        indices_first (Tuple[Int]): Indices of first-level access (coordinates in the field).
        indices_second (Tuple[Int]): Indices of second-level access (indices in the vector/matrix).
    """

    def __init__(self, arr, indices_first, indices_second):
        self.ndarr = arr
        self.arr = arr.arr
        self.indices = indices_first + indices_second

        def getter():
            self.ndarr._initialize_host_accessor()
            return self.ndarr.host_accessor.getter(*self.ndarr._pad_key(self.indices))

        def setter(value):
            self.ndarr._initialize_host_accessor()
            self.ndarr.host_accessor.setter(value, *self.ndarr._pad_key(self.indices))

        self.getter = getter
        self.setter = setter


__all__ = [
    "Ndarray",
    "ScalarNdarray",
    "StructNdarray",
    "StructNdarrayScalarMemberView",
    "StructNdarrayTensorMemberView",
]
