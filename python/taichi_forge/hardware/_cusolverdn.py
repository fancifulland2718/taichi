"""Device-resident explicit dense Cholesky; no implicit solver or Graph route."""

import ctypes as ct
from functools import partial
import threading
from types import MappingProxyType
import weakref

from taichi_forge.hardware._cusolverdn_abi import (
    ABI,
    DriverBinding,
    check,
    load_library,
    passive_status,
    probe_provider,
    resolve_library_path,
)
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import validate_runtime_generation
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32, f64, i32, u8


def _closed(*args):
    raise TaichiRuntimeError("cuSOLVERDn binding is closed or its runtime was reset")


def _unfactored():
    raise TaichiRuntimeError(
        "submit factor() before solve(), or use factor_and_solve()"
    )


def _size(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= 0x7FFFFFFF
    ):
        raise ValueError(f"{name} must be an integer in [1, INT_MAX]")
    return value


def _pointer(value, dtype, shape, name, program):
    if (
        not isinstance(value, Ndarray)
        or value.dtype != dtype
        or value.element_shape != ()
    ):
        raise TypeError(f"{name} must be a scalar {dtype} Taichi ndarray")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.arr is None or value._runtime_prog is not program:
        raise TaichiRuntimeError(f"{name} belongs to a different or reset runtime")
    return int(program.get_ndarray_data_ptr_as_int(value.arr))


class CusolverDnProvider:
    """User-installed dense-solver library; each plan owns its vendor handle."""

    def __init__(self, library_path=None):
        program = impl.get_runtime().prog
        if program is None or active_backend() != "cuda":
            raise TaichiRuntimeError(
                "CusolverDnProvider requires an initialized CUDA runtime"
            )
        self._library = load_library(library_path)
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._plans = weakref.WeakSet()
        self._lock = threading.RLock()
        self.closed = False
        self.identity = MappingProxyType(
            {
                "provider_abi": ABI,
                "provider_version": self._library.version,
                "vendor_library": self._library.path,
            }
        )
        impl.get_runtime().register_runtime_object(self)

    def cholesky_plan(self, rows, *, rhs_count=1, dtype=f32):
        with self._lock:
            if self.closed:
                raise TaichiRuntimeError("CusolverDnProvider has been closed")
            validate_runtime_generation(
                self, "CusolverDnProvider belongs to an old runtime"
            )
            return CusolverDnCholeskyPlan(self, rows, rhs_count=rhs_count, dtype=dtype)

    def close(self):
        with self._lock:
            if self.closed:
                return
            if any(not plan.closed for plan in self._plans):
                raise TaichiRuntimeError("close cuSOLVERDn plans before their provider")
            self.closed = True

    def _invalidate_runtime(self):
        with self._lock:
            for plan in tuple(self._plans):
                plan.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CusolverDnCholeskyPlan:
    """Fixed 2D f32/f64 SPD factorization and repeated RHS solve.

    A's lower triangle is meaningful. A is preserved; its private factor is
    row-major lower (the vendor sees column-major upper). RHS/output are (n,)
    for one vector, otherwise (rhs_count, n): each row is one RHS vector, so
    cuSOLVER needs no transpose or host staging. Numerical validity is reported
    through device info, not implicit host checks. One live binding per plan.
    """

    def __init__(self, provider, rows, *, rhs_count=1, dtype=f32):
        if provider.closed:
            raise TaichiRuntimeError("CusolverDnProvider has been closed")
        validate_runtime_generation(
            provider, "CusolverDnProvider belongs to an old runtime"
        )
        self.rows, self.rhs_count = _size(rows, "rows"), _size(rhs_count, "rhs_count")
        if dtype not in (f32, f64):
            raise TypeError("cuSOLVERDn Cholesky dtype must be f32 or f64")
        self.dtype = dtype
        self.provider = provider
        self._runtime_prog = provider._runtime_prog
        self._lock = provider._lock
        self._binding = None
        self._handle, self._params = ct.c_void_p(), ct.c_void_p()
        self.closed = False
        self._factor = ScalarNdarray(dtype, (rows, rows))
        self._info = ScalarNdarray(i32, (2,))
        self._info.fill(-1)
        program = self._runtime_prog
        self._factor_pointer = _pointer(
            self._factor, dtype, (rows, rows), "factor", program
        )
        self._info_pointer = _pointer(self._info, i32, (2,), "status", program)
        self._driver = DriverBinding(self._factor_pointer)
        self._scalar = 0 if dtype == f32 else 1
        self._element_bytes = 4 if dtype == f32 else 8
        api = provider._library
        try:
            with self._driver.activate():
                check(api.create(ct.byref(self._handle)), "cusolverDnCreate")
                check(api.set_stream(self._handle, None), "cusolverDnSetStream")
                check(
                    api.create_params(ct.byref(self._params)), "cusolverDnCreateParams"
                )
                device, host = ct.c_size_t(), ct.c_size_t()
                check(
                    api.buffer_size(
                        self._handle,
                        self._params,
                        1,
                        rows,
                        self._scalar,
                        self._factor_pointer,
                        rows,
                        self._scalar,
                        ct.byref(device),
                        ct.byref(host),
                    ),
                    "cusolverDnXpotrf_bufferSize",
                )
            self.workspace_bytes, self.host_workspace_bytes = device.value, host.value
            self._workspace = (
                ScalarNdarray(u8, (device.value,)) if device.value else None
            )
            self._workspace_pointer = (
                _pointer(self._workspace, u8, (device.value,), "workspace", program)
                if device.value
                else 0
            )
            self._host_workspace = (
                ct.create_string_buffer(host.value) if host.value else None
            )
            self._factor_call = partial(
                api.factor,
                self._handle,
                self._params,
                1,
                rows,
                self._scalar,
                self._factor_pointer,
                rows,
                self._scalar,
                self._workspace_pointer,
                device.value,
                self._host_workspace,
                host.value,
                self._info_pointer,
            )
            provider._plans.add(self)
        except Exception:
            with self._driver.activate():
                if self._params:
                    api.destroy_params(self._params)
                if self._handle:
                    api.destroy(self._handle)
            self._params = self._handle = None
            self._driver.close()
            self.closed = True
            raise

    @property
    def info(self):
        """Device i32[2]: factor status, solve status. No readback performed."""
        return self._info

    def bind(self, a, rhs, solution):
        with self._lock:
            if self.closed:
                raise TaichiRuntimeError("cuSOLVERDn plan has been closed")
            validate_runtime_generation(
                self.provider, "cuSOLVERDn plan belongs to an old runtime"
            )
            if self._binding is not None:
                raise TaichiRuntimeError(
                    "a cuSOLVERDn plan has one immutable binding; create another plan to rebind"
                )
            shape = (self.rows,) if self.rhs_count == 1 else (self.rhs_count, self.rows)
            pointers = (
                _pointer(
                    a, self.dtype, (self.rows, self.rows), "A", self._runtime_prog
                ),
                _pointer(rhs, self.dtype, shape, "RHS", self._runtime_prog),
                _pointer(solution, self.dtype, shape, "solution", self._runtime_prog),
            )
            if pointers[0] in pointers[1:]:
                raise ValueError("A must not alias RHS or solution")
            self._binding = CusolverDnBinding(self, a, rhs, solution, pointers)
            return self._binding

    def status(self):
        """Explicit synchronized status query; info remains usable by GPU kernels."""
        with self._lock:
            if self.closed:
                raise TaichiRuntimeError("cuSOLVERDn plan has been closed")
            factor, solve = map(int, self._info.to_numpy())
            return MappingProxyType(
                {
                    "factor_info": factor,
                    "solve_info": solve,
                    "factor_ok": factor == 0,
                    "solve_ok": factor == 0 and solve == 0,
                }
            )

    def memory_report(self):
        resident = not self.closed
        components = tuple(
            HardwareMemoryComponent(
                name, size, True, "provider_generation", "provider", resident=resident
            )
            for name, size in (
                ("factor", self.rows * self.rows * self._element_bytes),
                ("device_workspace", self.workspace_bytes),
                ("device_status", 8),
            )
        )
        return make_memory_report(
            "cusolverdn",
            "cuda",
            (
                *components,
                HardwareMemoryComponent(
                    "vendor_context_and_driver",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=resident,
                ),
            ),
            ownership_scope="plan; excludes caller arrays and host workspace",
            lifecycle_state="closed" if self.closed else "ready",
        )

    def close(self):
        with self._lock:
            if self.closed:
                return
            # One cold retirement wait covers vendor handle, host workspace and
            # device factors. Nothing is freed while queued work still uses it.
            self._runtime_prog.synchronize()
            if self._binding is not None:
                self._binding._invalidate()
            with self._driver.activate():
                check(
                    self.provider._library.destroy_params(self._params),
                    "cusolverDnDestroyParams",
                )
                self._params = None
                check(self.provider._library.destroy(self._handle), "cusolverDnDestroy")
                self._handle = None
            self._driver.close()
            self.closed = True
            self._factor_call = _closed
            self._factor = self._workspace = self._info = self._host_workspace = None
            self._binding = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CusolverDnBinding:
    def __init__(self, plan, a, rhs, solution, pointers):
        self._plan = plan
        self.solution = solution
        self._resources = (a, rhs, solution, plan._factor, plan._info) + (
            (plan._workspace,) if plan._workspace is not None else ()
        )
        self._a, self._rhs, self._output = pointers
        self._solve_call = partial(
            plan.provider._library.solve,
            plan._handle,
            plan._params,
            1,
            plan.rows,
            plan.rhs_count,
            plan._scalar,
            plan._factor_pointer,
            plan.rows,
            plan._scalar,
            self._output,
            plan.rows,
            plan._info_pointer + 4,
        )
        self.factor = partial(self._submit, "factor")
        self.solve = _unfactored
        self.factor_and_solve = partial(self._submit, "both")

    def _invoke(self, mode):
        plan = self._plan
        with plan._driver.activate():
            if mode != "solve":
                check(
                    plan._driver.clear(plan._info_pointer + 4, 0xFFFFFFFF, 1, None),
                    "invalidate previous solve status",
                )
                check(
                    plan._driver.copy(
                        plan._factor_pointer,
                        self._a,
                        plan.rows * plan.rows * plan._element_bytes,
                        None,
                    ),
                    "copy Cholesky input",
                )
                check(plan._factor_call(), "cusolverDnXpotrf")
            if mode != "factor":
                if self._rhs != self._output:
                    check(
                        plan._driver.copy(
                            self._output,
                            self._rhs,
                            plan.rows * plan.rhs_count * plan._element_bytes,
                            None,
                        ),
                        "copy RHS",
                    )
                check(self._solve_call(), "cusolverDnXpotrs")

    def _submit(self, mode):
        with self._plan._lock:
            return self._run(mode)

    def _run(self, mode):
        with external_cuda_submission(
            self._plan._runtime_prog, self._resources
        ) as submission:
            submission.invoke(self._invoke, mode)
        if mode != "solve":
            self.solve = partial(self._submit, "solve")
        return self.solution

    def _invalidate(self):
        self.factor = self.solve = self.factor_and_solve = _closed
        self._solve_call = _closed
        self._invoke = _closed
        self._run = _closed
        self._resources = ()


__all__ = ["CusolverDnProvider", "CusolverDnCholeskyPlan", "CusolverDnBinding"]
