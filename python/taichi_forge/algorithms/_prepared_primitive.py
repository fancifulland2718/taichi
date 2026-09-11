"""Fixed-binding native primitives, without data-mutating preparation."""

import hashlib
import json

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError


class _PreparedNativePlan:
    """Shared fixed-binding lifetime and recording, not a new dispatch registry."""

    def _validate_lifetime(self):
        if self._command is None:
            raise TaichiRuntimeError(f"{type(self).__name__} is closed")
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError(f"{type(self).__name__} belongs to another runtime")

    def run(self):
        """Execute on current contents without host readback."""
        self._validate_lifetime()
        self._workspace_bytes = self._execute(self._command)

    def close(self):
        """Drop bindings without clearing shared scratch or waiting for GPU work."""
        self._command = None
        self._owners = ()
        self._execute = None
        self._program = None

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *_):
        self.close()

    def report(self):
        return {
            **self._description,
            "physical_plan_id": self._physical_id,
            "closed": self._command is None,
            "binding_policy": "fixed_dense_storage",
            "replay_mode": "rerecord",
            "stream_binding": "runtime_ordered",
            "workspace_owner": "program_primitive_arena",
            "workspace_preparation": "lazy_first_execution",
            "workspace_bytes_last_observed": self._workspace_bytes,
            "device_capture": False,
        }

    def record(self):
        """Return a runtime-ordered root-Graph action, not a capture recipe.

        The action retains these fixed bindings. Explicitly closing this plan
        invalidates its recordings; an enclosing Graph may use segmented or
        ordinary execution.
        """
        self._validate_lifetime()
        return _native_recording(self)


class PreparedSortPlan(_PreparedNativePlan):
    """A stable ascending in-place sort over fixed CUDA/Vulkan dense storage.

    Keys and optional payload are equally sized, compact 1D scalar ranges with
    i32/u32/f32/i64/u64/f64 elements. Ndarrays and qualified dense field/views
    share this path; no field-to-ndarray copy is performed. Equal keys preserve
    payload order. Vulkan supports ``nan_policy="last"`` only; CUDA also
    supports ``"bitwise"`` (the existing sortable-bit order). Other algorithms, descending order and raw provider
    selection are deliberately not options of this plan.

    Preparation validates layout, access and lifetime without sorting, issuing
    GPU work, synchronizing or allocating scratch. Backend pipelines/workspace
    remain lazy and Program-owned, so the first run is not a steady-state timing
    sample. Native workspace growth can synchronize as in ordinary sort.

    Contents may change in place between runs. Replacement, shape/layout change
    or runtime reset requires a new plan. ``close()`` invalidates this plan and
    recordings made from it, but does not clear shared Program scratch or wait
    for the GPU. Existing runtime completion owns in-flight allocation release.
    """

    def __init__(self, keys, values=None, *, nan_policy="last"):
        if nan_policy not in ("last", "bitwise"):
            raise ValueError("Prepared sort nan_policy must be 'last' or 'bitwise'")
        owners = (keys,) if values is None else (keys, values)
        descriptions = tuple(describe_storage(value, access="readwrite") for value in owners)
        for name, description in zip(("keys", "values"), descriptions):
            if not description.supported:
                raise TaichiRuntimeError(
                    f"Prepared sort {name} is not supported dense storage: {description.failure_reason}"
                )
        program = impl.get_runtime().prog
        prepare = getattr(program, "_prepare_primitive_sort", None)
        if prepare is None:
            raise TaichiRuntimeError("Prepared sort requires native prepared-primitive support")
        command = prepare(
            descriptions[0].descriptor,
            None if values is None else descriptions[1].descriptor,
            0 if nan_policy == "last" else 1,
        )
        self._program = program
        self._owners = owners
        self._command = command
        self._execute = program._execute_primitive_sort
        self._workspace_bytes = None
        self._backend = "cuda" if impl.current_cfg().arch == _ti_core.Arch.cuda else "vulkan"
        self._description = {
            "kind": "stable_radix_sort",
            "backend": self._backend,
            "count": int(descriptions[0].descriptor.index_shape[0]),
            "key_dtype": str(descriptions[0].descriptor.scalar_type),
            "value_dtype": None if values is None else str(descriptions[1].descriptor.scalar_type),
            "nan_policy": nan_policy,
            "ascending": True,
            "stable": True,
        }
        canonical = json.dumps(self._description, sort_keys=True, separators=(",", ":"))
        self._physical_id = "prepared-native-sort-v1:" + hashlib.sha256(canonical.encode()).hexdigest()
        self._access = ("read_write",) * len(owners)
        self._workspace_effect = "forge-native-sort-workspace"


def prepare_sort(keys, values=None, *, nan_policy="last"):
    """Prepare a fixed-binding stable sort; see :class:`PreparedSortPlan`."""
    return PreparedSortPlan(keys, values, nan_policy=nan_policy)


def _native_recording(plan):
    # Import the existing generic recording seam only when requested. Ordinary
    # algorithms import neither Graph nor hardware provider discovery.
    from taichi_forge.graph._ir import GraphAccess, ResourceEffect
    from taichi_forge.graph._native import BackendCommandRecording
    from taichi_forge.hardware._native_adapter import native_recording_node

    class PrimitiveRecording(BackendCommandRecording):
        def __init__(self):
            super().__init__(
                backend=plan._backend,
                binding_names=(),
                command_count=1,
                barrier_policy="internal",
                workspace_ownership="provider_generation",
                replay_mode="rerecord",
            )
            object.__setattr__(self, "_plan", plan)
            # Freeze effects before close() can release the plan's references.
            object.__setattr__(
                self,
                "_effects",
                tuple(
                    ResourceEffect(value, GraphAccess(access), runtime_bound=False)
                    for value, access in zip(plan._owners, plan._access)
                )
                + (ResourceEffect(plan._workspace_effect, GraphAccess.READ_WRITE, runtime_bound=False),),
            )
            object.__setattr__(self, "_graph_physical_plan_id", plan._physical_id)

        @property
        def resource_effects(self):
            return self._effects

        def execute(self, bindings):
            self._plan.run()

        def prepare_graph_execute(self, bindings):
            self._plan._validate_lifetime()
            return self._plan.run

        def _as_graph_native_node(self):
            return native_recording_node(
                self,
                runtime_bindings=(),
                lifetime_leases=(self._plan,),
                debug_info=self._plan._description,
            )

    return PrimitiveRecording()


__all__ = ["PreparedSortPlan", "prepare_sort"]
