"""FidelityFX Parallel Sort with fixed Forge storage and retained GPU commands."""

from functools import partial

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
)
from taichi_forge.hardware._parallel_sort_jit import compile_shaders
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import u32


class VulkanParallelSortPlan:
    """Ascending stable sort of fixed 1D u32 keys and optional u32 payload.

    ``compiler_path`` explicitly names DXC with SPIR-V support. Forge ships
    the MIT source; compilation, pipelines and workspace are prepared here,
    never in replay. No full FidelityFX framework or CUDA toolkit is needed.
    ``run()`` is asynchronous on the Forge queue; caller arrays are modified
    in place. Root Graph recording retains this plan and its exact storage.
    This is an expert execution API, not a CompileIQ fixed-sort routing axis.
    """

    graph_runtime_lifetime_check_required = False

    def __init__(self, keys, values=None, *, compiler_path):
        self._closed = True
        for array in (keys,) if values is None else (keys, values):
            if (
                not isinstance(array, ScalarNdarray)
                or array.dtype != u32
                or len(array.shape) != 1
                or array.shape[0] <= 0
            ):
                raise ValueError(
                    "Parallel Sort requires nonempty scalar 1D u32 ndarrays"
                )
        if values is keys or (values is not None and values.shape != keys.shape):
            raise ValueError(
                "Parallel Sort payload must be distinct and have the keys' shape"
            )
        program = impl.get_runtime().prog
        if program is None or active_backend() != "vulkan":
            raise RuntimeError("Parallel Sort requires an initialized Vulkan backend")
        create = getattr(program, "_create_vulkan_parallel_sort_plan", None)
        if create is None:
            raise RuntimeError(
                "Parallel Sort native bridge is unavailable in this runtime build"
            )
        shaders, facts = compile_shaders(compiler_path, values is not None)
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._keys, self._values = keys, values
        self._handle = create(keys.arr, None if values is None else values.arr, shaders)
        self._closed = False
        try:
            self._statistics = dict(
                program._vulkan_parallel_sort_plan_statistics(self._handle)
            )
            self._jit_facts = facts
            self._semantic_id = _digest(
                ("stable-ascending-u32-sort-v1", keys.shape, values is not None)
            )
            self._physical_id = _digest((self._semantic_id, self._statistics, facts))
            self._submit = partial(program._vulkan_parallel_sort_execute, self._handle)
        except BaseException:
            self.close()
            raise

    closed = property(lambda self: self._closed)

    def run(self):
        """Enqueue the retained GPU sequence, without input readback or wait."""
        self._submit()

    def validate_graph_lifetime(self):
        if self.closed or not runtime_generation_matches(self):
            raise TaichiRuntimeError(
                "Parallel Sort plan is closed or belongs to a previous runtime generation"
            )

    def record(self, *, keys="keys", values="values"):
        """Root Graph node; fixed bindings are checked only at publication."""
        self.validate_graph_lifetime()
        names = (keys,) if self._values is None else (keys, values)
        if any(not isinstance(name, str) or not name for name in names) or len(
            set(names)
        ) != len(names):
            raise ValueError(
                "Parallel Sort binding names must be nonempty and distinct"
            )
        return _Recording(self, names).as_node()

    def statistics(self):
        return {
            **self._statistics,
            **self._jit_facts,
            "physical_plan_id": self._physical_id,
            "graph_integration": "root_ordered",
            "gpu_sequence": "retained_secondary_commands",
            "compileiq_search": "not_exposed_fixed_sort_route",
        }

    def memory_report(self):
        valid = runtime_generation_matches(self)
        return make_memory_report(
            "fidelityfx_parallel_sort",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "workspace_requested_bytes",
                    self._statistics["workspace_bytes"],
                    True,
                    "provider_generation",
                    "provider",
                    resident=valid and not self.closed,
                ),
                HardwareMemoryComponent(
                    "driver_objects_and_pending_commands",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=valid,
                ),
            ),
            lifecycle_state=(
                "closed" if self.closed else "ready" if valid else "runtime_invalid"
            ),
            ownership_scope="requested workspace excludes caller storage and opaque driver allocations; close is not a retirement observation",
        )

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_identity(self):
        return ("fidelityfx_parallel_sort", self._runtime_generation, self._handle)

    def close(self):
        if not self.closed:
            self._runtime_prog._destroy_vulkan_parallel_sort_plan(self._handle)
            self._closed = True

    def __enter__(self):
        self.validate_graph_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()


class _Recording(BackendCommandRecording):
    def __init__(self, plan, names):
        super().__init__(
            backend="vulkan",
            binding_names=names,
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="native_replay",
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_graph_semantic_fingerprint", plan._semantic_id)
        object.__setattr__(self, "_graph_physical_plan_id", plan._physical_id)

    source = property(lambda self: self.plan)

    @property
    def resource_effects(self):
        return tuple(
            ResourceEffect(name, GraphAccess.READ_WRITE) for name in self.binding_names
        )

    def validate_graph_bindings(self, bindings):
        for name, array in zip(
            self.binding_names, (self.plan._keys, self.plan._values)
        ):
            if bindings[name] is not array:
                raise TaichiRuntimeError(
                    "Parallel Sort Graph requires its original ndarray bindings"
                )

    def execute(self, bindings):
        self.plan.run()

    def as_node(self):
        return native_recording_node(
            self,
            lifetime_leases=(self.plan,),
            debug_info={
                "kind": "fidelityfx_parallel_sort",
                "graph_integration": "root_ordered",
                "gpu_sequence": "retained_secondary_commands",
            },
            publish_time_binding_validation_stable=True,
        )
