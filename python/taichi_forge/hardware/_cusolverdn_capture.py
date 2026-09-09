"""Optional fixed cuSOLVER recording, owned entirely by the provider.

No vendor calls, stream capture, pointer queries or instantiation occur in replay.
This is a root-ordered Graph command, not enclosing Graph fusion or a search axis.
"""

import ctypes as ct
from functools import partial

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.hardware._cusolverdn_abi import _function, check
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.lang.exception import TaichiRuntimeError


def _closed():
    raise TaichiRuntimeError("cuSOLVERDn recording is closed or its plan was retired")


class CusolverDnCapture:
    graph_runtime_lifetime_check_required = False

    def __init__(self, binding, mode):
        self.binding, self.mode = binding, mode
        self._closed = True
        self._exec = ct.c_void_p()
        plan = binding._plan
        driver, p = plan._driver.library, ct.c_void_p
        # Optional symbols are resolved only when capture is explicitly requested.
        create = _function(driver, "cuStreamCreate", [ct.POINTER(p), ct.c_uint])
        begin = _function(driver, "cuStreamBeginCapture", [p, ct.c_int])
        end = _function(driver, "cuStreamEndCapture", [p, ct.POINTER(p)])
        instantiate = _function(
            driver, "cuGraphInstantiateWithFlags", [ct.POINTER(p), p, ct.c_ulonglong]
        )
        destroy_graph = _function(driver, "cuGraphDestroy", [p])
        destroy_stream = _function(driver, "cuStreamDestroy_v2", [p])
        self._destroy = _function(driver, "cuGraphExecDestroy", [p])
        launch = _function(driver, "cuGraphLaunch", [p, p])
        stream, graph = p(), p()
        capturing = False
        # The handle/workspace must not still be executing while the cold vendor
        # stream setting is changed. This wait is preparation, never replay.
        plan._runtime_prog.synchronize()
        with plan._driver.activate():
            try:
                check(create(ct.byref(stream), 1), "create solver capture stream")
                check(
                    plan.provider._library.set_stream(plan._handle, stream),
                    "set solver capture stream",
                )
                check(begin(stream, 1), "begin solver capture")
                capturing = True
                binding._invoke(mode, stream)
                result = end(stream, ct.byref(graph))
                capturing = False
                check(result, "end solver capture")
                check(
                    instantiate(ct.byref(self._exec), graph, 0),
                    "instantiate solver capture",
                )
            except BaseException:
                if self._exec:
                    self._destroy(self._exec)
                    self._exec = p()
                raise
            finally:
                if capturing:
                    end(stream, ct.byref(graph))  # End invalid captures on failure too.
                if graph:
                    destroy_graph(graph)
                if stream:
                    destroy_stream(stream)
                check(
                    plan.provider._library.set_stream(plan._handle, None),
                    "restore solver stream",
                )
        self._closed = False
        self._launch = partial(launch, self._exec, None)
        self._solve_ready = partial(binding._submit, "solve")
        self._submit = self._run
        plan._captures.add(self)
        self._identity = _digest(
            (
                dict(plan.provider.identity),
                plan.rows,
                plan.rhs_count,
                str(plan.dtype),
                mode,
                "cuda-fixed-capture-v1",
            )
        )
        self._semantic_id = _digest(
            (
                "spd-lower-preserved-cholesky-v1",
                plan.rows,
                plan.rhs_count,
                str(plan.dtype),
                mode,
            )
        )

    closed = property(lambda self: self._closed)

    def validate_graph_lifetime(self):
        if self._closed or self.binding._plan.closed:
            _closed()

    def run(self):
        return self._submit()

    def _run(self):
        plan = self.binding._plan
        with plan._lock, plan._driver.activate():
            with external_cuda_submission(
                plan._runtime_prog, self.binding._resources
            ) as submission:
                check(submission.invoke(self._launch), "launch solver capture")
            if self.mode == "both":
                self.binding.solve = self._solve_ready
        return self.binding.solution

    def record(self, *, a="a", rhs="rhs", solution="solution"):
        """Publish an immutable root Graph command for this fixed binding."""
        self.validate_graph_lifetime()
        names = (a, rhs, solution)
        if (
            any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 3
        ):
            raise ValueError("solver Graph names must be nonempty and distinct")
        return _Recording(self, names).as_node()

    def _release(self):
        if not self._closed:
            with self.binding._plan._driver.activate():
                check(self._destroy(self._exec), "destroy solver capture")
            self._exec = ct.c_void_p()
            self._closed = True
            self._launch = _closed
            self._submit = _closed

    def memory_report(self):
        """Shared plan requests; CUDA graph/driver residency remains unknown."""
        return self.binding._plan.memory_report()

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_identity(self):
        return ("cusolverdn", id(self.binding._plan))

    def close(self):
        with self.binding._plan._lock:
            if not self._closed:
                self.binding._plan._runtime_prog.synchronize()
                self._release()

    def __enter__(self):
        self.validate_graph_lifetime()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Recording(BackendCommandRecording):
    def __init__(self, capture, names):
        super().__init__(
            backend="cuda",
            binding_names=names,
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="native_replay",
        )
        object.__setattr__(self, "capture", capture)
        object.__setattr__(self, "_graph_semantic_fingerprint", capture._semantic_id)
        object.__setattr__(self, "_graph_physical_plan_id", capture._identity)

    source = property(lambda self: self.capture)

    @property
    def resource_effects(self):
        binding = self.capture.binding
        return (
            ResourceEffect(self.binding_names[0], GraphAccess.READ),
            ResourceEffect(
                self.binding_names[1],
                (
                    GraphAccess.READ_WRITE
                    if binding._rhs == binding._output
                    else GraphAccess.READ
                ),
            ),
            ResourceEffect(self.binding_names[2], GraphAccess.WRITE),
        )

    def validate_graph_bindings(self, bindings):
        for name, original in zip(
            self.binding_names, self.capture.binding._resources[:3]
        ):
            if bindings[name] is not original:
                raise TaichiRuntimeError(
                    "cuSOLVERDn recording requires its original fixed arrays"
                )

    def execute(self, bindings):
        self.capture.run()

    def as_node(self):
        return native_recording_node(
            self,
            lifetime_leases=(self.capture,),
            debug_info={
                "kind": "cusolverdn",
                "graph_integration": "root_ordered",
                "vendor_execution": "retained_cuda_graph",
            },
            publish_time_binding_validation_stable=True,
        )
