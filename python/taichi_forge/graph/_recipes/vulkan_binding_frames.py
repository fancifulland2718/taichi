"""Cold immutable binding frames for mixed Vulkan kernel/provider Graphs."""

from dataclasses import replace
import weakref


def eligible(spec, backend):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode, _CompiledNativeGraphNode
    from taichi_forge.hardware._vulkan_fft import _Recording
    from taichi_forge.lang import impl

    config = impl.current_cfg()
    if (
        backend != "vulkan"
        or config.arch != core.Arch.vulkan
        or config.debug
        or config.kernel_profiler
        or spec.snode_tree_dependency_info
        or spec.runtime_lifetime_leases
        or not hasattr(core, "_prepare_vulkan_graph_recording")
    ):
        return False
    if not spec.nodes:
        return False
    for node in spec.nodes:
        if isinstance(node, _CompiledCGraphNode):
            if (
                node.temporary_actions
                or node.parallel_dispatch_groups
                or node.source_native_count
                or any(op[0] != "dispatch" for op in node.recipe_operations)
            ):
                return False
        elif isinstance(node, _CompiledNativeGraphNode):
            recording = getattr(node.executable, "_recording", None)
            if not isinstance(
                recording, _Recording
            ) or not recording.plan._statistics.get("inline_recording_available"):
                return False
        else:
            return False
    return True


class VulkanBindingFrameExecutor:
    execution_kind = "vulkan_prepared_binding_graph"
    physical_submission_mode = "vulkan_secondary_immutable_argument_frames"

    def __init__(self, instance):
        from taichi_forge._lib import core
        from taichi_forge.graph._graph import _CompiledCGraphNode, _GraphRunContext
        from taichi_forge.lang import impl

        spec = instance.spec
        if not eligible(spec, "vulkan"):
            raise ValueError(
                "Vulkan binding frames require fixed buffer dispatches and inline-recordable FFT plans"
            )
        self._program = impl.get_runtime().prog
        self._prepare = core._prepare_vulkan_graph_recording
        self._sources = []
        for node in spec.nodes:
            if isinstance(node, _CompiledCGraphNode):
                self._sources.append(node.compiled_graph)
            else:
                recording = node.executable._recording
                recording.plan.validate_graph_lifetime()
                self._sources.append(
                    self._program._vulkan_fft_graph_command(
                        recording.plan._handle, recording.data
                    )
                )
        self._frames = weakref.WeakSet()
        self._context = _GraphRunContext()
        self._dispatch_count = spec.dispatch_count
        self._task_count = sum(
            len(stage["tasks"]) for stage in spec.pipeline_definition
        )

    def prewarm(self):
        return self

    def _frame(self, arguments, flattened=None):
        self._context.begin(arguments, flattened_args=flattened)
        try:
            frame = self._prepare(
                self._program, self._sources, self._context.flattened_args()
            )
        finally:
            self._context.end()
        # Materialization/binding boundary only: never silently change the
        # physical recipe into the primary-submit compatibility path.
        if not frame.uses_secondary_commands():
            frame.close()
            raise ValueError("Vulkan recipe requires embeddable secondary commands")
        self._frames.add(frame)
        return frame

    def prepare_binding_version(self, version):
        if not version.fast_path_qualified:
            raise ValueError(
                "Vulkan immutable frames require publication-qualified owned bindings"
            )
        frame = self._frame(version.execution_arguments, version.flattened_args)
        return replace(
            version,
            execution_frame=frame,
            fast_path_qualified=True,
            volatile_reasons=(),
        )

    def run_prepared(self, invocation):
        if invocation.binding_version is not None:
            invocation.binding_version.execution_frame.run()
        else:
            # A raw mapping explicitly includes cold preparation. Published
            # Graph.bind versions own the amortized, upload-free replay path.
            frame = self._frame(invocation.arguments, invocation.flattened_args)
            try:
                frame.run()
            finally:
                frame.close()

    def invalidate_runtime(self, preserve_executables=False):
        for frame in tuple(self._frames):
            frame.close()
        self._frames.clear()
        self._sources.clear()
        self._context = None

    @property
    def snapshot_graph_stats(self):
        from taichi_forge.graph._graph import _empty_backend_stats

        result = _empty_backend_stats()
        result.update(
            backend="vulkan",
            last_path="vulkan_prepared_binding_plan",
            diagnostics_counters_complete=False,
            known_compiled_dispatches=self._dispatch_count,
            known_compiled_tasks=self._task_count,
            known_persistent_argument_bytes=sum(
                frame.argument_bytes() for frame in self._frames
            ),
        )
        return result

    @property
    def debug_graph_stats(self):
        return self.snapshot_graph_stats
