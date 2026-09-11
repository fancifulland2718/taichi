"""Whole-Graph immutable argument-image and executable-reuse recipes."""

from dataclasses import replace

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


def _eligible(spec, backend):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import _CompiledCGraphNode
    from taichi_forge.lang import impl

    if backend != "cuda" or impl.current_cfg().arch != core.Arch.cuda:
        return False
    native = getattr(core, "_CudaGraphBindingExecutor", None)
    if native is None or not native.available():
        return False
    retained_events = getattr(native, "retains_completion_events_until_close", None)
    if retained_events is None or not retained_events():
        return False
    if spec._texture_binding_requirements:
        supports_textures = getattr(native, "supports_sampled_texture_bindings", None)
        if supports_textures is None or not supports_textures():
            return False
        if any(arg.tag != core.ArgKind.TEXTURE for arg in spec._texture_binding_requirements):
            return False
    config = impl.current_cfg()
    if config.debug or config.kernel_profiler or len(spec.nodes) != 1:
        return False
    node = spec.nodes[0]
    if not isinstance(node, _CompiledCGraphNode):
        return False
    if getattr(node, "source_native_count", 0):
        supports_commands = getattr(native, "supports_capture_commands", None)
        if supports_commands is None or not supports_commands():
            return False
        # Frozen definitions expose sources; materialized segments expose their
        # actual execution leases. Neither discovery path creates vendor plans.
        sources = tuple(
            getattr(operation[1], "_recording", None)
            for operation in node.recipe_operations
            if operation[0] == "native"
        ) or tuple(
            lease for lease in node.lifetime_leases if getattr(lease, "_graph_binding_frame_capture_safe", False)
        )
        if len(sources) != node.source_native_count or not all(
            getattr(source, "_graph_binding_frame_capture_safe", False) for source in sources
        ):
            return False
    return (
        isinstance(node, _CompiledCGraphNode)
        and spec.needs_runtime_args
        and not spec.snode_tree_dependency_info
        and not node.temporary_actions
        and not node.parallel_dispatch_groups
        and all(operation[0] in ("dispatch", "native") for operation in node.recipe_operations)
    )


class _BindingFrameExecutor:
    """Installed once; public binding versions own the immutable native frames."""

    execution_kind = "cuda_prepared_binding_graph"
    physical_submission_mode = "cuda_immutable_argument_frames_exec_reuse"

    def __init__(self, instance):
        from taichi_forge._lib import core
        from taichi_forge.graph._graph import _GraphRunContext
        from taichi_forge.lang import impl

        spec = instance.spec
        if not _eligible(spec, "cuda"):
            raise ValueError(
                "immutable argument frames require one CUDA buffer/sampled-Texture Graph "
                "with qualified fixed-plan commands"
            )
        node = spec.nodes[0]
        recordings = tuple(
            lease for lease in node.lifetime_leases if getattr(lease, "_graph_binding_frame_capture_safe", False)
        )
        retained_owners = {
            id(owner) for recording in recordings for owner in (recording, getattr(recording, "plan", None))
        }
        if any(id(lease) not in retained_owners for lease in spec.runtime_lifetime_leases):
            raise ValueError("immutable frames cannot discharge an unrelated provider lifetime")
        self._provider_validators = tuple(lease.validate_graph_lifetime for lease in spec.runtime_lifetime_leases)
        for validate in self._provider_validators:
            validate()
        self._native = core._CudaGraphBindingExecutor(node.compiled_graph, impl.current_cfg(), impl.get_runtime().prog)
        self._dispatch_count = node.physical_dispatch_count
        self._task_count = sum(len(stage["tasks"]) for stage in spec.pipeline_definition)
        self._raw_context = _GraphRunContext()
        self._spec = spec
        # Pin the concrete matrix, not merely its replaceable Python wrapper.
        # FFT execution resources are pinned by the native frame executor.
        self._provider_owners = tuple(
            lease.matrix.matrix
            for lease in node.lifetime_leases
            if getattr(lease, "_graph_binding_frame_capture_safe", False) and hasattr(lease, "matrix")
        )
        # Only this materialized, single-lane spec uses the retained native
        # executor. Its lifetime pins retire with that executor on close/reset;
        # ordinary Graph specs and caller-owned plans keep their old policy.
        spec.runtime_lifetime_leases = ()

    def prewarm(self):
        return self

    def prepare_binding_version(self, version):
        if not version.fast_path_qualified:
            if not self._spec.native_count or set(version.volatile_reasons).difference(
                ("volatile_lifetime_provider", "volatile_runtime_provider")
            ):
                raise ValueError("immutable argument frames require a publication-qualified Graph binding")
            # This executor only accepts certified fixed-plan commands. Validate
            # their current owners here; native preparation validates shapes,
            # aliasing and allocation lifetime before capturing immutable args.
            for validate in self._provider_validators:
                validate()
            context = self._raw_context
            context.begin(version.execution_arguments)
            try:
                flattened = dict(context.flattened_args())
            finally:
                context.end()
            version = replace(version, flattened_args=flattened)
        frame = self._native.prepare(version.flattened_args)
        return replace(version, execution_frame=frame, fast_path_qualified=True, volatile_reasons=())

    def run_prepared(self, invocation):
        version = invocation.binding_version
        if version is None:
            # A mapping has no immutable publication identity: this is an
            # explicit prepare+run call, not an allegedly upload-free replay.
            context = self._raw_context
            context.begin(invocation.arguments, flattened_args=invocation.flattened_args)
            try:
                frame = self._native.prepare(context.flattened_args())
            finally:
                context.end()
        else:
            frame = version.execution_frame
        self._native.run(frame)

    def invalidate_runtime(self, preserve_executables=False):
        self._native.close()
        self._provider_owners = ()
        self._raw_context = None

    @property
    def snapshot_graph_stats(self):
        from taichi_forge.graph._graph import _empty_backend_stats

        native = self._native.snapshot()
        result = _empty_backend_stats()
        result.update(
            backend="cuda",
            # This names the installed plan, not a counted last launch. No
            # replay counters or timing instrumentation are enabled by reading.
            last_path="cuda_prepared_binding_plan",
            diagnostics_counters_complete=False,
            known_compiled_dispatches=self._dispatch_count,
            known_compiled_tasks=self._task_count,
            known_persistent_argument_bytes=native["argument_bytes"],
            binding_frame_state=native,
        )
        return result

    @property
    def debug_graph_stats(self):
        return self.snapshot_graph_stats


class GraphBindingFrameRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "binding_frames",
        capabilities=(
            "immutable-argument-images",
            "whole-graph-executable-reuse",
            "typed-runtime-fragment",
            "fixed-plan-provider-capture",
            "sampled-texture-resource-retention",
            "vulkan-secondary-image-recording",
        ),
        domain_version="immutable-binding-frame-domain-v6",
        semantic_fingerprint="cuda-vulkan-composed-binding-retained-images-v6",
    )

    def fragments(self, definition):
        spec = definition._runtime_spec
        vulkan = definition.backend == "vulkan"
        if vulkan:
            from taichi_forge.graph._recipes.vulkan_binding_frames import eligible

            # Mixed FFT/provider graphs retain their existing provider-owned
            # submission fragment; do not generate a duplicate physical plan.
            if spec.native_count or not eligible(spec, definition.backend):
                return ()
        elif not _eligible(spec, definition.backend):
            return ()
        native = bool(spec.native_count)
        return (
            _fragment(
                definition,
                family="binding_frames",
                source_key="whole-graph-bindings",
                choice_id="immutable-argument-images",
                coverage=tuple(source.region_id for source in definition.sources),
                tasks=(
                    GraphFragmentTask.create(
                        "whole-graph-bindings:execute",
                        "vulkan_complete_graph_binding_reuse" if vulkan else "cuda_complete_graph_binding_reuse",
                        effects=spec.pre_optimization_ir_root.effects,
                        bindings=spec.pre_optimization_ir_root.bindings,
                        physical={
                            "argument_images": "immutable_per_published_binding",
                            "argument_upload": "preparation_only",
                            "argument_lifetime": "published_binding_and_inflight_work",
                            "workspace_lanes": 1,
                            **(
                                {
                                    "submission": "embedded_secondary_commands",
                                    "image_layouts": "closed_cycle_with_entry_repair_after_layout_change",
                                    "binding_transition": "select_immutable_secondary",
                                }
                                if vulkan
                                else {
                                    "queue": "default",
                                    "executable_count": 1,
                                    "binding_transition": "whole_executable_update",
                                    "completion_events": "reuse_observed_peak_until_executor_close",
                                }
                            ),
                            **({"provider_parameters": "captured_per_binding_fixed_plan"} if native else {}),
                        },
                    ),
                ),
                provider_descriptor=self.descriptor,
                executor_kind="vulkan_immutable_argument_frames" if vulkan else "cuda_immutable_argument_frames",
            ),
        )

    def contribute_runtime(self, assembly, selection):
        if selection.source_key != "whole-graph-bindings" or selection.choice_id != "immutable-argument-images":
            raise ValueError("unknown whole-Graph immutable binding selection")
        if assembly.definition.backend == "vulkan":
            from taichi_forge.graph._recipes.vulkan_binding_frames import VulkanBindingFrameExecutor

            assembly.select_binding_executor(VulkanBindingFrameExecutor)
        else:
            assembly.select_binding_executor(_BindingFrameExecutor)

    def describe(self, definition, fragment_key):
        if definition.backend == "vulkan":
            return {
                **super().describe(definition, fragment_key),
                "display_name": "Whole-Graph immutable Vulkan binding frames",
                "changes": (
                    "prepare arguments, descriptors and secondary commands at binding publication",
                    "retain buffer/image resources until frame and parent command retirement",
                    "record a closed image-layout cycle; repair entry layouts only after layout changes",
                ),
                "limitations": (
                    "flat Vulkan kernel Graph, one workspace lane; no SNode, AS or external synchronization domains",
                    "one-mip sampled/storage images; simultaneous sampled/storage alias in one task is unavailable",
                    "raw mapping calls include argument preparation; use Graph.bind to amortize it",
                    "uploads and intervening graphics operations retain their existing explicit boundaries",
                    "driver-owned command/descriptor memory remains opaque; benefit requires workload measurements",
                ),
            }
        return {
            **super().describe(definition, fragment_key),
            "display_name": "Whole-Graph immutable argument frames",
            "changes": (
                "prepare argument images when bindings are published",
                "reuse one executable across prepared bindings without reuploading arguments",
                "retain argument images and allocation leases until last device use",
                "retain sampled Texture objects in each immutable binding frame",
                "reuse completed event handles up to the observed queue peak until executor close",
            ),
            "limitations": (
                "one CUDA Graph and one workspace lane; only certified fixed-plan FFT/SpMM/matmul commands may join JIT dispatches",
                "no SNode, external synchronization domain or device-controlled topology; capture must contain only kernel nodes",
                "raw mapping calls include argument preparation; use Graph.bind to amortize it",
                "Texture content uploads remain explicit; published frames own sampled resources until retirement",
                "prepared frames trade retained argument memory and setup for binding-switch cost",
                "cached completion handles retain opaque driver storage, not measured ndarray or peak VRAM bytes",
                "wraps baseline or explicitly compatible FFT/SpMM/matmul region strategies; unrelated replacements remain unavailable",
                "benefit and driver-owned memory require workload measurements",
            ),
        }


__all__ = ["GraphBindingFrameRecipeProvider"]
