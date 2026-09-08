"""Batch workspace and complete submission strategies for Vulkan FFT regions."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


def _sources(definition):
    from taichi_forge.graph._graph import _CompiledNativeGraphNode
    from taichi_forge.hardware._vulkan_fft import _Recording

    if definition.backend != "vulkan":
        return
    regions = {region.path: region.region_id for region in definition.regions}
    for index, node in enumerate(definition._runtime_spec.nodes):
        if isinstance(node, _CompiledNativeGraphNode):
            recording = getattr(node.executable, "_recording", None)
            if isinstance(recording, _Recording):
                path = f"graph/{index}:{node.ir_node.kind}"
                yield index, path, regions[path], recording


def _strategies(plan):
    if plan._statistics.get("recipe_extension_abi") != 1:
        return ()
    current = int(plan._statistics["batch_tile"])
    return tuple(
        tile
        for tile in dict.fromkeys((max(1, plan.batch_count // 2), 1))
        if tile != current
    )


class VulkanFftRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit complete-recipe provider for VulkanFftPlan.record() regions.

    Batch partitions trade repeated dispatches for reusable scratch; they do
    not choose library routes. A separate composable submission fragment embeds
    the complete mixed Graph in an immutable secondary command sequence.
    Caller-owned baseline plans/storage remain live while constructing recipes;
    each selected replacement owns only its requested plan. No global plan
    cache, pre-generation of vendor candidates, or replay discovery is added.
    """

    descriptor = runtime_family_provider_descriptor(
        "vulkan_fft",
        capabilities=(
            "compact-inplace-c2c-region",
            "batch-workspace-reuse",
            "whole-graph-secondary-recording",
        ),
        domain_version="vulkan-fft-batch-and-submission-v1",
        semantic_fingerprint="compact-f32-c2c-finite-rank1-3-v1",
    )

    def fragments(self, definition):
        from taichi_forge.graph._recipes.vulkan_binding_frames import eligible

        result = []
        sources = tuple(_sources(definition))
        for _, path, region, recording in sources:
            plan = recording.plan
            plan.validate_graph_lifetime()
            for tile in _strategies(plan):
                result.append(
                    _fragment(
                        definition,
                        family="vulkan_fft",
                        source_key=path,
                        choice_id=f"batch-tile-{tile}",
                        coverage=(region,),
                        tasks=(
                            GraphFragmentTask.create(
                                f"{path}:partition-{tile}",
                                "retained_batched_fft",
                                effects=recording.resource_effects,
                                physical={
                                    "semantic_fft": plan._semantic_id,
                                    "dimensions": plan.dimensions,
                                    "batch_count": plan.batch_count,
                                    "direction": plan.direction,
                                    "normalization": plan.normalization,
                                    "batch_tile": tile,
                                    "batch_groups": (plan.batch_count + tile - 1)
                                    // tile,
                                    "tail_batch_count": plan.batch_count % tile,
                                    "component_binary": plan._adapter_sha256,
                                    "workspace": "one_full_tile_application_and_optional_tail",
                                    "actual_shader_and_workspace_facts": "observed_at_materialization",
                                },
                            ),
                        ),
                        exclusive_submission=True,
                        provider_descriptor=self.descriptor,
                        compatible_executor_kinds=("vulkan_immutable_argument_frames",),
                    )
                )
        if sources and eligible(definition._runtime_spec, definition.backend):
            result.append(
                _fragment(
                    definition,
                    family="vulkan_fft",
                    source_key="whole-graph-bindings",
                    choice_id="secondary-frames",
                    coverage=tuple(region.region_id for region in definition.regions),
                    tasks=(
                        GraphFragmentTask.create(
                            "vulkan-graph:prepared",
                            "vulkan_complete_graph_binding_reuse",
                            effects=definition._runtime_spec.pre_optimization_ir_root.effects,
                            bindings=definition._runtime_spec.pre_optimization_ir_root.bindings,
                            physical={
                                "submission": "embedded_secondary_commands",
                                "workspace_lanes": 1,
                                "argument_images": "immutable_per_published_binding",
                                "argument_upload": "preparation_only",
                                "lifetime": "binding_version_and_parent_command_retirement",
                                "provider_calls": "cold_inline_recording_only",
                            },
                        ),
                    ),
                    provider_descriptor=self.descriptor,
                    executor_kind="vulkan_immutable_argument_frames",
                )
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        from taichi_forge.graph._graph import _CompiledNativeGraphNode
        from taichi_forge.graph._recipes.vulkan_binding_frames import (
            VulkanBindingFrameExecutor,
        )
        from taichi_forge.hardware._vulkan_fft import VulkanFftPlan

        if (
            selection.source_key == "whole-graph-bindings"
            and selection.choice_id == "secondary-frames"
        ):
            assembly.select_binding_executor(VulkanBindingFrameExecutor)
            return
        index, _, _, recording = next(
            row
            for row in _sources(assembly.definition)
            if row[1] == selection.source_key
        )
        plan = recording.plan
        choices = {f"batch-tile-{tile}": tile for tile in _strategies(plan)}
        tile = choices[selection.materialization_choice]

        def rewrite(node):
            plan.validate_graph_lifetime()
            replacement = VulkanFftPlan(
                plan._data,
                plan.dimensions,
                batch_count=plan.batch_count,
                direction=plan.direction,
                normalization=plan.normalization,
                adapter_path=plan._adapter_path,
                _batch_tile=tile,
            )
            try:
                if replacement._adapter_sha256 != plan._adapter_sha256:
                    raise ValueError(
                        "Vulkan FFT adapter changed since the Graph was frozen"
                    )
                return _CompiledNativeGraphNode(
                    replacement.record(data=recording.data).compile()
                )
            except BaseException:
                replacement.close()
                raise

        assembly.rewrite_node(index, rewrite)

    def explain_discovery(self, definition):
        sources = tuple(_sources(definition))
        return {
            "source": "provider_declared_not_measured",
            "semantic_source_count": len(sources),
            "reason": (
                "fixed_vulkan_fft_sources"
                if sources
                else "no_vulkan_fft_semantic_source"
            ),
            "semantic_api": "ti.hardware.fft.VulkanFftPlan.record",
            "scope": "compact in-place complex-f32, rank 1--3, finite caller inputs",
        }

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        choice = fragment.provider_metadata["family_selection"]
        embedded = choice["source_key"] == "whole-graph-bindings"
        return {
            **fragment.provider_metadata,
            "changes": (
                ("record kernels and FFT regions as one immutable secondary Graph",)
                if embedded
                else (
                    "partition independent batches to reuse tile scratch without intermediate host transfers",
                )
            ),
            "selected_physical_plan": tuple(task.physical for task in fragment.tasks),
            "limitations": (
                "caller keeps the original compact ndarray and baseline plan open for recipe construction",
                "new process rebuilds equivalent baseline plans/storage before resolving the selected recipe",
                "selected partitions alone are created at materialization; baseline allocations are caller-owned",
                "finite complex-f32 inputs; no bitwise reproducibility or production tolerance qualification",
                "small transforms can be slower with no scratch benefit; measured trade-offs decide selection",
                "use Graph.bind for upload-free replay; raw mappings include argument preparation",
                "one ordered workspace lane; no SNode, textures, host-return kernels or controlled Graph topology",
                "provider requested bytes exclude opaque driver command/pipeline memory; no total-VRAM claim",
            ),
        }


__all__ = ["VulkanFftRecipeProvider"]
