"""Whole-transform and separable FFT plans in the existing Graph composer."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _recipe_operation_dispatch_count,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask
from taichi_forge.graph._recipes.providers import GraphRecipeProviderError


def _sources(definition):
    regions = {region.path: region.region_id for region in definition.regions}
    for node_index, node in enumerate(definition._runtime_spec.nodes):
        ordinal = 0
        for operation in getattr(node, "recipe_operations", ()):
            if operation[0] == "native":
                executable = operation[1]
                source = getattr(getattr(executable, "_recording", None), "_graph_fft_source", None)
                if source is not None:
                    path = f"graph/{node_index}:{node.ir_node.kind}/{ordinal}:native_call"
                    yield path, regions[path], source, executable
            ordinal += _recipe_operation_dispatch_count(operation)


class FftRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit physical provider for linalg.record_fft() mathematical regions.

    Add alongside graph.default_recipe_providers(). No vendor route or kernel
    launch parameter is exposed to CompileIQ; it schedules complete recipes.
    """

    descriptor = runtime_family_provider_descriptor(
        "fft",
        capabilities=(
            "c2c-fft-region",
            "retained-separable-plan",
            "cross-batch-column-plan",
            "output-scale-fusion",
        ),
        domain_version="fft-retained-plans-v5",
        semantic_fingerprint="compact-2d-c2c-finite-f32-v1",
    )

    def fragments(self, definition):
        fragments = []
        for path, region_id, source, _ in _sources(definition):
            if source.transform != "c2c":
                # Real regions already lower through the complete baseline and
                # enclosing executor. Do not invent another FFT algorithm axis.
                continue
            strategy = "row_batch_column_inplace"
            prepared = source.preparation_report()
            if strategy not in prepared:
                raise GraphRecipeProviderError(
                    "Prepare the FFT operation before complete recipe search",
                    error_key="fft_plan_preparation_required",
                    provider_namespace=self.descriptor.namespace,
                )
            for strategy in prepared:
                if strategy == "whole_transform":
                    continue
                task = GraphFragmentTask.create(
                    f"{path}:{strategy}",
                    "retained_fft_transform",
                    physical={
                        "semantic_contract": source.semantics,
                        "config": source.physical_config(strategy),
                        "component": source.component,
                        "workspace_bytes": prepared[strategy]["workspace_bytes"],
                        "vendor_internal_kernel_topology": "unknown",
                        **(
                            {"callback": prepared[strategy]["callback"]}
                            if "callback" in prepared[strategy]
                            else {}
                        ),
                    },
                )
                tasks = [task]
                if (
                    source.output_scale != 1.0
                    and strategy != "whole_transform_store_scale"
                ):
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:{strategy}:scale",
                            "fft_output_scale",
                            depends_on=(task.task_id,),
                            physical={
                                "scale": source.output_scale,
                                "lowering": "forge_f32_scale_kernel_v1",
                            },
                        )
                    )
                fragments.append(
                    _fragment(
                        definition,
                        family="fft",
                        source_key=path,
                        choice_id=strategy,
                        coverage=(region_id,),
                        tasks=tuple(tasks),
                        exclusive_submission=True,
                        provider_descriptor=self.descriptor,
                        compatible_executor_kinds=("cuda_immutable_argument_frames",),
                    )
                )
        return tuple(fragments)

    def contribute_runtime(self, assembly, selection):
        matches = tuple(row for row in _sources(assembly.definition) if row[0] == selection.source_key)
        if len(matches) != 1:
            raise ValueError("Frozen FFT semantic region is unavailable")
        _, _, source, executable = matches[0]
        strategy = selection.materialization_choice
        if strategy not in source.preparation_report():
            raise ValueError("FFT plan was not prepared; no physical substitution was performed")

        def rewrite(builder, operation):
            source.append(builder, strategy, operation[2])

        assembly.select_operation(executable, rewrite)

    def explain_discovery(self, definition):
        sources = tuple(_sources(definition))
        count = sum(source.transform == "c2c" for _, _, source, _ in sources)
        return {
            "source": "provider_declared_not_measured",
            "semantic_source_count": len(sources),
            "c2c_decomposition_source_count": count,
            "real_whole_plan_source_count": len(sources) - count,
            "real_plan_scope": "whole-transform baseline and enclosing Graph executor; no C2C decomposition/callback",
            "reason": (
                "prepared_batched_2d_sources" if count else
                ("real_whole_plan_with_enclosing_executor" if sources else "no_frozen_fft_semantic_source")
            ),
            "semantic_api": "ti.linalg.record_fft",
            "scope": "CUDA complex-f32, compact batched 2D, prepared or imported expected facts",
        }

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        selection = fragment.provider_metadata["family_selection"]
        source = next(row[2] for row in _sources(definition) if row[0] == selection["source_key"])
        strategy = selection["materialization_choice"]
        return {
            **fragment.provider_metadata,
            "semantic_contract": source.semantics,
            "frozen_config": source.physical_config(strategy),
            "component_applicability": source.component,
            "preparation_observation": {
                **source.preparation_report()[strategy],
                "measurement_scope": (
                    "host_elapsed_for_callback_compile_and_fft_plan_creation"
                    if strategy == "whole_transform_store_scale"
                    else "host_elapsed_for_fft_plan_creation"
                ),
                "shared_initialization": "not_separated",
                "preparation_origin": source._preparation_origin,
                "selected_only_restore": (
                    "observed" if strategy in source._restoration else "not_measured"
                ),
                "restoration_observation": source._restoration.get(strategy),
            },
            "limitations": (
                "CUDA f32 complex-to-complex only; compact two-dimensional transforms and explicit batch count",
                "both transforms are unnormalized, followed by explicit output_scale when declared; finite inputs and caller tolerances",
                "store-scale LTO is opt-in and requires compatible external NVRTC/nvJitLink; no replay compiler or callerInfo allocation",
                "operation.close releases search-owned plans; live Graphs retain their own plans, frozen definitions retain descriptions",
                "prepared metadata does not retain unused plans; per-plan workspace is not total process VRAM",
                "in-place columns use the output allocation; public input and output remain distinct",
                "cross-batch columns execute once per column; per-image columns execute once per batch, neither is universally faster",
                "vendor-internal kernel count is unobserved by the manifest; use profiler evidence when needed",
            ),
        }


__all__ = ["FftRecipeProvider"]
