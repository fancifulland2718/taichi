"""Complete matmul regions in the existing opaque Graph recipe composer."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _recipe_operation_dispatch_count,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import (
    GraphFragmentResourceRequirement,
    GraphFragmentTask,
)


def _sources(definition):
    regions = {region.path: region.region_id for region in definition.regions}
    for index, node in enumerate(definition._runtime_spec.nodes):
        ordinal = 0
        for operation in getattr(node, "recipe_operations", ()):
            if operation[0] == "native":
                source = getattr(
                    getattr(operation[1], "_recording", None),
                    "_graph_matmul_source",
                    None,
                )
                if source is not None:
                    path = f"graph/{index}:{node.ir_node.kind}/{ordinal}:native_call"
                    yield path, regions[path], source, operation[1]
            ordinal += _recipe_operation_dispatch_count(operation)


class MatmulRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit frozen-algorithm, operand-layout and epilogue region provider.

    Include alongside graph.default_recipe_providers(). Only complete Graph
    recipe IDs reach CompileIQ; library and kernel configuration stay here.
    """

    descriptor = runtime_family_provider_descriptor(
        "matmul",
        capabilities=(
            "semantic-f32-matmul",
            "frozen-algorithm-config",
            "operand-packing",
            "bounded-relu-epilogue",
        ),
        domain_version="matmul-complete-region-v1",
        semantic_fingerprint="compact-f32-fixed-shape-matmul-v1",
    )

    def fragments(self, definition):
        from taichi_forge.linalg._matmul import _storage_bytes

        result = []
        for path, region, source, _ in _sources(definition):
            for key, config in source.facts["choices"].items():
                if key == source.baseline:
                    continue
                tasks = []
                for name in config["packed_inputs"]:
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:pack:{name}",
                            "matmul_operand_pack",
                            depends_on=(tasks[-1].task_id,) if tasks else (),
                            physical={"operand": name, "transpose": True},
                        )
                    )
                tasks.append(
                    GraphFragmentTask.create(
                        f"{path}:matmul",
                        "retained_matmul_region",
                        depends_on=(tasks[-1].task_id,) if tasks else (),
                        physical={
                            "semantics": source.semantics,
                            "configuration": config,
                            "component": source.facts["component"],
                            "vendor_internal_kernel_topology": "unobserved",
                        },
                    )
                )
                if config["epilogue"] == "separate":
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:relu",
                            "matmul_relu",
                            depends_on=(tasks[-1].task_id,),
                            physical={"activation": "relu"},
                        )
                    )
                requested = (
                    _storage_bytes(source.semantics, config)
                    + config["algorithm"]["workspace_bytes"]
                )
                resources = (
                    (
                        GraphFragmentResourceRequirement(
                            f"{path}:workspace",
                            "matmul_region_storage",
                            requested,
                            ownership="graph_instance",
                            lifetime="graph",
                            exclusive_submission=True,
                        ),
                    )
                    if requested
                    else ()
                )
                result.append(
                    _fragment(
                        definition,
                        family="matmul",
                        source_key=path,
                        choice_id=key,
                        coverage=(region,),
                        tasks=tuple(tasks),
                        resources=resources,
                        exclusive_submission=True,
                        provider_descriptor=self.descriptor,
                        compatible_executor_kinds=("cuda_immutable_argument_frames",),
                    )
                )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        matches = tuple(
            row
            for row in _sources(assembly.definition)
            if row[0] == selection.source_key
        )
        if len(matches) != 1:
            raise ValueError("Frozen matmul semantic source is unavailable")
        _, _, source, executable = matches[0]
        source.physical_config(selection.materialization_choice)
        assembly.select_operation(
            executable,
            lambda builder, operation: source.append(
                builder, selection.materialization_choice, operation[2]
            ),
        )

    def explain_discovery(self, definition):
        sources = tuple(_sources(definition))
        return {
            "source": "provider_declared_not_measured",
            "semantic_source_count": len(sources),
            "reason": (
                "prepared_matmul_regions" if sources else "no_frozen_matmul_source"
            ),
            "semantic_api": "ti.linalg.record_matmul",
            "unavailable": tuple(row[2].facts["unavailable"] for row in sources),
        }

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        selection = fragment.provider_metadata["family_selection"]
        source = next(
            row[2] for row in _sources(definition) if row[0] == selection["source_key"]
        )
        return {
            **fragment.provider_metadata,
            "semantic_contract": source.semantics,
            "frozen_config": source.physical_config(
                selection["materialization_choice"]
            ),
            "component_applicability": source.facts["component"],
            "preparation_observation": source.facts["preparation"],
            "limitations": (
                "CUDA compact scalar f32 matrices; fixed dimensions and optional strided batch",
                "finite inputs and application-qualified tolerance; no per-replay value scan",
                "packing refreshes mutable inputs on every replay; it is not a retained-value cache",
                "vendor workspace is known; driver state and process peak VRAM are unknown",
                "no library routing or algorithm number is a public CompileIQ axis",
            ),
        }


__all__ = ["MatmulRecipeProvider"]
