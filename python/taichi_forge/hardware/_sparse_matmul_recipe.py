"""Complete shared-A sparse matmul recipes, with provider-owned algorithms."""

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
    regions = {r.path: r.region_id for r in definition.regions}
    for index, node in enumerate(definition._runtime_spec.nodes):
        ordinal = 0
        for operation in getattr(node, "recipe_operations", ()):
            if operation[0] == "native":
                source = getattr(
                    getattr(operation[1], "_recording", None),
                    "_graph_sparse_matmul_source",
                    None,
                )
                if source is not None:
                    path = f"graph/{index}:{node.ir_node.kind}/{ordinal}:native_call"
                    yield path, regions[path], source, operation[1]
            ordinal += _recipe_operation_dispatch_count(operation)


class SparseMatmulRecipeProvider(GraphRuntimeFragmentProvider):
    """Search prepared shared-A FP16 2:4 regions, not bare algorithm IDs.

    Include alongside graph.default_recipe_providers(). The baseline already
    reuses one compression per invocation with the vendor's default algorithm.
    """

    descriptor = runtime_family_provider_descriptor(
        "sparse_matmul",
        domain_version="shared-a-sparse-matmul-region-v1",
        semantic_fingerprint="current-fp16-row-2of4-shared-a-products-v1",
        capabilities=(
            "semantic-shared-a-sparse-matmul",
            "frozen-algorithm-resources",
            "compression-reuse-within-invocation",
            "matmul-epilogue-dataflow",
        ),
    )

    def fragments(self, definition):
        fragments = []
        for path, region, source, _ in _sources(definition):
            semantics = source.semantics
            for key, config in source.facts["choices"].items():
                if key == source.baseline:
                    continue
                tasks = []

                def append(suffix, kind, physical):
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:{suffix}",
                            kind,
                            depends_on=(tasks[-1].task_id,) if tasks else (),
                            physical=physical,
                        )
                    )

                append(
                    "compress",
                    "structured_sparse_compression",
                    dict(
                        algorithm=config["algorithm"],
                        input=semantics["a"],
                        refresh="once_per_graph_invocation",
                        component=source.facts["component"],
                    ),
                )
                for index, names in enumerate(semantics["products"]):
                    append(
                        f"product:{index}",
                        "retained_sparse_matmul",
                        dict(
                            algorithm=config["algorithm"],
                            bindings=names,
                            shape=(semantics["m"], semantics["n"], semantics["k"]),
                            alpha=semantics["alpha"],
                            beta=semantics["beta"],
                            compressed_storage="region_shared",
                            workspace="region_shared",
                            vendor_kernel_topology="unobserved",
                        ),
                    )
                if config["epilogue"] == "separate_relu":
                    for index, names in enumerate(semantics["products"]):
                        append(
                            f"relu:{index}",
                            "sparse_matmul_pointwise_epilogue",
                            dict(output=names[2], activation="relu"),
                        )
                resources = (
                    GraphFragmentResourceRequirement(
                        f"{path}:plan_storage",
                        "shared_sparse_matmul_storage",
                        sum(config["resources"]),
                        ownership="graph_instance",
                        lifetime="graph",
                        exclusive_submission=True,
                    ),
                )
                fragments.append(
                    _fragment(
                        definition,
                        family="sparse_matmul",
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
        return tuple(fragments)

    def contribute_runtime(self, assembly, selection):
        matches = tuple(
            row
            for row in _sources(assembly.definition)
            if row[0] == selection.source_key
        )
        if len(matches) != 1:
            raise ValueError("Frozen sparse matmul source is unavailable")
        _, _, source, executable = matches[0]
        source.physical_config(selection.materialization_choice)
        assembly.select_operation(
            executable,
            lambda builder, operation: source.append(
                builder,
                selection.materialization_choice,
                operation[2],
            ),
        )

    def explain_discovery(self, definition):
        sources = tuple(_sources(definition))
        return dict(
            source="provider_declared_not_measured",
            semantic_source_count=len(sources),
            reason=(
                "prepared_sparse_matmul_regions"
                if sources
                else "no_frozen_sparse_matmul_source"
            ),
            semantic_api="ti.linalg.record_sparse_matmul",
            baselines=tuple(
                dict(
                    source_key=path,
                    semantic_contract=source.semantics,
                    frozen_dataflow=source.physical_config(source.baseline),
                    physical_id=source.physical_id(source.baseline),
                    component_applicability=source.facts["component"],
                    preparation_observation=source.facts["preparation"],
                )
                for path, _, source, _ in sources
            ),
            unavailable=tuple(row[2].facts["unavailable"] for row in sources),
        )

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        selection = fragment.provider_metadata["family_selection"]
        source = next(
            row[2] for row in _sources(definition) if row[0] == selection["source_key"]
        )
        return {
            **fragment.provider_metadata,
            "semantic_contract": source.semantics,
            "frozen_dataflow": source.physical_config(
                selection["materialization_choice"]
            ),
            "component_applicability": source.facts["component"],
            "preparation_observation": source.facts["preparation"],
            "limitations": (
                "CUDA compact FP16 row 2:4 A, FP32 accumulation, equal M/N/K products and caller tolerance",
                "current A is compressed once each invocation; no pruning, content readback or cross-replay value cache",
                "only each product's own C/D may alias; outputs may not overwrite other inputs or outputs",
                "algorithm and compressed storage belong to one retained plan; no cross-algorithm sharing",
                "known compressed/scratch/workspace bytes exclude inputs, opaque vendor state and driver peak",
                "restoration rebuilds frozen algorithm attributes, not serialized vendor kernel binaries",
                "descriptor enumeration and all performance measurements remain separate",
            ),
        }


__all__ = ["SparseMatmulRecipeProvider"]
