"""Complete contraction dataflows; CompileIQ never receives a vendor plan axis."""

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
                    "_graph_contraction_source",
                    None,
                )
                if source is not None:
                    path = f"graph/{index}:{node.ir_node.kind}/{ordinal}:native_call"
                    yield path, regions[path], source, operation[1]
            ordinal += _recipe_operation_dispatch_count(operation)


class ContractionRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit contraction/packing/epilogue regions using fixed input contracts.

    Include alongside graph.default_recipe_providers(). Restoration recreates
    the selected vendor plan request, not a serialized vendor kernel binary.
    """

    descriptor = runtime_family_provider_descriptor(
        "contraction",
        capabilities=(
            "semantic-f32-contraction",
            "operand-mode-packing",
            "contraction-epilogue-dataflow",
            "retained-contraction-capture",
        ),
        domain_version="contraction-complete-region-v1",
        semantic_fingerprint="compact-f32-explicit-modes-contraction-v1",
    )

    def fragments(self, definition):
        from taichi_forge.linalg._contraction import _packing, _storage_bytes

        result = []
        for path, region, source, _ in _sources(definition):
            for key, config in source.facts["choices"].items():
                if key == source.baseline:
                    continue
                tasks = []

                def append(suffix, kind, facts):
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:{suffix}",
                            kind,
                            depends_on=(tasks[-1].task_id,) if tasks else (),
                            physical=facts,
                        )
                    )

                for name in _packing(source.semantics, config):
                    append(
                        f"pack:{name}",
                        "contraction_operand_permutation",
                        {
                            "operand": name,
                            "permutation": config["permutations"][name],
                            "refresh": "every_replay",
                        },
                    )
                append(
                    "contract",
                    "retained_contraction_region",
                    {
                        "semantics": source.semantics,
                        "configuration": config,
                        "component": source.facts["component"],
                        "vendor_internal_kernel_topology": "unobserved",
                    },
                )
                if (
                    config["epilogue"] == "separate"
                    or source.semantics["activation"] == "relu"
                ):
                    append(
                        "epilogue",
                        "contraction_pointwise_epilogue",
                        {
                            "alpha_beta": (
                                "separate"
                                if config["epilogue"] == "separate"
                                else "vendor"
                            ),
                            "activation": source.semantics["activation"],
                        },
                    )
                requested = (
                    _storage_bytes(source.semantics, config) + config["workspace_bytes"]
                )
                resources = (
                    (
                        GraphFragmentResourceRequirement(
                            f"{path}:storage",
                            "contraction_region_storage",
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
                        family="contraction",
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
            raise ValueError("Frozen contraction semantic source is unavailable")
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
        return dict(
            source="provider_declared_not_measured",
            semantic_source_count=len(sources),
            reason=(
                "prepared_contraction_regions"
                if sources
                else "no_frozen_contraction_source"
            ),
            semantic_api="ti.linalg.record_contraction",
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
                "CUDA compact f32 tensors with explicit modes, matched extents and caller precision/tolerance",
                "no broadcasting or repeated input modes; Forge helper rank and element limits apply",
                "packing and epilogue read current inputs each replay; no input-value cache",
                "vendor kernel identities are unobserved; restoration reconstructs the selected plan request",
                "known workspace and private buffers do not include opaque vendor state or driver peak",
                "only complete Graph identities reach CompileIQ; no raw library/kernel routing",
            ),
        }


__all__ = ["ContractionRecipeProvider"]
