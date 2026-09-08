"""Whole sparse-solve lifecycle recipes; CompileIQ sees opaque identities."""

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
                    "_graph_sparse_solve_source",
                    None,
                )
                if source is not None:
                    path = f"graph/{index}:{node.ir_node.kind}/{ordinal}:native_call"
                    yield path, regions[path], source, operation[1]
            ordinal += _recipe_operation_dispatch_count(operation)


class SparseSolveRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit provider for prepared sparse-solve regions, alongside defaults."""

    descriptor = runtime_family_provider_descriptor(
        "sparse_solve",
        domain_version="shared-pattern-sparse-solve-region-v2",
        semantic_fingerprint="csr-f32-shared-rhs-lifecycle-capture-storage-v2",
        capabilities=(
            "semantic-sparse-solve",
            "frozen-ordering-lifecycles",
            "shared-factor-rhs-group",
            "retained-private-analysis",
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

                if config["numeric_phase"] != "solve":
                    append(
                        "factor",
                        "sparse_numerical_factorization",
                        dict(
                            phase=config["numeric_phase"],
                            values=semantics["values"],
                            configuration=config["configuration"],
                            pattern=semantics["topology_fingerprint"],
                            analysis="retained_from_materialization",
                            reuse="all_rhs_in_region",
                            vendor_kernel_topology="unobserved",
                        ),
                    )
                for index, pair in enumerate(semantics["rhs_pairs"]):
                    append(
                        f"rhs:{index}",
                        "sparse_triangular_solve",
                        dict(
                            bindings=pair,
                            configuration=config["configuration"],
                            factor_lifetime=semantics["matrix_lifetime"],
                            shared_factor_owner=path,
                            submission="retained_stream_capture",
                            capture_parameter_storage=config[
                                "capture_parameter_storage"
                            ],
                            component=source.facts["component"],
                            vendor_kernel_topology="unobserved",
                        ),
                    )
                resources = (
                    sum(config["resources"].values()) + 4 * semantics["nonzeros"]
                )
                fragments.append(
                    _fragment(
                        definition,
                        family="sparse_solve",
                        source_key=path,
                        choice_id=key,
                        coverage=(region,),
                        tasks=tuple(tasks),
                        resources=(
                            GraphFragmentResourceRequirement(
                                f"{path}:solver_storage",
                                "sparse_solver_requested_storage",
                                resources,
                                ownership="graph_instance",
                                lifetime="graph",
                                exclusive_submission=True,
                            ),
                        ),
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
            raise ValueError("Frozen sparse solve source is unavailable")
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
                "prepared_sparse_solve_regions"
                if sources
                else "no_frozen_sparse_solve_source"
            ),
            semantic_api="ti.linalg.record_sparse_solve",
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
                "CUDA f32 scalar CSR; immutable pattern and caller-declared nonsingular matrix class",
                "one shared analysis/factor owner, sequential compact vector RHS; no hidden batching or format conversion",
                "fixed-matrix and current-values regions have distinct semantics, never interchangeable choices",
                "caller evaluates residuals; no per-replay input scan, error readback or convergence test",
                "capture parameter storage follows the reported native capability; device copies and workspace clears remain",
                "per-binding resident parameter bytes are reported by Graph execution memory, not counted as shared plan workspace",
                "known requested payload includes private and catalog numeric snapshots, excludes caller inputs and opaque driver pool residency",
                "ordering defaults remain vendor policy; kernel topology and resolved default algorithm are unobserved",
                "restoration rebuilds selected configuration and factors, never serializes an executable",
            ),
        }


__all__ = ["SparseSolveRecipeProvider"]
