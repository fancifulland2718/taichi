"""External whole-Graph provider, measured search and portable selection.

Run with a compatible Forge runtime and the maintained CompileIQ fork installed.
This integer example demonstrates the integration contract, not application
acceleration. The measured objective is synchronized wall time, not GPU time.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import taichi_forge as ti


@ti.kernel
def twice(source: ti.types.ndarray(dtype=ti.i32, ndim=1), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in source:
        output[i] = source[i] * 2


@ti.kernel
def increment(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in output:
        output[i] += 1


@ti.kernel
def fused(source: ti.types.ndarray(dtype=ti.i32, ndim=1), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in source:
        output[i] = source[i] * 2 + 1


def make_builder(single_pass=False):
    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    if single_pass:
        builder.dispatch(fused, source, output)
    else:
        builder.dispatch(twice, source, output)
        builder.dispatch(increment, output)
    return builder


class AffineProvider:
    """Only replaces the explicitly supplied example definition, never guesses."""

    descriptor = ti.graph.GraphRecipeProviderDescriptor(
        namespace="example.affine",
        provider_version="1",
        domain_version="1",
        semantic_fingerprint="i32-modular-twice-plus-one-v1",
        assembly_protocols=(ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,),
    )

    def __init__(self, definition):
        self.semantic_graph_id = definition.semantic_graph_id

    def discover(self, definition):
        if definition.semantic_graph_id != self.semantic_graph_id:
            return ()
        return (
            ti.graph.GraphRecipeFragment.create(
                definition,
                provider_namespace=self.descriptor.namespace,
                provider_version=self.descriptor.provider_version,
                provider_domain_version=self.descriptor.domain_version,
                fragment_key="single-pass",
                coverage_region_ids=tuple(r.region_id for r in definition.regions),
                tasks=(
                    ti.graph.GraphFragmentTask.create(
                        "affine",
                        "kernel",
                        physical={"implementation": "twice-plus-one-v1"},
                    ),
                ),
                binding_requirements=tuple(
                    ti.graph.GraphFragmentBindingRequirement(name) for name in ("source", "output")
                ),
                assembly_protocol=ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,
            ),
        )

    def resolve(self, definition, fragment_key):
        fragments = self.discover(definition)
        if fragment_key != "single-pass" or not fragments:
            raise ti.graph.GraphRecipeProviderError(
                "Example recipe is not available for this definition",
                error_key="recipe_fragment_unavailable",
                provider_namespace=self.descriptor.namespace,
                fragment_key=fragment_key,
            )
        return fragments[0]

    def expand(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return ()

    def materialize(self, scope, fragment):
        graph = make_builder(single_pass=True).freeze().compile()
        scope.own_executor(graph)  # Enroll explicit retirement before observation.
        return ti.graph.GraphMaterializedFragment.create(fragment, graph)

    def assemble(self, scope, definition, recipe, fragments):
        graph = fragments[0].payload
        return ti.graph.GraphMaterializationProduct(
            graph,
            ti.graph.CompiledGraphPhysicalManifest.from_graph(definition, recipe, graph),
        )

    def describe(self, definition, fragment_key):
        self.resolve(definition, fragment_key)
        return {
            "change": "two integer passes become one; no intermediate allocation",
            "limitations": "example-owned i32 affine operation only",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--evaluation-limit", type=int, default=16)
    parser.add_argument(
        "--environment-id", required=True, help="Caller-owned stable device/driver/runtime environment description"
    )
    args = parser.parse_args()
    ti.init(arch=ti.cuda, offline_cache=False)
    source = ti.ndarray(ti.i32, args.size)
    output = ti.ndarray(ti.i32, args.size)
    values = np.arange(args.size, dtype=np.int32) % 97
    expected = values * 2 + 1
    definition = make_builder().freeze()
    providers = (AffineProvider(definition),)  # Baseline always remains available.
    target = ti.graph.GraphOptimizationTarget(objectives=(("wall_ns", "min"),))
    workload = ti.graph.GraphWorkloadContext({"operation": "affine", "size": args.size})
    evaluation = ti.graph.GraphEvaluationContract(
        {
            "correctness": "exact i32, refreshed input/output for each evaluation",
            "warmup": 4,
            "replays": 32,
            "metric_definitions": {
                "wall_ns": {
                    "unit": "ns",
                    "source": "perf_counter_ns",
                    "scope": "host submission plus completion wait, per replay; not device time",
                    "interval": "32 bound replays after warmup; ti.sync at both ends",
                }
            },
        }
    )
    environment = ti.graph.GraphBackendEnvironment({"identity": args.environment_id})
    contracts = dict(workload_context=workload, evaluation_contract=evaluation, backend_environment=environment)

    def evaluator(graph, recipe):
        source.from_numpy(values)
        output.fill(0)
        bindings = graph.bind({"source": source, "output": output})
        for _ in range(4):
            graph.run(bindings)
        ti.sync()
        begin = time.perf_counter_ns()
        for _ in range(32):
            graph.run(bindings)
        ti.sync()
        elapsed = (time.perf_counter_ns() - begin) / 32
        np.testing.assert_array_equal(output.to_numpy(), expected)
        return {"wall_ns": elapsed}

    if args.restore:
        selection = json.loads(args.restore.read_text(encoding="utf8"))
        applicability = definition.check_recipe_applicability(
            selection,
            providers=providers,
            target=target,
            **contracts,
        )
        print(applicability.to_dict())
        # Structural reuse can remain valid even when measurements need renewal.
        handle = definition.resolve_recipe(selection, providers=providers)
        with definition.materialize(handle) as materialized:
            print(evaluator(materialized.executor, handle))
        return

    checkpoint = None if not args.resume else json.loads(args.resume.read_text(encoding="utf8"))
    decision = definition.search_recipes(
        providers=providers,
        target=target,
        budget=ti.graph.GraphSearchBudget(evaluation_limit=args.evaluation_limit, repeat_count=2),
        checkpoint=checkpoint,
        **contracts,
    ).run(evaluator)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(decision.report.to_json(), encoding="utf8")
    (args.output / "report.md").write_text(decision.report.to_markdown(), encoding="utf8")
    (args.output / "checkpoint.json").write_text(
        json.dumps(decision.checkpoint.to_dict(), indent=2),
        encoding="utf8",
    )
    print(decision.status, decision.next_action)
    if decision.status == "selected":
        (args.output / "selection.json").write_text(
            json.dumps(decision.selection_artifact.to_dict(), indent=2),
            encoding="utf8",
        )
        with definition.materialize(decision.selection) as materialized:
            print(evaluator(materialized.executor, decision.selection))
    # Partial/failed outcomes keep reports and checkpoint. Do not accidentally
    # call materialize(None): that explicitly requests the baseline.


if __name__ == "__main__":
    main()
