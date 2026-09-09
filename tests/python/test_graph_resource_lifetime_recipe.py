"""Resource grouping, physical identity and public complete-recipe ownership."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import gen_cpp_kernel
from taichi_forge.graph._ir import (
    DispatchNode,
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    RuntimeBinding,
    SequentialRegion,
    TemporaryRequirement,
)
from taichi_forge.graph._native import DispatchGraphAction, NativeGraphExecutable, NativeGraphNode
from taichi_forge.graph._recipes import resource_lifetime
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.physical import observe_graph_physical_manifest
from taichi_forge.graph._recipes.resource_lifetime import GraphResourceLifetimeRecipeProvider
from taichi_forge.graph._recipes.semantic_families import GraphReductionRecipeProvider
from taichi_forge.graph._recipes.submission_families import GraphRecordingPartitionRecipeProvider
from tests import test_utils


def _definition(*, groups=2, size=131, reduction=False):
    @ti.kernel
    def stage(value: ti.i32, scratch: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in scratch:
            scratch[i] = value + i * 3

    @ti.kernel
    def finish(scratch: ti.types.ndarray(dtype=ti.i32, ndim=1), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = scratch[i] * 2 + 7

    builder = ti.graph.GraphBuilder()
    for index in range(groups):
        scratch = builder.private_ndarray(f"scratch_{index}", ti.i32, size)
        value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"value_{index}", ti.i32)
        output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, f"output_{index}", ti.i32, ndim=1)
        builder.dispatch(stage, value, scratch)
        builder.dispatch(finish, scratch, output)
    if reduction:
        values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
        total = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "total", ti.i32, ndim=1)
        builder.reduce(values, total, count=size)
    return builder.freeze()


def _providers():
    if not core._CudaGraphMemoryPool.available():
        pytest.skip("CUDA generation-owned pools unavailable")
    return (GraphRuntimeAssemblyProvider(), GraphResourceLifetimeRecipeProvider())


def _arguments(groups=2, size=131):
    return {
        name: value
        for index in range(groups)
        for name, value in ((f"value_{index}", index * 5 + 1), (f"output_{index}", ti.ndarray(ti.i32, size)))
    }


def _check(arguments, groups=2, size=131):
    for index in range(groups):
        np.testing.assert_array_equal(
            arguments[f"output_{index}"].to_numpy(),
            (arguments[f"value_{index}"] + np.arange(size, dtype=np.int32) * 3) * 2 + 7,
        )


def test_resource_alias_usage_closes_structural_regions_without_claiming_static_resources():
    first = ti.graph.GraphOwnedNdarray(ti.i32, (17,))
    second = ti.graph.GraphOwnedNdarray(ti.i32, (29,))
    root = SequentialRegion(
        children=(
            SequentialRegion(
                children=(
                    DispatchNode("first", bindings=(RuntimeBinding("first", "ndarray"),)),
                    NativeCallNode(
                        "external", effects=(ResourceEffect(np.arange(3), GraphAccess.READ, runtime_bound=False),)
                    ),
                )
            ),
            SequentialRegion(
                children=(
                    DispatchNode("alias", bindings=(RuntimeBinding("alias", "ndarray"),)),
                    DispatchNode("second", bindings=(RuntimeBinding("second", "ndarray"),)),
                )
            ),
        )
    )
    nodes = resource_lifetime._nodes_by_path(root)
    definition = SimpleNamespace(
        _runtime_spec=SimpleNamespace(
            fixed_runtime_args={"first": first, "alias": first, "second": second},
            temporary_memory_plan=SimpleNamespace(allocations=()),
        ),
        sources=tuple(
            SimpleNamespace(path=path, region_id=path)
            for path, node in nodes.items()
            if node.kind in ("dispatch", "native_call")
        ),
        regions=tuple(SimpleNamespace(path=path, region_id=path) for path in nodes),
    )
    groups = resource_lifetime._resource_groups(definition, nodes)
    assert len(groups) == 1
    assert groups[0].bindings == ("alias", "first", "second")
    assert groups[0].private_bytes == (17 + 29) * 4
    assert groups[0].coverage == tuple(nodes)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_generation_storage_composes_with_an_independent_reduction_family():
    definition = _definition(groups=1, size=4097, reduction=True)
    catalog = definition.recipe_catalog(providers=(*_providers(), GraphReductionRecipeProvider()))
    storage = next(
        fragment for fragment in catalog.fragments if fragment.provider_namespace.endswith(".resource_lifetime")
    )
    reduction = next(
        fragment for fragment in catalog.fragments if fragment.provider_namespace.endswith(".graph_reduction")
    )
    assert set(storage.coverage_region_ids).isdisjoint(reduction.coverage_region_ids)
    recipe = catalog.composer.compose((storage, reduction))
    arguments = _arguments(groups=1, size=4097)
    arguments.update(values=ti.ndarray(ti.i32, 4097), total=ti.ndarray(ti.i32, 1))
    values = np.arange(4097, dtype=np.int32) % 7
    arguments["values"].from_numpy(values)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            binding = graph.bind(arguments)
            for _ in range(3):
                graph.run(binding)
            _check(arguments, groups=1, size=4097)
            assert arguments["total"].to_numpy()[0] == values.sum()
            assert len(graph.execution_stats().storage_pools) == 1
            assert materialized.manifest.planned_physical_id == recipe.planned_physical_id


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_recording_partition_preserves_private_storage_shared_across_segments():
    definition = _definition(groups=4)
    catalog = definition.recipe_catalog(
        providers=(GraphRuntimeAssemblyProvider(), GraphRecordingPartitionRecipeProvider())
    )
    frontier = next(
        source for source in definition._runtime_spec._graph_recording_partition_sources if source.cut_index == 1
    )
    fragment = next(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_metadata["family_selection"]["source_key"] == frontier.source_key
    )
    recipe = catalog.composer.compose((fragment,))
    arguments = _arguments(groups=4)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            first, second = graph._spec.nodes
            assert first.fixed_runtime_args["scratch_0"] is second.fixed_runtime_args["scratch_0"]
            assert len(graph._instance._internal_storages) == 4
            graph.run(graph.bind(arguments))
            _check(arguments, groups=4)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_resource_usage_components_compose_and_have_actual_distinct_pool_topology():
    definition = _definition()
    catalog = definition.recipe_catalog(providers=_providers())
    assert sorted(len(fragment.coverage_region_ids) for fragment in catalog.fragments) == [2, 2, 4]
    separate = tuple(fragment for fragment in catalog.fragments if len(fragment.coverage_region_ids) == 2)
    composed = catalog.composer.compose(separate)
    joined = catalog.composer.compose(
        tuple(fragment for fragment in catalog.fragments if len(fragment.coverage_region_ids) == 4)
    )
    assert any(
        fragment.provider_namespace.endswith(".resource_lifetime") for fragment in definition.recipe_catalog().fragments
    )
    arguments = _arguments()
    physical_ids = []
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        for recipe, expected_pools in ((catalog.baseline.recipe, 0), (composed, 2), (joined, 1)):
            with context.materialize(recipe) as materialized:
                graph = materialized.executor
                binding = graph.bind(arguments)
                graph.run(binding)
                _check(arguments)
                before = graph.execution_stats()
                warm_manifest = observe_graph_physical_manifest(definition, recipe, graph)
                assert len(before.storage_pools) == expected_pools
                physical_ids.append(materialized.manifest.materialized_physical_id)
                assert sum(pool.requested_bytes for pool in before.storage_pools) == (
                    2 * 131 * 4 if expected_pools else 0
                )
                for pool in before.storage_pools:
                    assert pool.used_current_bytes == pool.requested_bytes
                    assert pool.reserved_current_bytes >= pool.used_current_bytes
                    assert pool.release_threshold_bytes == 0
                # After one warm run, replay must not create another allocation
                # or consult an owner-level probe/report/allocation method.
                allocation_calls = core.query_int64("cuda_async_allocation_calls")
                owners = graph._instance._storage_owners
                for _ in range(17):
                    graph.run(binding)
                _check(arguments)
                assert core.query_int64("cuda_async_allocation_calls") == allocation_calls
                after_manifest = observe_graph_physical_manifest(definition, recipe, graph)
                assert after_manifest.materialized_physical_id == warm_manifest.materialized_physical_id
                assert [
                    resource for resource in warm_manifest.resources if resource.kind == "cuda_generation_pool"
                ] == [
                    resource for resource in materialized.manifest.resources if resource.kind == "cuda_generation_pool"
                ]
                pool_resources = [
                    resource for resource in after_manifest.resources if resource.kind == "cuda_generation_pool"
                ]
                assert len(pool_resources) == expected_pools
                assert sum(resource.allocation_count for resource in pool_resources) == (2 if expected_pools else 0)
                assert sorted(member for resource in pool_resources for member in resource.allocation_members) == (
                    ["private:scratch_0", "private:scratch_1"] if expected_pools else []
                )
                report = asdict(graph.execution_stats())
                if expected_pools:
                    assert "reserved_current_bytes" in json.dumps(report)
                else:
                    assert report["storage_pools"] == ()
            assert all(owner.storage_pool_report().closed for owner in owners)
            assert all(owner.storage_pool_report().used_current_bytes is None for owner in owners)
    assert len(set(physical_ids)) == 3


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_resource_recipe_fork_search_and_fresh_definition_resolve(monkeypatch):
    from taichi_forge.graph import _trial_observations

    definition = _definition(groups=1)
    providers = _providers()
    arguments = _arguments(groups=1)
    session = definition.search_recipes(
        providers=providers,
        target=ti.graph.GraphOptimizationTarget(objectives=(("isolated_generation", "max"),)),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    )
    observed = set()
    original_snapshot = _trial_observations._resource_boundary

    def evaluate(graph, request):
        def unexpected_replay_observation(*args, **kwargs):
            raise AssertionError("search observation entered Graph replay")

        with monkeypatch.context() as patch:
            patch.setattr(_trial_observations, "_resource_boundary", unexpected_replay_observation)
            bound = graph.bind(arguments)
            for _ in range(3):
                graph.run(bound)
        _check(arguments, groups=1)
        observed.add(request.recipe_id)
        # Deterministic interface contract, explicitly not a speed benchmark.
        return {"isolated_generation": float(bool(graph.execution_stats().storage_pools))}

    decision = session.run(evaluate)
    assert decision.status == "selected", decision.report.results
    assert decision.report.search_complete
    assert len(observed) == 2
    assert "resident VRAM" in json.dumps(decision.report.recipe_annotations)
    assert _trial_observations._resource_boundary is original_snapshot
    for annotation in decision.report.recipe_annotations:
        (trial,) = annotation["trial_boundaries"]
        assert not trial["trial_failed"]
        assert trial["cleanup_status"] == "complete"
        assert trial["after_evaluator_status"] == "observed"
        cold = trial["after_materialization"]
        evaluated = trial["after_evaluator"]
        # Binding lazily creates Graph argument storage for both recipes. This
        # must remain visible rather than being reported as a cold memory peak.
        assert evaluated["persistent_allocated_bytes"] > cold["persistent_allocated_bytes"]
        assert evaluated["materialized_physical_id"] in annotation["measurement"]["materialized_physical_ids"]
        assert all(value >= 0 for value in trial["host_wall_seconds"].values())
    assert "After materialization bytes" in decision.report.to_markdown()
    assert "Peak bytes" not in decision.report.to_markdown()
    assert '"storage_allocator": "cuda_generation_pool"' in decision.report.to_markdown()
    fresh = _definition(groups=1)
    selection = fresh.resolve_recipe(decision.selection_artifact, providers=providers)
    with fresh.materialize(selection) as materialized:
        evaluate(materialized.executor, selection)
        assert len(materialized.executor.execution_stats().storage_pools) == 1


class _TemporaryAction(DispatchGraphAction):
    def __init__(self, dispatches, symbol):
        super().__init__(dispatches, conditional_body_safe=True)
        self.symbol = symbol

    @property
    def temporary_bindings(self):
        return {self.symbol: "scratch"}

    def bind_graph_temporaries(self, temporaries):
        assert temporaries["scratch"].offset == 0
        return {self.symbol: temporaries["scratch"].storage}


class _TemporaryWork(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, size):
        @ti.kernel
        def stage(scratch: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in scratch:
                scratch[i] = i * 3 + 11

        @ti.kernel
        def finish(scratch: ti.types.ndarray(dtype=ti.i32, ndim=1), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
            for i in output:
                output[i] = scratch[i] * 2 + 5

        scratch = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "__arena_scratch", ti.i32, ndim=1)
        output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
        self.action = _TemporaryAction(
            (
                (gen_cpp_kernel(stage, (scratch,)), (scratch,)),
                (gen_cpp_kernel(finish, (scratch, output)), (scratch, output)),
            ),
            scratch.name,
        )
        self.size = size

    def compile(self):
        return self

    @property
    def runtime_arg_schema(self):
        return (RuntimeBinding("output", "ndarray"),)

    @property
    def resource_effects(self):
        return (ResourceEffect("output", GraphAccess.WRITE),)

    @property
    def temporary_requirements(self):
        return (TemporaryRequirement("scratch", self.size * 4, 16),)

    @property
    def recordable_action(self):
        return self.action

    def run(self, runtime_args=None):
        raise AssertionError("recordable workspace was not lowered into the Graph")


class _TaggedTemporaryWork(_TemporaryWork):
    def __init__(self, size):
        @ti.kernel
        def stage(scratch: ti.types.ndarray(dtype=ti.i32, ndim=1), tag: ti.i32):
            for i in scratch:
                scratch[i] = i * 3 + tag

        @ti.kernel
        def finish(
            scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
            output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        ):
            for i in output:
                output[i] = scratch[i] * 2 + 5

        scratch = ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "__arena_scratch", ti.i32, ndim=1
        )
        output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
        tag = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "tag", ti.i32)
        self.action = _TemporaryAction(
            (
                (gen_cpp_kernel(stage, (scratch, tag)), (scratch, tag)),
                (gen_cpp_kernel(finish, (scratch, output)), (scratch, output)),
            ),
            scratch.name,
        )
        self.size = size

    @property
    def runtime_arg_schema(self):
        return (RuntimeBinding("output", "ndarray"), RuntimeBinding("tag", "scalar"))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_ordered_temporary_recipe_reuses_one_cold_slot_across_inflight_bindings(
    monkeypatch,
):
    from taichi_forge.graph._graph import _GraphTemporaryArena

    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "4")
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(65537))
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(
        f
        for f in catalog.fragments
        if f.fragment_key.endswith(":queue-ordered-storage")
    )
    recipe = catalog.compose((fragment.fragment_id,), stage="ordered-contract").recipe
    outputs = [ti.ndarray(ti.i32, 65537) for _ in range(9)]
    with definition.materialize(recipe, providers=_providers()) as materialized:
        graph = materialized.executor
        bindings = [
            graph.bind(dict(output=output, tag=index * 17 - 51))
            for index, output in enumerate(outputs)
        ]
        before = graph.execution_stats()
        assert before.memory.temporary_arena_slots == 1
        assert before.storage_pools[0].requested_bytes == 65537 * 4

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Ordered replay touched arena allocation, polling or binding callbacks"
            )

        with monkeypatch.context() as replay:
            replay.setattr(_GraphTemporaryArena, "_new_slot", forbidden)
            replay.setattr(_GraphTemporaryArena, "_reclaim", forbidden)
            replay.setattr(_TemporaryAction, "bind_graph_temporaries", forbidden)
            tickets = [graph.submit(binding) for binding in bindings]
            # Mixing run and submit must preserve the same queue order.
            graph.run(bindings[0])
            for ticket in reversed(tickets):
                ticket.wait()
        after = graph.execution_stats()
        assert after.memory.temporary_arena_waits == 0
        assert after.memory.temporary_arena_allocations == 1
        assert after.storage_pools[0].allocation_count == 1
        graph.submit(bindings[-1])  # Retire the Graph while work may remain.
    for index, output in enumerate(outputs):
        np.testing.assert_array_equal(
            output.to_numpy(),
            (np.arange(65537, dtype=np.int32) * 3 + index * 17 - 51) * 2 + 5,
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_ordered_temporary_final_executor_and_callback_contracts_are_cold(monkeypatch):
    from copy import copy
    from taichi_forge.graph._recipes.runtime_storage import (
        ordered_temporary_reuse_eligible,
    )

    builder = ti.graph.GraphBuilder()
    builder.append_native(_TemporaryWork(257))
    definition = builder.freeze()
    spec = definition._runtime_spec
    assert ordered_temporary_reuse_eligible(spec)
    for field, value in (
        ("_binding_executor_factory", lambda instance: None),
        ("runtime_lifetime_leases", (object(),)),
        ("nodes", (*spec.nodes, *spec.nodes)),
    ):
        altered = copy(spec)
        setattr(altered, field, value)
        assert not ordered_temporary_reuse_eligible(altered)
    altered = copy(spec)
    node = copy(spec.nodes[0])
    node.parallel_dispatch_groups = ((0,), (1,))
    altered.nodes = (node,)
    assert not ordered_temporary_reuse_eligible(altered)
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(
        f
        for f in catalog.fragments
        if f.fragment_key.endswith(":queue-ordered-storage")
    )
    recipe = catalog.compose(
        (fragment.fragment_id,), stage="cold-binding-contract"
    ).recipe
    foreign = ti.ndarray(ti.i32, 257)
    with monkeypatch.context() as invalid:
        invalid.setattr(
            _TemporaryAction,
            "bind_graph_temporaries",
            lambda action, bindings: {action.symbol: foreign},
        )
        with pytest.raises(
            (ValueError, RuntimeError), match="generation-owned storage"
        ):
            definition.materialize(recipe, providers=_providers())
    with definition.materialize(recipe, providers=_providers()) as selected:
        selected.executor.run(dict(output=foreign))
        np.testing.assert_array_equal(
            foreign.to_numpy(), (np.arange(257, dtype=np.int32) * 3 + 11) * 2 + 5
        )
    observed = set()

    def evaluate(graph, request):
        graph.run(dict(output=foreign))
        np.testing.assert_array_equal(
            foreign.to_numpy(), (np.arange(257, dtype=np.int32) * 3 + 11) * 2 + 5
        )
        observed.add(request.recipe_id)
        return {
            "temporary_bytes": float(
                graph._instance.temporary_arena_stats["reserved_bytes"]
            )
        }

    decision = definition.search_recipes(
        providers=_providers(),
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("temporary_bytes", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
        strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
    ).run(evaluate)
    assert decision.status == "selected" and decision.report.search_complete
    assert recipe.recipe_id in observed and len(observed) == 3
    assert "runtime_stream_order" in decision.report.to_json()
    fresh_builder = ti.graph.GraphBuilder()
    fresh_builder.append_native(_TemporaryWork(257))
    fresh = fresh_builder.freeze()
    resolved = fresh.resolve_recipe(decision.selection_artifact, providers=_providers())
    with fresh.materialize(resolved) as restored:
        evaluate(restored.executor, resolved)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_resource_recipe_records_real_temporary_work_and_freezes_ring_capacity(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "3")
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TemporaryWork(257))
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(
        f
        for f in catalog.fragments
        if f.fragment_key.endswith(":generation-owned-storage")
    )
    recipe = catalog.compose((fragment.fragment_id,), stage="ring-contract").recipe
    output = ti.ndarray(ti.i32, 257)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(recipe) as materialized:
            graph = materialized.executor
            before = graph.execution_stats()
            assert before.memory.temporary_arena_slots == 3
            assert before.storage_pools[0].requested_bytes == 3 * 257 * 4
            for _ in range(13):
                graph.run(dict(output=output))
            np.testing.assert_array_equal(output.to_numpy(), (np.arange(257, dtype=np.int32) * 3 + 11) * 2 + 5)
            after = graph.execution_stats()
            assert after.memory.temporary_arena_allocations == 3
            assert after.storage_pools[0].allocation_count == 3
            assert after.storage_pools[0].allocation_members == ("temporary_arena",)
            assert any(
                resource.allocation_members == ("temporary_arena",) for resource in materialized.manifest.resources
            )
        monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "2")
        # A changed setup plan must not silently reuse the old recipe token.
        with pytest.raises((ValueError, RuntimeError, KeyError), match="unavailable|changed|resolve"):
            context.materialize(recipe)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_resource_recipe_resolution_builds_only_requested_fragment(monkeypatch):
    definition = _definition(groups=4)
    provider = GraphResourceLifetimeRecipeProvider()
    fragments = provider.fragments(definition)
    assert len(fragments) == 5
    original = resource_lifetime._fragment
    constructed = []

    def record_fragment(*args, **kwargs):
        constructed.append(kwargs["source_key"])
        return original(*args, **kwargs)

    monkeypatch.setattr(resource_lifetime, "_fragment", record_fragment)
    for fragment in fragments:
        constructed.clear()
        assert provider.resolve(definition, fragment.fragment_key).to_dict() == fragment.to_dict()
        assert constructed == [fragment.provider_metadata["family_selection"]["source_key"]]
    constructed.clear()
    with pytest.raises(KeyError, match="unavailable"):
        provider.resolve(definition, fragments[0].fragment_key + ":missing")
    assert not constructed
    monkeypatch.setattr(resource_lifetime, "_pool_available", lambda generation: False)
    with pytest.raises(KeyError, match="unavailable"):
        provider.resolve(definition, fragments[0].fragment_key)
    assert not constructed


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_resource_recipe_capability_miss_and_observation_failure_preserve_baseline(monkeypatch):
    from taichi_forge.graph._recipes import families

    definition = _definition(groups=1)
    catalog = definition.recipe_catalog(providers=_providers())
    recipe = next(entry.recipe for entry in catalog.entries() if entry.recipe.fragments)
    owners = []

    def fail_manifest(definition, recipe, graph):
        owners.extend(graph._instance._storage_owners)
        assert owners[-1].storage_pool_report().allocation_count == 1
        raise ValueError("injected resource observation failure")

    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with monkeypatch.context() as patch:
            patch.setattr(families, "observe_graph_physical_manifest", fail_manifest)
            with pytest.raises((ValueError, RuntimeError), match="injected resource observation failure"):
                context.materialize(recipe)
        assert owners[0].storage_pool_report().closed
        with context.materialize(recipe) as recovered:
            arguments = _arguments(groups=1)
            recovered.executor.run(arguments)
            _check(arguments, groups=1)
    monkeypatch.setattr(resource_lifetime, "_pool_available", lambda generation: False)
    assert GraphResourceLifetimeRecipeProvider().fragments(definition) == ()
    ordinary = definition.compile()
    arguments = _arguments(groups=1)
    ordinary.run(arguments)
    _check(arguments, groups=1)
    assert ordinary.execution_stats().storage_pools == ()
