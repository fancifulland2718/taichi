"""Certified value substitution across a fixed segmented-reduction region.

Adjacency identifies possible regions, not dataflow. Every replacement carries
a publication-time binding proof. The compiler owns the executable value DAG;
this provider owns coverage, storage, phase ordering and complete recipes.
"""

from dataclasses import dataclass

from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _subtree_regions,
    _workspace_resource,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask


@dataclass(frozen=True)
class _Map:
    dispatch: object
    program: dict

    def name(self, argument):
        return self.dispatch.args[argument].name

    @property
    def output(self):
        return self.name(self.program["output_argument"])

    @property
    def inputs(self):
        return tuple(
            sorted(
                {
                    n["argument"]
                    for n in self.program["nodes"]
                    if n["kind"] == "array_load"
                }
            )
        )


@dataclass(frozen=True)
class _Choice:
    source: object
    manifest: object
    producer: _Map | None
    consumer: _Map | None
    consumer_input: int
    first: int
    last: int
    submission: str

    @property
    def facts(self):
        return {
            "reduction": self.manifest.to_dict(),
            "producer": None if self.producer is None else self.producer.program,
            "consumer": None if self.consumer is None else self.consumer.program,
            "consumer_input": self.consumer_input,
            "submission": self.submission,
            "binding_contract": {
                "producer_output": (
                    None if self.producer is None else self.producer.output
                ),
                "consumer_input": (
                    None
                    if self.consumer is None
                    else self.consumer.name(self.consumer_input)
                ),
                "identity": "fixed_reduction_values_and_output",
                "coverage": "exact_zero_based_domain_including_observable_stores",
                "storage": "disjoint_writes_and_unforwarded_reads_plain_ndarrays",
                "validation": "binding_publication",
            },
        }

    @property
    def choice_id(self):
        return "segmented-values:" + _digest(self.facts)


def _map_node(node):
    from taichi_forge.graph._graph import _CompiledCGraphNode

    return (
        isinstance(node, _CompiledCGraphNode)
        and bool(node.recording_dispatches)
        and len(node.recording_dispatches) == node.dispatch_count
        and not node.source_native_count
        and not node.fixed_runtime_args
        and not node.temporary_actions
        and not node.lifetime_leases
        and not node.parallel_dispatch_groups
        and not node.snode_tree_dependency_info
        and all(op[0] == "dispatch" for op in node.recipe_operations)
        and all(item.dispatch_packet is None for item in node.recording_dispatches)
    )


def _inspect(dispatch):
    from taichi_forge._lib import core

    program = core._graph_pointwise_value_program(dispatch.kernel)
    return _Map(dispatch, program) if program["available"] else None


class _BindingProof:
    # Existing Graph publication snapshots pin ndarrays/scalars. No provider
    # callbacks are needed for their subsequent steady replay.
    graph_publish_time_binding_validation_stable = True
    graph_runtime_lifetime_check_required = False

    def __init__(self, choice, owned):
        self.choice, self.owned = choice, tuple(owned)

    def validate_graph_bindings(self, args):
        from taichi_forge.lang._ndarray import ScalarNdarray

        choice = self.choice
        source = choice.source
        producer, consumer = choice.producer, choice.consumer
        if producer is not None and args[producer.output] is not source.values:
            raise ValueError(
                "value fusion requires producer output to bind the fixed reduction values"
            )
        if (
            consumer is not None
            and args[consumer.name(choice.consumer_input)] is not source.output
        ):
            raise ValueError(
                "value fusion requires consumer input to bind the fixed reduction output"
            )

        reads = []
        writes = (
            [source.values, source.output] if producer is not None else [source.output]
        )
        if producer is None:
            reads.append(source.values)
        reads.append(source.layout._offsets)
        for mapping, count, forwarded in (
            (producer, source.layout.num_items, -1),
            (consumer, source.layout.num_segments, choice.consumer_input),
        ):
            if mapping is None:
                continue
            domain = mapping.program["metadata"]["iteration_domain"]
            kind = domain["kind"]
            if kind == "constant_range":
                end = domain["end"]
            else:
                value = args[mapping.name(domain["arg_id"][0])]
                if kind == "scalar_argument":
                    end = value
                elif kind == "external_tensor" and isinstance(value, ScalarNdarray):
                    end = value.shape[0]
                else:
                    raise ValueError(
                        "value fusion has an unsupported iteration binding"
                    )
            if domain["begin"] != 0 or end != count:
                raise ValueError(
                    "value fusion requires exact iteration coverage; observable tail stores cannot be discarded"
                )
            for argument in {*mapping.inputs, mapping.program["output_argument"]}:
                array = args[mapping.name(argument)]
                if (
                    not isinstance(array, ScalarNdarray)
                    or len(array.shape) != 1
                    or array.shape[0] < count
                ):
                    raise ValueError(
                        "value fusion requires capacity-matched plain scalar ndarrays"
                    )
            reads.extend(
                args[mapping.name(index)]
                for index in mapping.inputs
                if index != forwarded
            )
            if mapping is consumer:
                writes.append(args[mapping.output])

        # Plain ndarrays own allocations. Read/read aliases remain legal; any
        # extra write/read relation would need a different forwarding proof.
        # Native publication separately pins and validates the actual ABI.
        def aliases(left, right):
            return left is right or left.arr is right.arr

        if any(
            aliases(left, right)
            for i, left in enumerate(writes)
            for right in writes[i + 1 :]
        ) or any(aliases(write, read) for write in writes for read in reads):
            raise ValueError(
                "value fusion has an unforwarded storage alias across reduction phases"
            )

    def _graph_provider_memory_report(self):
        from taichi_forge.hardware._memory import (
            HardwareMemoryComponent,
            make_memory_report,
        )

        return make_memory_report(
            "graph_value_fusion",
            "cuda",
            tuple(
                HardwareMemoryComponent(
                    name,
                    int(array.shape[0]) * 4,
                    True,
                    "provider_generation",
                    "provider",
                    resident=True,
                )
                for name, array in self.owned
            ),
            ownership_scope="graph_native_action",
        )

    def _graph_provider_memory_identity(self):
        return ("graph_value_fusion", id(self))


def _materialize(spec, choice):
    from taichi_forge._lib import core
    from taichi_forge.graph._graph import (
        Arg,
        ArgKind,
        _CompiledCGraphNode,
        _RecordingDispatch,
        gen_cpp_kernel,
    )
    from taichi_forge.graph._segmented_reduce import _BLOCK_DIMS, _PARTIAL
    from taichi_forge.graph._segmented_reduce_kernels import reduction_kernel
    from taichi_forge.lang import impl
    from taichi_forge.types.primitive_types import i32

    source, producer, consumer = choice.source, choice.producer, choice.consumer
    # Private symbols are deterministic but cannot collide with public names.
    prefix = "__forge_segmented_values_" + str(source._recipe_node_index) + "_"
    while any(name.startswith(prefix) for name in spec.runtime_arg_names):
        prefix += "_"
    fixed = {}

    def private(name, value):
        name = prefix + name
        fixed[name] = value
        return name

    values = producer.output if producer else private("values", source.values)
    output = (
        consumer.name(choice.consumer_input)
        if consumer
        else private("output", source.output)
    )
    owned = []
    if choice.manifest.strategy == _PARTIAL:
        import numpy as np

        tiles, ends = source.partial_layout
        for name, data in (("tiles", tiles), ("ends", ends)):
            array = impl.ndarray(i32, shape=len(data))
            array.from_numpy(np.asarray(data, np.int32))
            owned.append((name, array))
        owned.append(
            ("partials", impl.ndarray(source.values.dtype, shape=len(tiles) - 1))
        )
        tile_name, end_name, scratch = (private(name, array) for name, array in owned)
        stages = (
            (values, tile_name, scratch, len(tiles) - 1, 128, producer, None),
            (scratch, end_name, output, source.layout.num_segments, 32, None, consumer),
        )
    else:
        stages = (
            (
                values,
                private("offsets", source.layout._offsets),
                output,
                source.layout.num_segments,
                _BLOCK_DIMS[choice.manifest.strategy],
                producer,
                consumer,
            ),
        )
    recordings = []
    if producer:
        recordings.extend(spec.nodes[choice.first].recording_dispatches[:-1])
    for (
        input_name,
        offsets_name,
        output_name,
        count,
        block_dim,
        before,
        after,
    ) in stages:
        arguments = [
            Arg(ArgKind.NDARRAY, name, dtype, ndim=1)
            for name, dtype in (
                (input_name, source.values.dtype),
                (offsets_name, i32),
                (output_name, source.values.dtype),
            )
        ]
        kernel = gen_cpp_kernel(
            reduction_kernel(source.values.dtype, count, block_dim=block_dim), arguments
        )
        if before is None and after is None:
            recordings.append(_RecordingDispatch(kernel, tuple(arguments)))
            continue
        compiled = core._compile_graph_segmented_reduce_values(
            impl.get_runtime().prog,
            kernel,
            arguments,
            before.dispatch.kernel if before else None,
            before.dispatch.args if before else (),
            after.dispatch.kernel if after else None,
            after.dispatch.args if after else (),
            choice.consumer_input if after else -1,
        )
        # Reference-internal kernel handles retain their owning synthetic Graph
        # after these dispatches are recomposed with unaffected prefix/suffix.
        recordings.extend(
            _RecordingDispatch(kernel, tuple(args))
            for kernel, args in compiled._owned_jit_dispatch_sources
        )
    if consumer:
        recordings.extend(spec.nodes[choice.last].recording_dispatches[1:])
    builder = core.GraphBuilder()
    for dispatch in recordings:
        builder.dispatch(dispatch.kernel, dispatch.args)
    return _CompiledCGraphNode(
        builder.compile(),
        len(recordings),
        {arg.name for dispatch in recordings for arg in dispatch.args},
        recording_dispatches=recordings,
        fixed_runtime_args=fixed,
        lifetime_leases=(_BindingProof(choice, owned),),
    )


class GraphValueFusionRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "value_fusion",
        capabilities=(
            "certified-pointwise-values",
            "segmented-reduction-phases",
            "cold-binding-proof",
        ),
        domain_version="segmented-value-fusion-v1",
        semantic_fingerprint="integer32-preserved-stores-exact-domains-disjoint-storage-v1",
    )

    def materialize(self, scope, fragment):
        # One ordered CUDA submission stream reuses this Graph's private
        # scratch without a host completion wait. Do not turn a fixed lane
        # restriction into an exclusive-workspace lease on every replay.
        if scope._context.workspace_lanes != 1:
            raise ValueError("value fusion requires one ordered workspace lane")
        return super().materialize(scope, fragment)

    def _choices(self, definition):
        from taichi_forge._lib import core
        from taichi_forge.graph._segmented_reduce import _SegmentedReductionSource
        from taichi_forge.lang import impl

        # Provider-local discovery memo, never a process-global executable cache.
        if getattr(self, "_definition", None) is definition:
            return self._discovered
        choices = []
        spec = definition._runtime_spec
        if (
            definition.backend == "cuda"
            and not impl.current_cfg().debug
            and not spec.snode_tree_dependency_info
        ):
            for source in spec._graph_native_algorithm_sources:
                if not isinstance(source, _SegmentedReductionSource):
                    continue
                index = source._recipe_node_index
                producer = (
                    _inspect(spec.nodes[index - 1].recording_dispatches[-1])
                    if index > 0 and _map_node(spec.nodes[index - 1])
                    else None
                )
                consumer = (
                    _inspect(spec.nodes[index + 1].recording_dispatches[0])
                    if index + 1 < len(spec.nodes) and _map_node(spec.nodes[index + 1])
                    else None
                )
                modes = [(producer, None, -1)] if producer else []
                if consumer:
                    for role in consumer.inputs:
                        # Compiler still verifies exact merged ABI during build.
                        modes.append((None, consumer, role))
                        if producer:
                            modes.append((producer, consumer, role))
                for before, after, role in modes:
                    first, last = index - bool(before), index + bool(after)
                    submissions = ["cached_launch"]
                    native = getattr(core, "_CudaGraphBindingExecutor", None)
                    if (
                        first == 0
                        and last == len(spec.nodes) - 1
                        and not impl.current_cfg().kernel_profiler
                        and native is not None
                        and native.available()
                        and native.retains_completion_events_until_close()
                    ):
                        submissions.append("immutable_frames")
                    for manifest in source.manifests():
                        for submission in submissions:
                            choices.append(
                                _Choice(
                                    source,
                                    manifest,
                                    before,
                                    after,
                                    role,
                                    first,
                                    last,
                                    submission,
                                )
                            )
        self._definition, self._discovered = definition, tuple(choices)
        return self._discovered

    def fragments(self, definition):
        result = []
        for choice in self._choices(definition):
            coverage = tuple(
                region_id
                for region in definition.regions
                if region.parent_region_id == definition.regions[0].region_id
                and any(
                    region.path.startswith(f"graph/{index}:")
                    for index in range(choice.first, choice.last + 1)
                )
                for region_id in _subtree_regions(definition, region.region_id)
            )
            result.append(
                _fragment(
                    definition,
                    family="value_fusion",
                    source_key=choice.source._recipe_source_key,
                    choice_id=choice.choice_id,
                    coverage=coverage,
                    tasks=(
                        GraphFragmentTask.create(
                            choice.choice_id + ":region",
                            "certified_segmented_value_region",
                            physical=choice.facts,
                        ),
                    ),
                    resources=_workspace_resource(
                        choice.source._recipe_source_key, choice.manifest
                    ),
                    exclusive_submission=True,
                    provider_descriptor=self.descriptor,
                )
            )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        choice = next(
            (
                item
                for item in self._choices(assembly.definition)
                if item.choice_id == selection.choice_id
                and item.source._recipe_source_key == selection.source_key
            ),
            None,
        )
        if choice is None:
            raise ValueError("certified value-fusion recipe is unavailable")
        replacement = _materialize(assembly.spec, choice)
        assembly.expand_node(
            choice.first,
            lambda _node: (replacement,),
            source_node_indices=range(choice.first, choice.last + 1),
        )
        for index in range(choice.first + 1, choice.last + 1):
            assembly.expand_node(index, lambda _node: ())
        if choice.submission == "immutable_frames":
            from taichi_forge.graph._recipes.binding_frames import _BindingFrameExecutor

            assembly.select_binding_executor(_BindingFrameExecutor)

    def describe(self, definition, fragment_key):
        metadata = super().describe(definition, fragment_key)
        selected = metadata["family_selection"]["choice_id"]
        choice = next(
            item for item in self._choices(definition) if item.choice_id == selected
        )
        return {
            **metadata,
            "display_name": "Certified pointwise / segmented reduction value fusion",
            "physical_changes": choice.facts,
            "limitations": (
                "CUDA scalar i32/u32 pointwise value DAGs; no AD or debug instrumentation",
                "binding must prove exact domains and the fixed reduction array identities",
                "all visible stores retained; views, in-place writes and unforwarded write aliases unavailable",
                "whole affected nodes are reserved, including unchanged prefix/suffix dispatches",
                "immutable frames only for one complete CGraph and one workspace lane; raw mappings prepare arguments",
                "performance and driver-owned VRAM require workload evidence",
            ),
        }
