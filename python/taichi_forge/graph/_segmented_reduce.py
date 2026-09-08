"""Complete segmented-reduction recipes; no new scheduler or raw launch axis."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._algorithms import _check_segmented_request
from taichi_forge.algorithms._autodiff import is_fwd_mode_active, is_tape_active
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    SequentialRegion,
)
from taichi_forge.graph._native import (
    BackendCommandPlan,
    NativeGraphExecutable,
    NativeGraphNode,
)
from taichi_forge.graph._segmented_reduce_kernels import reduction_kernel
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, u32


_SERIAL = "segment_serial"
_WARP = "warp_segment"
_BLOCK = "block_segment"
_PARTIAL = "chunk_partial_finalize"
_CHUNK_ITEMS = 1024
_BLOCK_DIMS = {_SERIAL: 0, _WARP: 32, _BLOCK: 128, _PARTIAL: 128}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _implementation_id():
    # Cold identity, never a replay check or a wheel/HEAD compatibility pin.
    root = Path(__file__).parent
    return "sha256:" + _hash(
        {
            name: hashlib.sha256(
                (root / name).read_text(encoding="utf-8").encode()
            ).hexdigest()
            for name in ("_segmented_reduce.py", "_segmented_reduce_kernels.py")
        }
    )


@dataclass(frozen=True)
class _ReductionManifest:
    """Provider-owned facts, reconstructed from a live equivalent definition."""

    recipe_id: str
    payload_json: str

    @classmethod
    def create(cls, payload):
        return cls("graph-segmented-reduce:" + _hash(payload), _json(payload))

    @property
    def strategy(self):
        return json.loads(self.payload_json)["strategy"]

    def to_dict(self):
        return {"recipe_id": self.recipe_id, **json.loads(self.payload_json)}


class _ReductionExecutable(NativeGraphExecutable):
    exclusive_graph_submission = True
    graph_runtime_lifetime_check_required = False

    def __init__(self, source, manifest):
        from taichi_forge.graph._graph import Arg, ArgKind, GraphBuilder

        self._source = source
        self._manifest = manifest
        self._owned = ()
        self._graph = None
        builder = GraphBuilder(
            _capture_recipe_sources=False, _explicit_map_source_groups=()
        )
        dtype = source.values.dtype
        values = Arg(ArgKind.NDARRAY, "values", dtype, ndim=1)
        offsets = Arg(ArgKind.NDARRAY, "offsets", i32, ndim=1)
        output = Arg(ArgKind.NDARRAY, "output", dtype, ndim=1)
        bindings = {
            "values": source.values,
            "offsets": source.layout._offsets,
            "output": source.output,
        }
        strategy = manifest.strategy
        if strategy == _PARTIAL:
            import numpy as np

            tile_bounds, partial_bounds = source.partial_layout
            tile_offsets = impl.ndarray(i32, shape=len(tile_bounds))
            final_offsets = impl.ndarray(i32, shape=len(partial_bounds))
            partials = impl.ndarray(dtype, shape=len(tile_bounds) - 1)
            tile_offsets.from_numpy(np.asarray(tile_bounds, dtype=np.int32))
            final_offsets.from_numpy(np.asarray(partial_bounds, dtype=np.int32))
            self._owned = (tile_offsets, final_offsets, partials)
            tiles = Arg(ArgKind.NDARRAY, "tile_offsets", i32, ndim=1)
            ends = Arg(ArgKind.NDARRAY, "partial_offsets", i32, ndim=1)
            scratch = Arg(ArgKind.NDARRAY, "partials", dtype, ndim=1)
            builder.dispatch(
                reduction_kernel(dtype, len(tile_bounds) - 1, block_dim=128),
                values,
                tiles,
                scratch,
            )
            builder.dispatch(
                reduction_kernel(dtype, source.layout.num_segments, block_dim=32),
                scratch,
                ends,
                output,
            )
            # The staged route reads frozen tile boundaries, not the original
            # offsets. Do not bind an argument absent from its compiled ABI.
            bindings.pop("offsets")
            bindings.update(
                tile_offsets=tile_offsets,
                partial_offsets=final_offsets,
                partials=partials,
            )
        else:
            builder.dispatch(
                reduction_kernel(
                    dtype, source.layout.num_segments, block_dim=_BLOCK_DIMS[strategy]
                ),
                values,
                offsets,
                output,
            )
        self._graph = builder.compile()
        self._bindings = self._graph.bind(bindings)

    def run(self):
        self._graph.run(self._bindings)

    @property
    def graph_physical_plan_id(self):
        return self._manifest.recipe_id

    @property
    def backend_command_plan(self):
        return BackendCommandPlan(
            backend="cuda",
            command_count=2 if self._manifest.strategy == _PARTIAL else 1,
            command_count_exact=True,
            provider_replay=False,
            fragmentation_reason="retained_segmented_reduction_graph",
        )

    @property
    def graph_ir_node(self):
        return self._source.semantic_root.children[0]

    @property
    def debug_info(self):
        return {
            "kind": "graph_segmented_reduce",
            "strategy": self._manifest.strategy,
            "action_owned_bytes": self._source.action_owned_bytes(
                self._manifest.strategy
            ),
            "nested_graph_replay": True,
        }

    def _graph_provider_memory_report(self):
        components = tuple(
            HardwareMemoryComponent(
                name,
                int(array.shape[0]) * 4,
                True,
                "provider_generation",
                "provider",
                resident=True,
            )
            for name, array in zip(
                ("tile_offsets", "partial_offsets", "partial_results"), self._owned
            )
        )
        return make_memory_report(
            "graph_segmented_reduce",
            "cuda",
            components,
            ownership_scope="graph_native_action",
        )

    def _graph_provider_memory_identity(self):
        return ("graph_segmented_reduce", id(self))


class _ReductionNode(NativeGraphNode):
    def __init__(self, source, manifest):
        self._source, self._manifest = source, manifest

    def compile(self):
        return _ReductionExecutable(self._source, self._manifest)


class _SegmentedReductionSource:
    def __init__(self, values, layout, output, *, operation):
        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise TaichiRuntimeError(
                "Graph segmented reduction recipes currently require CUDA"
            )
        if is_tape_active() or is_fwd_mode_active():
            raise TaichiRuntimeError(
                "Graph segmented reduction recipes do not support automatic differentiation"
            )
        if operation != "sum":
            raise ValueError("Graph segmented reduction currently supports op='sum'")
        ndarray_mode, _ = _check_segmented_request(
            "GraphBuilder.segmented_reduce()",
            values,
            layout,
            output,
            method="serial",
            workspace=None,
            scan=False,
        )
        if not ndarray_mode or values.dtype not in (i32, u32):
            raise TaichiRuntimeError(
                "Graph segmented reduction requires disjoint plain i32/u32 ndarrays"
            )
        if layout._offsets_host is None:
            raise TaichiRuntimeError(
                "Graph segmented reduction requires host-published immutable segment offsets"
            )
        self.values, self.layout, self.output = values, layout, output
        self.offsets = tuple(int(value) for value in layout._offsets_host)
        self.baseline_strategy = _SERIAL
        self.selected_recipe_id = self.selected_strategy = ""
        # Expand only when a segment actually spans multiple chunks. Small
        # segments do not allocate a redundant partial/finalize candidate.
        self.partial_layout = None
        if layout.max_segment_length > _CHUNK_ITEMS:
            tile_bounds, partial_bounds = [self.offsets[0]], [0]
            for begin, end in zip(self.offsets, self.offsets[1:]):
                tile_bounds.extend(
                    min(index + _CHUNK_ITEMS, end)
                    for index in range(begin, end, _CHUNK_ITEMS)
                )
                partial_bounds.append(len(tile_bounds) - 1)
            self.partial_layout = (tuple(tile_bounds), tuple(partial_bounds))
        strategies = [_SERIAL]
        if layout.max_segment_length > 1:
            strategies.extend((_WARP, _BLOCK))
        if self.partial_layout is not None:
            strategies.append(_PARTIAL)
        source_lock = _implementation_id()
        self._manifests = tuple(
            self._manifest(strategy, source_lock) for strategy in strategies
        )

    @property
    def semantics(self):
        return {
            "operation": "sum",
            "dtype": str(self.values.dtype),
            "associativity": "modular_integer_sum",
            "determinism": "exact",
            "empty_segment": "zero",
            "capacity": self.layout.capacity,
            "num_items": self.layout.num_items,
            "num_segments": self.layout.num_segments,
            "topology_fingerprint": "segmented-layout:" + _hash(self.offsets),
            "input": {"shape": list(self.values.shape), "fixed_resource": True},
            "output": {"shape": list(self.output.shape), "fixed_resource": True},
        }

    @property
    def semantic_root(self):
        return SequentialRegion(
            (
                NativeCallNode(
                    name="graph_segmented_reduce",
                    effects=(
                        ResourceEffect(
                            "fixed_segmented_reduce_input", GraphAccess.READ
                        ),
                        ResourceEffect(
                            "fixed_segmented_reduce_output", GraphAccess.WRITE
                        ),
                    ),
                    bindings=(),
                    temporaries=(),
                    opaque=True,
                ),
            ),
            name="graph",
        )

    def action_owned_bytes(self, strategy):
        if strategy != _PARTIAL:
            return 0
        tile_bounds, partial_bounds = self.partial_layout
        return 4 * (2 * len(tile_bounds) - 1 + len(partial_bounds))

    def _manifest(self, strategy, source_lock):
        stage_names = (
            ("chunk_partials", "segment_finalize")
            if strategy == _PARTIAL
            else (strategy,)
        )
        return _ReductionManifest.create(
            {
                "algorithm": "segmented_reduce",
                "strategy": strategy,
                "semantics": self.semantics,
                "physical_stages": [
                    {"name": name, "execution_kind": "taichi_dispatch", "call_count": 1}
                    for name in stage_names
                ],
                "topology": {
                    "kind": strategy,
                    "block_dim": _BLOCK_DIMS[strategy],
                    "chunk_items": _CHUNK_ITEMS if strategy == _PARTIAL else 0,
                    "partial_count": (
                        len(self.partial_layout[0]) - 1 if strategy == _PARTIAL else 0
                    ),
                },
                "workspace": {
                    "ownership": (
                        "graph_native_action" if strategy == _PARTIAL else "none"
                    ),
                    "action_owned_bytes": self.action_owned_bytes(strategy),
                    "provider_shared_scope": "none",
                },
                "submission": {
                    "resource_binding": "fixed_graph_action",
                    "exclusive": True,
                },
                "source_lock": source_lock,
            }
        )

    def manifests(self):
        return self._manifests

    def materialize(self, builder, requested_recipe_id=None, *, record_selection=True):
        manifest = self._manifests[0]
        if requested_recipe_id is not None:
            manifest = next(
                (
                    item
                    for item in self._manifests
                    if item.recipe_id == requested_recipe_id
                ),
                None,
            )
            if manifest is None:
                raise TaichiRuntimeError(
                    "segmented reduction recipe is absent from this definition"
                )
        builder._append_native(
            _ReductionNode(self, manifest), prewarm=False, admission="explicit"
        )
        if record_selection:
            self.selected_recipe_id, self.selected_strategy = (
                manifest.recipe_id,
                manifest.strategy,
            )
        return manifest


def append_graph_segmented_reduce(builder, values, layout, output, *, op="sum"):
    source = _SegmentedReductionSource(values, layout, output, operation=op)
    source.materialize(builder)
    return source
