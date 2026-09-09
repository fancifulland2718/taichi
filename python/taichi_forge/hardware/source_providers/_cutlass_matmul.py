"""CUTLASS-owned complete matmul regions over the existing addon C ABI."""

import ctypes
import hashlib
import math
import os
from pathlib import Path

from taichi_forge._lib import core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import (
    BackendCommandRecording,
    NativeGraphNode,
    _CudaGraphCaptureRecipe,
)
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import (
    GraphFragmentResourceRequirement,
    GraphFragmentTask,
)
from taichi_forge.hardware._matmul_recipe import _sources
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    HardwareRecordingExecutable,
    runtime_generation_matches,
    validate_runtime_generation,
)
from taichi_forge.hardware._retained import (
    RetainedExecutionContract,
    HardwareExecutionCostModel,
    attach_retained_execution_contract,
    fixed_cost,
    scale_cost,
    make_retained_plan_identity,
)
from taichi_forge.hardware._source_provider import load_source_provider_manifest
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.linalg._matmul import _shapes
from taichi_forge.types.primitive_types import f32, u8

_ABI = "taichi-forge-cutlass-matmul-c-abi1"
_STRATEGIES = ("direct_fused", "split_k_reduce_fused", "split_k_wide_reduce_fused")


class _Invocation(ctypes.Structure):
    _fields_ = (
        ("struct_size", ctypes.c_uint32),
        ("strategy", ctypes.c_uint32),
        ("transpose_a", ctypes.c_uint32),
        ("transpose_b", ctypes.c_uint32),
        ("m", ctypes.c_int32),
        ("n", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("activation", ctypes.c_int32),
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("workspace", ctypes.c_void_p),
        ("workspace_bytes", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
    )


def _payload(semantics, strategy):
    return _Invocation(
        struct_size=ctypes.sizeof(_Invocation),
        strategy=_STRATEGIES.index(strategy),
        transpose_a=semantics["transpose_a"],
        transpose_b=semantics["transpose_b"],
        m=semantics["m"],
        n=semantics["n"],
        k=semantics["k"],
        activation=semantics["activation"] == "relu",
        alpha=semantics["alpha"],
        beta=semantics["beta"],
    )


class _Library:
    def __init__(self, path):
        self.manifest = load_source_provider_manifest(
            path,
            expected_provider_id="cutlass_matmul",
            expected_provider_abi=_ABI,
        )
        if (
            self.manifest.provider_abi_version != 1
            or self.manifest.build_profile is None
        ):
            raise ValueError(
                "CUTLASS requires ABI 1 and an explicit addon build profile"
            )
        runtime = impl.get_runtime()
        if runtime.prog is None or runtime.prog.config().arch != core.cuda:
            raise TaichiRuntimeError(
                "CUTLASS preparation requires ti.init(arch=ti.cuda)"
            )
        self._runtime_prog = runtime.prog
        self._runtime_generation = int(impl.runtime_generation())
        self.compatibility = self.manifest.cuda_compatibility(
            int(impl.get_cuda_compute_capability()), core.cuda_driver_api_version()
        )
        if not self.compatibility["eligible"]:
            raise TaichiRuntimeError(f"CUTLASS addon unavailable: {self.compatibility}")
        self.dll = ctypes.CDLL(str(self.manifest.binary_path))
        if os.name == "nt":
            mapped = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
            mapped.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint)
            mapped.restype = ctypes.c_uint
            name = ctypes.create_unicode_buffer(32768)
            if not mapped(self.dll._handle, name, len(name)):
                raise ctypes.WinError(ctypes.get_last_error())
            if (
                hashlib.sha256(Path(name.value).read_bytes()).hexdigest()
                != self.manifest.binary_sha256
            ):
                raise TaichiRuntimeError(
                    "Loaded CUTLASS binary differs from the manifest"
                )
        abi = self.dll.ti_forge_cutlass_abi
        abi.argtypes, abi.restype = (), ctypes.c_uint32
        version = self.dll.ti_forge_cutlass_version
        version.argtypes, version.restype = (), ctypes.c_uint32
        dependencies = [
            x
            for x in self.manifest.toolchain["source_dependencies"]
            if x.name == "cutlass"
        ]
        if len(dependencies) != 1:
            raise ValueError(
                "CUTLASS manifest must bind exactly one header-tree identity"
            )
        major, minor, patch = (int(x) for x in dependencies[0].version.split("."))
        if abi() != 1 or version() != major * 10000 + minor * 100 + patch:
            raise TaichiRuntimeError(
                "CUTLASS ABI or source version disagrees with manifest"
            )
        self.query = self.dll.ti_forge_cutlass_query
        self.query.argtypes = (
            ctypes.POINTER(_Invocation),
            ctypes.POINTER(ctypes.c_size_t),
        )
        self.query.restype = ctypes.c_uint32
        self.capture = self.dll.ti_forge_cutlass_capture
        self.capture.argtypes, self.capture.restype = (
            ctypes.c_void_p,
        ), ctypes.c_uint32
        self.error = self.dll.ti_forge_cutlass_error
        self.error.argtypes, self.error.restype = (
            ctypes.c_void_p,
            ctypes.c_size_t,
        ), ctypes.c_size_t

    def workspace(self, semantics, strategy):
        validate_runtime_generation(self, "CUTLASS addon belongs to a retired runtime")
        invocation, size = _payload(semantics, strategy), ctypes.c_size_t()
        if self.query(ctypes.byref(invocation), ctypes.byref(size)):
            message = ctypes.create_string_buffer(512)
            self.error(message, len(message))
            raise TaichiRuntimeError(message.value.decode(errors="replace"))
        return size.value


class _Capture(_CudaGraphCaptureRecipe):
    kind = "cutlass_complete_matmul"

    def __init__(self, recording):
        # No recording -> capture -> recording cycle: discarded search trials
        # must release their large partial-product arrays promptly.
        for name in (
            "library",
            "semantics",
            "strategy",
            "workspace_bytes",
            "binding_names",
        ):
            setattr(self, name, getattr(recording, name))

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        r = self
        validate_runtime_generation(
            r.library, "CUTLASS addon belongs to a retired runtime"
        )
        payload = _payload(r.semantics, r.strategy)
        payload.workspace_bytes = r.workspace_bytes
        fields = ("a", "b", "output") + (("workspace",) if r.workspace_bytes else ())
        shapes = _shapes(r.semantics) + (
            ((r.workspace_bytes,),) if r.workspace_bytes else ()
        )
        builder._dispatch_cuda_addon_capture_recipe(
            program,
            ctypes.cast(r.library.capture, ctypes.c_void_p).value,
            bytes(payload),
            _Invocation.stream.offset,
            tuple(
                Arg(
                    ArgKind.NDARRAY,
                    name,
                    u8 if field == "workspace" else f32,
                    ndim=len(shape),
                )
                for name, field, shape in zip(r.binding_names, fields, shapes)
            ),
            tuple(getattr(_Invocation, field).offset for field in fields),
            tuple(math.prod(shape) for shape in shapes),
            (False, False, True) + ((True,) if r.workspace_bytes else ()),
            ctypes.cast(r.library.error, ctypes.c_void_p).value,
        )


class _Recording(BackendCommandRecording):
    graph_publish_time_binding_validation_stable = True

    def __init__(
        self, library, semantics, strategy, workspace_bytes, prefix, physical_id
    ):
        public = tuple(semantics[x] for x in ("a", "b", "output"))
        workspace = impl.ndarray(u8, workspace_bytes) if workspace_bytes else None
        name = prefix + "_workspace"
        while name in public:
            name += "_"
        super().__init__(
            backend="cuda",
            binding_names=public + ((name,) if workspace_bytes else ()),
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="stream_capture",
        )
        for key, value in dict(
            library=library,
            semantics=semantics,
            strategy=strategy,
            workspace=workspace,
            workspace_bytes=workspace_bytes,
            _graph_physical_plan_id=physical_id,
            _graph_semantic_fingerprint=_digest(semantics),
        ).items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "_cuda_capture_recipe", _Capture(self))
        identity = make_retained_plan_identity(
            "linalg.matmul.cutlass",
            "cutlass_matmul",
            "cuda",
            provider_scope={
                "provider_abi": _ABI,
                "provider_binary_identity": library.manifest.binary_sha256,
                "build_profile": library.manifest.build_report(),
            },
            problem_scope=semantics,
            execution_scope={"strategy": strategy, "workspace_bytes": workspace_bytes},
        )
        attach_retained_execution_contract(
            self,
            RetainedExecutionContract(
                identity=identity,
                workspace_ownership="provider_generation",
                concurrency_policy="runtime_ordered",
                automatic_selection_policy="forbidden",
                cost_model=HardwareExecutionCostModel(
                    (
                        fixed_cost("addon_validation", "process"),
                        fixed_cost("workspace_and_capture", "graph_instance"),
                        scale_cost("matmul_and_epilogue", "m*n*k"),
                    )
                ),
            ),
        )

    @property
    def resource_effects(self):
        access = (
            GraphAccess.READ,
            GraphAccess.READ,
            GraphAccess.READ_WRITE if self.semantics["beta"] else GraphAccess.WRITE,
        )
        if self.workspace_bytes:
            access += (GraphAccess.READ_WRITE,)
        return tuple(
            ResourceEffect(name, mode) for name, mode in zip(self.binding_names, access)
        )

    def validate_graph_bindings(self, bindings):
        values = []
        for name, shape in zip(self.binding_names[:3], _shapes(self.semantics)):
            value = bindings[name]
            if (
                not isinstance(value, Ndarray)
                or value.dtype != f32
                or tuple(value.shape) != shape
                or value.element_shape
            ):
                raise TaichiRuntimeError(
                    f"CUTLASS {name} requires a compact scalar f32 ndarray of shape {shape}"
                )
            values.append(value)
        if values[2] is values[0] or values[2] is values[1]:
            raise TaichiRuntimeError("CUTLASS output must not alias its inputs")

    def _as_graph_native_node(self):
        return _Node(self)

    def execute(self, bindings):
        raise TaichiRuntimeError("CUTLASS recipes require native root Graph capture")

    def _graph_provider_memory_report(self):
        valid = runtime_generation_matches(self.library)
        return make_memory_report(
            "cutlass_matmul",
            "cuda",
            (
                HardwareMemoryComponent(
                    "partial_products",
                    self.workspace_bytes,
                    True,
                    "provider_generation",
                    "provider",
                    resident=valid,
                ),
                HardwareMemoryComponent(
                    "driver_module_state",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=valid,
                ),
            ),
            lifecycle_state="ready" if valid else "runtime_invalid",
            ownership_scope="plan_generation",
        )


class _Node(NativeGraphNode):
    def __init__(self, recording):
        self.recording = recording

    def compile(self):
        r = self.recording
        return HardwareRecordingExecutable(
            r,
            runtime_bindings=tuple((name, "ndarray") for name in r.binding_names[:3]),
            lifetime_leases=(r,),
            debug_info={"kind": "cutlass_complete_matmul"},
            fixed_bindings=(
                {r.binding_names[3]: r.workspace} if r.workspace_bytes else {}
            ),
            publish_time_binding_validation_stable=True,
        )


class CutlassMatmulRecipeProvider(GraphRuntimeFragmentProvider):
    """Opt-in complete f32 matmul strategies from a separately built addon.

    Discovers prepared ``ti.linalg.record_matmul`` regions. Single matrices,
    either operand storage order, identity/ReLU, and live beta feedback are
    supported. It neither replaces ordinary matmul nor exposes tile parameters.
    """

    def __init__(self, manifest_path, *, workspace_limit_bytes=32 << 20):
        if (
            isinstance(workspace_limit_bytes, bool)
            or not isinstance(workspace_limit_bytes, int)
            or workspace_limit_bytes < 0
        ):
            raise ValueError("workspace_limit_bytes must be a nonnegative integer")
        if not hasattr(core.GraphBuilder, "_dispatch_cuda_addon_capture_recipe"):
            raise TaichiRuntimeError("Native addon capture support is unavailable")
        self._library = _Library(manifest_path)
        self._component = self._library.manifest.build_report()
        self._limit = workspace_limit_bytes
        self.descriptor = runtime_family_provider_descriptor(
            "cutlass_matmul",
            capabilities=(
                "semantic-f32-matmul",
                "native-addon-capture",
                "parallel-k-reduction",
            ),
            domain_version="cutlass-complete-matmul-v1",
            semantic_fingerprint="cutlass-f32-v1:"
            + _digest({"build": self._component, "workspace_limit": self._limit}),
        )

    def _sources(self, definition):
        validate_runtime_generation(
            self._library, "CUTLASS provider belongs to a retired runtime"
        )
        return tuple(_sources(definition)) if definition.backend == "cuda" else ()

    def _choices(self, source):
        s = source.semantics
        if s["batch_count"] != 1 or any(s[x] > 2147483647 for x in ("m", "n", "k")):
            return ()
        result = []
        for strategy in _STRATEGIES:
            size = self._library.workspace(s, strategy)
            if size <= self._limit:
                result.append((strategy, size))
        return tuple(result)

    def fragments(self, definition):
        result = []
        for path, region, source, _ in self._sources(definition):
            for strategy, size in self._choices(source):
                stages = (
                    ("partial_products", "reduce_epilogue")
                    if size
                    else ("matmul_epilogue",)
                )
                tasks = []
                for stage in stages:
                    tasks.append(
                        GraphFragmentTask.create(
                            f"{path}:cutlass:{stage}",
                            stage,
                            depends_on=(tasks[-1].task_id,) if tasks else (),
                            physical={
                                "semantics": source.semantics,
                                "strategy": strategy,
                                "numeric_policy": "simt-f32-no-tf32",
                                "component": self._component,
                                "workspace_bytes": size,
                                "stage": stage,
                            },
                        )
                    )
                resources = (
                    (
                        GraphFragmentResourceRequirement(
                            f"{path}:cutlass:partials",
                            "matmul_partial_products",
                            size,
                            ownership="graph_instance",
                            lifetime="graph",
                            exclusive_submission=True,
                        ),
                    )
                    if size
                    else ()
                )
                result.append(
                    _fragment(
                        definition,
                        family="cutlass_matmul",
                        source_key=path,
                        choice_id=strategy,
                        coverage=(region,),
                        tasks=tuple(tasks),
                        resources=resources,
                        exclusive_submission=True,
                        provider_descriptor=self.descriptor,
                    )
                )
        return tuple(result)

    def contribute_runtime(self, assembly, selection):
        matches = [
            x
            for x in self._sources(assembly.definition)
            if x[0] == selection.source_key
        ]
        if len(matches) != 1:
            raise ValueError("Frozen CUTLASS matmul source is unavailable")
        path, _, source, executable = matches[0]
        choices = dict(self._choices(source))
        strategy = selection.materialization_choice
        if strategy not in choices:
            raise ValueError(
                "CUTLASS strategy is unavailable for this semantic contract"
            )
        size = choices[strategy]
        identity = "cutlass-matmul:" + _digest(
            {
                "semantics": source.semantics,
                "strategy": strategy,
                "workspace_bytes": size,
                "component": self._component,
            }
        )

        def append(builder, operation):
            recording = _Recording(
                self._library,
                source.semantics,
                strategy,
                size,
                "__cutlass_" + _digest(path)[:16],
                identity,
            )
            builder.append_native(recording, admission=operation[2])

        assembly.select_operation(executable, append)

    def explain_discovery(self, definition):
        sources = self._sources(definition)
        return {
            "source": "provider_declared_not_measured",
            "semantic_source_count": len(sources),
            "semantic_api": "ti.linalg.record_matmul",
            "component_applicability": self._component,
            "choices": [
                {
                    "source_key": path,
                    "strategies": list(dict(self._choices(source))),
                    "reason": (
                        "single_matrix_only"
                        if source.semantics["batch_count"] != 1
                        else "workspace_bounded"
                    ),
                }
                for path, _, source, _ in sources
            ],
        }

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        return {
            **fragment.provider_metadata,
            "component_applicability": self._component,
            "physical_strategy": fragment.tasks[0].physical,
            "limitations": (
                "single compact f32 matrices; finite-input application tolerance",
                "split-K reassociates accumulation; no reduced-precision TF32",
                "separately compiled Toolkit addon; no automatic runtime route",
                "requested workspace is known; driver/module/peak VRAM is not measured",
            ),
        }


__all__ = ["CutlassMatmulRecipeProvider"]
