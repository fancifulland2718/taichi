from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge._hardware_telemetry import (
    instrument_hardware_recording,
    record_graph_recording,
)
from taichi_forge.graph._ir import (
    GraphAccess,
    NativeCallNode,
    ResourceEffect,
    RuntimeBinding,
    TemporaryRequirement,
    _graph_effect_to_dict,
)


class ProviderOwnedNdarrayBinding:
    """A non-owning Graph binding backed by a provider generation lease."""

    __slots__ = (
        "_arr",
        "_owner",
        "_runtime_prog",
        "_runtime_allocation_identity",
        "_runtime_storage_arguments",
    )

    def __init__(self, native_array, owner):
        if native_array is None or owner is None:
            raise ValueError(
                "Provider-owned ndarray bindings require storage and an owner"
            )
        self._arr = native_array
        self._owner = owner
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_allocation_identity = native_array.device_allocation().alloc_id
        self._runtime_storage_arguments = {}

    @property
    def arr(self):
        """The immutable native allocation published with this owner lease."""

        return self._arr

    def _runtime_storage_argument(self, consumer, mode):
        program = impl.get_runtime().prog
        if program is None or program is not self._runtime_prog:
            raise TaichiRuntimeError(
                "Provider-owned ndarray belongs to another Taichi runtime"
            )
        key = (consumer, mode)
        cached = self._runtime_storage_arguments.get(key)
        if cached is not None:
            return cached
        described = _ti_core._describe_ndarray_storage(self.arr, "readwrite")
        if not described.ok:
            raise TaichiRuntimeError(
                "Cannot describe provider-owned ndarray storage: " f"{described.reason}"
            )
        argument = _ti_core._make_runtime_storage_argument(
            program, described.descriptor, consumer, mode
        )
        qualification = dict(argument.qualification)
        if not qualification["bindable"] or not qualification["replayable"]:
            raise TaichiRuntimeError(
                "Provider-owned ndarray storage is not Graph eligible: "
                f"{qualification['reason']}"
            )
        if mode == "capture" and not qualification["capturable"]:
            raise TaichiRuntimeError(
                "Provider-owned ndarray storage is not Graph-capturable: "
                f"{qualification['reason']}"
            )
        self._runtime_storage_arguments[key] = argument
        return argument


@dataclass(frozen=True)
class PreparedGraphBindings:
    """One atomic provider binding result for a Graph invocation.

    ``replacements`` and ``submission_owners`` describe the same immutable
    provider generation.  Returning them together prevents a later owner
    query from observing a newer generation than the storage bindings that
    were prepared for this invocation.
    """

    replacements: object
    submission_owners: tuple = ()

    def __post_init__(self):
        if not isinstance(self.replacements, MappingProxyType) and not isinstance(
            self.replacements, dict
        ):
            raise TypeError("Prepared Graph replacements must be a mapping")
        object.__setattr__(self, "submission_owners", tuple(self.submission_owners))


@dataclass(frozen=True)
class GraphTemporaryBuffer:
    """One named byte slice in a Graph-owned runtime allocation."""

    storage: object
    offset: int
    bytes: int
    alignment: int
    slot: int


@dataclass(frozen=True)
class RecordableActionCapabilities:
    """Backend-neutral guarantees for one recordable runtime action."""

    backends: tuple = ("cpu", "cuda", "vulkan")
    conditional_body_safe: bool = False
    address_stable: bool = True
    update_policy: str = "rebind"
    synchronization_domain: str = "runtime_ordered"

    def __post_init__(self):
        backends = tuple(self.backends)
        if len(backends) != len(set(backends)):
            raise ValueError("Recordable action backends must be unique")
        if any(backend not in ("cpu", "cuda", "vulkan") for backend in backends):
            raise ValueError("Unsupported recordable action backend")
        if self.update_policy not in ("immutable", "rebind", "rebuild"):
            raise ValueError("Unsupported recordable action update policy")
        if self.synchronization_domain not in (
            "runtime_ordered",
            "explicit_stream",
        ):
            raise ValueError("Unsupported recordable action synchronization domain")
        object.__setattr__(self, "backends", backends)

    def to_dict(self):
        return {
            "backends": self.backends,
            "conditional_body_safe": self.conditional_body_safe,
            "address_stable": self.address_stable,
            "update_policy": self.update_policy,
            "synchronization_domain": self.synchronization_domain,
        }


@dataclass(frozen=True)
class NativeActionManifest:
    """Immutable effect and lowering contract for one native Graph action.

    The manifest freezes provider declarations when a Graph is compiled. It
    intentionally contains symbolic names and counts, never provider-owned
    storage objects or runtime addresses.
    """

    schema_version: int
    name: str
    recordable: bool
    opaque: bool
    synchronization: bool
    dispatch_count: int
    backends: tuple
    conditional_body_safe: bool
    address_stable: bool
    update_policy: str
    synchronization_domain: str
    runtime_bindings: tuple
    derived_runtime_bindings: tuple
    effects: tuple
    temporaries: tuple
    fixed_binding_names: tuple
    temporary_bindings: tuple
    lifetime_lease_count: int
    execution_kind: str = "opaque_host"
    recording_kind: str = "opaque"
    queue: str = "host"
    stream_binding: str = "opaque"
    barrier_policy: str = "opaque"
    workspace_ownership: str = "provider"
    replay_mode: str = "opaque"
    backend_command_count: object = None
    backend_command_count_exact: bool = False
    loose_helper_count: object = None
    loose_helper_count_exact: bool = False
    backend_command_replay: bool = False
    automatic_admissible: bool = False
    fragmentation_reason: str = "none"
    physical_plan_id: str = ""

    def to_dict(self, *, _static_resources=None):
        if _static_resources is None:
            _static_resources = {}
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "recordable": self.recordable,
            "opaque": self.opaque,
            "synchronization": self.synchronization,
            "dispatch_count": self.dispatch_count,
            "backends": self.backends,
            "conditional_body_safe": self.conditional_body_safe,
            "address_stable": self.address_stable,
            "update_policy": self.update_policy,
            "synchronization_domain": self.synchronization_domain,
            "runtime_bindings": tuple(
                binding.to_dict() for binding in self.runtime_bindings
            ),
            "derived_runtime_bindings": tuple(
                binding.to_dict()
                for binding in self.derived_runtime_bindings
            ),
            "effects": tuple(
                _graph_effect_to_dict(effect, _static_resources)
                for effect in self.effects
            ),
            "temporaries": tuple(
                temporary.to_dict() for temporary in self.temporaries
            ),
            "fixed_binding_names": self.fixed_binding_names,
            "temporary_bindings": self.temporary_bindings,
            "lifetime_lease_count": self.lifetime_lease_count,
            "execution_kind": self.execution_kind,
            "recording_kind": self.recording_kind,
            "queue": self.queue,
            "stream_binding": self.stream_binding,
            "barrier_policy": self.barrier_policy,
            "workspace_ownership": self.workspace_ownership,
            "replay_mode": self.replay_mode,
            "backend_command_count": self.backend_command_count,
            "backend_command_count_exact": self.backend_command_count_exact,
            "loose_helper_count": self.loose_helper_count,
            "loose_helper_count_exact": self.loose_helper_count_exact,
            "backend_command_replay": self.backend_command_replay,
            "automatic_admissible": self.automatic_admissible,
            "fragmentation_reason": self.fragmentation_reason,
            "physical_plan_id": self.physical_plan_id,
        }


@dataclass(frozen=True)
class BackendCommandPlan:
    """Provider command topology that is not yet an integrated CGraph action.

    The plan makes opaque native execution measurable without pretending that
    a provider-local replay cache is equivalent to one enclosing Graph.  It is
    intentionally descriptive: automatic admission remains false until the
    provider exposes ``recordable_action`` or ``recordable_sequence``.
    """

    backend: str
    helper_count: object = None
    helper_count_exact: bool = False
    command_count: object = None
    command_count_exact: bool = False
    provider_replay: bool = False
    no_host_readback: bool = True
    python_replay_loop: bool = False
    fragmentation_reason: str = "provider_command_not_graph_integrated"

    def __post_init__(self):
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("Unsupported backend command plan backend")
        if self.command_count is not None and int(self.command_count) < 0:
            raise ValueError("Backend command count must be nonnegative")
        if self.helper_count is not None and int(self.helper_count) < 0:
            raise ValueError("Backend helper count must be nonnegative")
        if self.command_count_exact and self.command_count is None:
            raise ValueError("Exact backend command plans require a count")
        if self.helper_count_exact and self.helper_count is None:
            raise ValueError("Exact backend helper plans require a count")
        if not self.fragmentation_reason:
            raise ValueError("Backend command fragmentation reason is required")


@dataclass(frozen=True)
class BackendCommandRecording:
    """Executable backend-command contract owned by one native action.

    Subclasses implement :meth:`execute` by entering a native runtime API that
    records the complete command sequence.  The Graph runtime invokes it once
    per action; a Python loop over individual driver or RHI commands is not an
    admissible implementation.

    A recording may implement ``prepare_graph_execute(bindings)`` to return a
    zero-argument callable for one exact root-Graph binding frame. Preparation
    must not submit work, synchronize, or mutate application data. The callable
    owns its native binding wrappers and must retain the provider's existing
    generation/lifetime enforcement. Qualified BindingVersions reuse it;
    changing a binding creates a new packet. This is not device command-buffer
    replay and does not change ``replay_mode``.
    """

    backend: str
    binding_names: tuple
    command_count: int
    queue: str = "compute"
    stream_binding: str = "runtime_ordered"
    barrier_policy: str = "declared_effects"
    workspace_ownership: str = "none"
    replay_mode: str = "rerecord"
    no_host_readback: bool = True

    def __post_init__(self):
        binding_names = tuple(self.binding_names)
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("Unsupported backend command recording backend")
        if any(not isinstance(name, str) or not name for name in binding_names):
            raise ValueError("Backend command binding names must be nonempty")
        if len(binding_names) != len(set(binding_names)):
            raise ValueError("Backend command binding names must be unique")
        if (
            isinstance(self.command_count, bool)
            or not isinstance(self.command_count, int)
            or self.command_count <= 0
        ):
            raise ValueError("Backend command recording requires a positive count")
        if self.queue not in ("compute", "graphics", "transfer"):
            raise ValueError("Unsupported backend command queue")
        if self.stream_binding not in ("runtime_ordered", "explicit_stream"):
            raise ValueError("Unsupported backend command stream binding")
        if self.barrier_policy not in (
            "declared_effects",
            "internal",
            "explicit",
        ):
            raise ValueError("Unsupported backend command barrier policy")
        if self.workspace_ownership not in (
            "none",
            "graph_temporary",
            "provider_generation",
        ):
            raise ValueError("Unsupported backend command workspace ownership")
        if self.replay_mode not in ("rerecord", "native_replay", "stream_capture"):
            raise ValueError("Unsupported backend command replay mode")
        object.__setattr__(self, "binding_names", binding_names)

    def execute(self, bindings):
        raise NotImplementedError

    def to_dict(self):
        return {
            "backend": self.backend,
            "binding_names": self.binding_names,
            "command_count": self.command_count,
            "queue": self.queue,
            "stream_binding": self.stream_binding,
            "barrier_policy": self.barrier_policy,
            "workspace_ownership": self.workspace_ownership,
            "replay_mode": self.replay_mode,
            "no_host_readback": self.no_host_readback,
        }


class _GraphValidatedBindings(Mapping):
    """Internal zero-copy marker for a Graph-validated binding frame."""

    __slots__ = ("_bindings",)

    def __init__(self, bindings):
        self._bindings = bindings

    def __getitem__(self, key):
        return self._bindings[key]

    def __iter__(self):
        return iter(self._bindings)

    def __len__(self):
        return len(self._bindings)


class _CudaGraphCaptureRecipe:
    """Internal compile-time lowering hook for one CUDA capture command.

    The hook may only append a typed C++ recipe while a Graph is compiled. It
    is never invoked from capture or replay execution.
    """

    kind = ""
    exact_binding_only = True

    def append_to_graph(self, builder, program):
        raise NotImplementedError


@dataclass(frozen=True)
class VulkanBufferCommand:
    """One immutable symbolic command in a Vulkan RHI buffer recording."""

    kind: str
    destination: str = ""
    source: str = ""
    destination_offset: int = 0
    source_offset: int = 0
    bytes: int = 0
    value: int = 0

    def __post_init__(self):
        for name in ("destination", "source"):
            binding = getattr(self, name)
            if not isinstance(binding, str):
                raise ValueError(
                    f"Vulkan buffer command {name} must be a string"
                )
        if self.kind not in (
            "fill_u32",
            "copy",
            "buffer_barrier",
            "memory_barrier",
        ):
            raise ValueError("Unsupported Vulkan buffer command kind")
        for name in ("destination_offset", "source_offset", "bytes", "value"):
            field = getattr(self, name)
            if isinstance(field, bool) or not isinstance(field, int) or field < 0:
                raise ValueError(
                    f"Vulkan buffer command {name} must be a nonnegative integer"
                )
        if self.value > 0xFFFFFFFF:
            raise ValueError("Vulkan buffer fill value must fit uint32")
        if self.kind in ("fill_u32", "copy"):
            if not self.destination or self.bytes <= 0:
                raise ValueError(
                    "Vulkan buffer fill and copy require a destination and bytes"
                )
            if self.destination_offset % 4 or self.bytes % 4:
                raise ValueError(
                    "Vulkan buffer command destination range must be four-byte aligned"
                )
        if self.kind == "copy":
            if not self.source or self.source_offset % 4:
                raise ValueError(
                    "Vulkan buffer copy requires a four-byte-aligned source"
                )
        elif self.source or self.source_offset:
            raise ValueError(
                "Only Vulkan buffer copy commands may declare a source"
            )
        if self.kind == "buffer_barrier":
            if not self.destination:
                raise ValueError("Vulkan buffer barrier requires a destination")
            if self.destination_offset or self.bytes or self.value:
                raise ValueError(
                    "Vulkan buffer barriers cover the complete symbolic buffer"
                )
        if self.kind == "memory_barrier" and (
            self.destination
            or self.source
            or self.destination_offset
            or self.source_offset
            or self.bytes
            or self.value
        ):
            raise ValueError("Vulkan memory barriers do not take buffer operands")

    @classmethod
    def fill_u32(cls, destination, bytes, value, *, offset=0):
        return cls(
            "fill_u32",
            destination=destination,
            destination_offset=offset,
            bytes=bytes,
            value=value,
        )

    @classmethod
    def copy(
        cls,
        destination,
        source,
        bytes,
        *,
        destination_offset=0,
        source_offset=0,
    ):
        return cls(
            "copy",
            destination=destination,
            source=source,
            destination_offset=destination_offset,
            source_offset=source_offset,
            bytes=bytes,
        )

    @classmethod
    def buffer_barrier(cls, resource):
        return cls("buffer_barrier", destination=resource)

    @classmethod
    def memory_barrier(cls):
        return cls("memory_barrier")

    @property
    def binding_names(self):
        return tuple(
            name for name in (self.destination, self.source) if name
        )


@instrument_hardware_recording("runtime.buffer_commands.vulkan")
class VulkanBufferCommandRecording(BackendCommandRecording):
    """A whole Vulkan buffer command sequence recorded by one C++ call."""

    def __init__(self, commands):
        commands = tuple(commands)
        if not commands or not all(
            isinstance(command, VulkanBufferCommand) for command in commands
        ):
            raise TypeError(
                "Vulkan buffer recordings require VulkanBufferCommand values"
            )
        binding_names = tuple(
            dict.fromkeys(
                name
                for command in commands
                for name in command.binding_names
            )
        )
        super().__init__(
            backend="vulkan",
            binding_names=binding_names,
            command_count=len(commands),
            queue="compute",
            stream_binding="runtime_ordered",
            barrier_policy="explicit",
            workspace_ownership="none",
            replay_mode="rerecord",
            no_host_readback=True,
        )
        object.__setattr__(self, "commands", commands)

    @staticmethod
    def _storage_description(value, name):
        from taichi_forge.lang._storage_view import describe_storage

        if isinstance(value, GraphTemporaryBuffer):
            value = value.storage
        description = describe_storage(value)
        if not description.supported:
            raise TaichiRuntimeError(
                f"Vulkan buffer binding {name!r} cannot describe dense storage: "
                f"{description.failure_reason}"
            )
        return description

    def execute(self, bindings):
        from taichi_forge._hardware_telemetry import hardware_failure_phase

        packet = (
            bindings
            if isinstance(bindings, _PreparedBufferCommands)
            else self._prepare_packet(bindings)
        )
        with hardware_failure_phase("provider_execution_failure"):
            packet.program._execute_vulkan_buffer_commands(packet.command)

    def prepare_graph_execute(self, bindings):
        return partial(self.execute, self._prepare_packet(bindings))

    def _prepare_packet(self, bindings):
        required = frozenset(self.binding_names)
        provided = frozenset(bindings)
        if provided != required:
            missing = sorted(required.difference(provided))
            unexpected = sorted(provided.difference(required))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "Vulkan buffer bindings do not match the recording: "
                + "; ".join(details)
            )
        descriptions = {
            name: self._storage_description(bindings[name], name)
            for name in self.binding_names
        }
        native_commands = []
        for command in self.commands:
            destination = (
                None
                if not command.destination
                else descriptions[command.destination].descriptor
            )
            source = (
                None if not command.source else descriptions[command.source].descriptor
            )
            native_commands.append(
                (
                    command.kind,
                    destination,
                    source,
                    command.destination_offset,
                    command.source_offset,
                    command.bytes,
                    command.value,
                )
            )
        program = impl.get_runtime().prog
        command = program._prepare_vulkan_buffer_commands(native_commands)
        owners = tuple(bindings[name] for name in self.binding_names)
        return _PreparedBufferCommands(
            program, command, owners, tuple(descriptions.values())
        )

    @property
    def resource_effects(self):
        access_by_name = {}

        def merge(name, access):
            if not name:
                return
            previous = access_by_name.get(name)
            access_by_name[name] = (
                access
                if previous is None or previous == access
                else GraphAccess.READ_WRITE
            )

        for command in self.commands:
            if command.kind == "fill_u32":
                merge(command.destination, GraphAccess.WRITE)
            elif command.kind == "copy":
                merge(command.source, GraphAccess.READ)
                merge(command.destination, GraphAccess.WRITE)
            elif command.kind == "buffer_barrier":
                merge(command.destination, GraphAccess.READ_WRITE)
        return tuple(
            ResourceEffect(name, access_by_name[name])
            for name in self.binding_names
        )

    def _as_graph_native_node(self):
        return _VulkanBufferCommandNode(self)


@dataclass(frozen=True)
class _PreparedBufferCommands:
    program: object
    command: object
    owners: tuple
    descriptions: tuple


@dataclass(frozen=True)
class BoundedPublicationTarget:
    """Graph-owned physical target for an optional producer specialization.

    The extent name and bounds describe semantic data already owned by the
    provider.  ``packet_storage`` is backend launch scratch owned by the
    enclosing Graph instance; accepting this target never transfers that
    ownership to the provider.
    """

    backend: str
    extent_name: str
    capacity: int
    block_dim: int
    packet_binding: object
    packet_storage: object
    packet_layout: str = "dispatch_indirect_u32x4"

    def __post_init__(self):
        if self.backend not in ("cpu", "cuda", "vulkan"):
            raise ValueError("Unsupported bounded publication backend")
        if not isinstance(self.extent_name, str) or not self.extent_name:
            raise ValueError("Bounded publication extent name must be nonempty")
        if not 1 <= int(self.capacity) <= 0x7FFFFFFF:
            raise ValueError("Bounded publication capacity is out of range")
        if not 1 <= int(self.block_dim) <= 1024:
            raise ValueError("Bounded publication block dimension is out of range")
        if self.packet_binding is None or self.packet_storage is None:
            raise ValueError("Bounded publication requires Graph-owned packet state")


class RecordableGraphAction:
    """Optional provider-neutral lowering contract for a native node.

    An action describes dispatches whose semantics are identical to
    ``NativeGraphExecutable.run()``. The Graph runtime may concatenate them
    with adjacent CGraph dispatches or place them in a structured region only
    when the advertised capabilities permit it. Provider-owned fixed bindings
    are injected by the runtime and never become public Graph arguments.
    """

    @property
    def capabilities(self):
        return RecordableActionCapabilities(backends=())

    def supports_backend(self, backend):
        return backend in self.capabilities.backends

    def supports_region(self, region_kind):
        if region_kind == "sequential":
            return True
        if region_kind in ("while_body", "if_branch", "switch_branch"):
            return self.capabilities.conditional_body_safe
        return False

    @property
    def dispatches(self):
        return ()

    @property
    def backend_command_recording(self):
        return None

    @property
    def fixed_bindings(self):
        return MappingProxyType({})

    @property
    def temporary_bindings(self):
        """Map private symbolic argument names to temporary requirement names."""
        return MappingProxyType({})

    @property
    def allows_unused_public_bindings(self):
        """Whether the physical recipe may consume a subset of public inputs.

        Fixed and temporary bindings must always be consumed. This opt-in is
        intended for a narrow transition action whose stable native interface
        is broader than the selected operation, not for silently incomplete
        provider lowering.
        """
        return False

    def bind_graph_temporaries(self, temporaries):
        """Resolve one arena slot into private fixed runtime bindings.

        Providers with nonempty ``temporary_bindings`` must override this and
        return a mapping whose keys exactly match the declared private symbols.
        Returning ``None`` fails closed before backend submission.
        """
        return MappingProxyType({}) if not temporaries else None


class DispatchGraphAction(RecordableGraphAction):
    """Recordable action backed by precompiled Taichi dispatches."""

    def __init__(
        self,
        dispatches,
        *,
        backends=("cpu", "cuda", "vulkan"),
        conditional_body_safe=True,
        fixed_bindings=None,
        allow_unused_public_bindings=False,
        update_policy="rebind",
        synchronization_domain="runtime_ordered",
    ):
        self._dispatches = tuple((kernel, tuple(args)) for kernel, args in dispatches)
        if not self._dispatches:
            raise ValueError("Recordable Graph action requires a dispatch")
        self._capabilities = RecordableActionCapabilities(
            backends=tuple(backends),
            conditional_body_safe=bool(conditional_body_safe),
            address_stable=True,
            update_policy=update_policy,
            synchronization_domain=synchronization_domain,
        )
        bindings = {} if fixed_bindings is None else dict(fixed_bindings)
        if any(not isinstance(name, str) or not name for name in bindings):
            raise ValueError("Recordable action fixed binding names must be nonempty")
        self._fixed_bindings = MappingProxyType(bindings)
        self._allows_unused_public_bindings = bool(
            allow_unused_public_bindings
        )

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def dispatches(self):
        return self._dispatches

    @property
    def fixed_bindings(self):
        return self._fixed_bindings

    @property
    def allows_unused_public_bindings(self):
        return self._allows_unused_public_bindings


class BackendCommandGraphAction(RecordableGraphAction):
    """Recordable action backed by one native command recording entrypoint."""

    def __init__(
        self,
        recording,
        *,
        conditional_body_safe=False,
        fixed_bindings=None,
        temporary_bindings=None,
        address_stable=True,
        update_policy="rebind",
    ):
        if not isinstance(recording, BackendCommandRecording):
            raise TypeError(
                "Backend command actions require a BackendCommandRecording"
            )
        if not recording.no_host_readback:
            raise ValueError("Recordable backend commands cannot read back to host")
        if recording.stream_binding != "runtime_ordered":
            raise ValueError(
                "Graph backend commands must use the runtime-ordered stream"
            )
        bindings = {} if fixed_bindings is None else dict(fixed_bindings)
        temporary = (
            {} if temporary_bindings is None else dict(temporary_bindings)
        )
        if any(not isinstance(name, str) or not name for name in bindings):
            raise ValueError("Backend command fixed binding names must be nonempty")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(requirement, str)
            or not requirement
            for name, requirement in temporary.items()
        ):
            raise ValueError(
                "Backend command temporary bindings must use nonempty names"
            )
        if recording.workspace_ownership == "graph_temporary" and not temporary:
            raise ValueError(
                "Graph-owned backend command workspace requires a temporary binding"
            )
        if recording.workspace_ownership != "graph_temporary" and temporary:
            raise ValueError(
                "Backend command temporary bindings require graph_temporary ownership"
            )
        private_names = set(bindings) | set(temporary)
        if not private_names <= set(recording.binding_names):
            raise ValueError(
                "Backend command private bindings must be declared by the recording"
            )
        if set(bindings) & set(temporary):
            raise ValueError(
                "Backend command fixed and temporary bindings must be disjoint"
            )
        self._recording = recording
        record_graph_recording(recording)
        self._capabilities = RecordableActionCapabilities(
            backends=(recording.backend,),
            conditional_body_safe=bool(conditional_body_safe),
            address_stable=bool(address_stable),
            update_policy=update_policy,
            synchronization_domain=recording.stream_binding,
        )
        self._fixed_bindings = MappingProxyType(bindings)
        self._temporary_bindings = MappingProxyType(temporary)

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def backend_command_recording(self):
        return self._recording

    @property
    def fixed_bindings(self):
        return self._fixed_bindings

    @property
    def temporary_bindings(self):
        return self._temporary_bindings

    def bind_graph_temporaries(self, temporaries):
        return MappingProxyType(
            {
                symbol: temporaries[requirement]
                for symbol, requirement in self._temporary_bindings.items()
            }
        )

    def execute(self, bindings):
        required = frozenset(self._recording.binding_names)
        provided = frozenset(bindings)
        if provided != required:
            missing = sorted(required.difference(provided))
            unexpected = sorted(provided.difference(required))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise TaichiRuntimeError(
                "Backend command bindings do not match the recording: "
                + "; ".join(details)
            )
        backend = _ti_core.arch_name(impl.current_cfg().arch)
        if backend in ("x64", "arm64"):
            backend = "cpu"
        if backend != self._recording.backend:
            raise TaichiRuntimeError(
                "Backend command recording is compiled for "
                f"{self._recording.backend}, not the active {backend} backend"
            )
        return self._recording.execute(MappingProxyType(dict(bindings)))

    def execute_graph_validated(self, bindings):
        """Execute a frame already certified by the owning compiled Graph."""

        return self._recording.execute(_GraphValidatedBindings(bindings))


class NativeGraphExecutable:
    """Compiled native graph node with an optional recordable action."""

    def prewarm(self):
        return self

    def run(self, runtime_args=None):
        raise NotImplementedError

    def run_with_graph_temporaries(self, temporaries, runtime_args=None):
        """Run once with named slices owned by the enclosing Graph."""
        if runtime_args is None:
            return self.run()
        return self.run(runtime_args)

    @property
    def debug_info(self):
        return {}

    @property
    def runtime_arg_schema(self):
        return ()

    @property
    def derived_runtime_arg_schema(self):
        """Private bindings derived from public arguments at submission.

        Derived bindings participate in recording but are not supplied by the
        Graph caller. Implementations must bind every declared name from the
        same public invocation in :meth:`bind_graph_arguments` without
        allocating or staging hidden storage.
        """
        return ()

    @property
    def resource_effects(self):
        return ()

    @property
    def temporary_requirements(self):
        return ()

    @property
    def lifetime_leases(self):
        return ()

    @property
    def recordable_action(self):
        return None

    @property
    def recordable_sequence(self):
        """Optional structured sequence equivalent to :meth:`run`.

        Flat providers should continue to expose ``recordable_action``. A
        provider whose semantics include structured control may instead
        return a Graph ``Sequential`` definition. The Graph frontend validates
        its public bindings and inlines it at compile time; implementations
        must never submit or run a second Graph from this property.
        """
        return None

    def recordable_root_actions(self, namespace):
        """Optional cold expansion into kernel/native actions at Graph root.

        Unlike a structured ``Sequential``, these actions may create recording
        boundaries. The frontend supplies a graph-local deterministic namespace
        for private bindings. Return compiled native executables, never execute
        or submit a nested Graph. Structured control does not call this hook.
        """
        return None

    def recordable_bounded_publication(self, target):
        """Optionally publish a semantic extent into Graph-owned launch state.

        Implementations must return an action equivalent to
        :attr:`recordable_action` for public state while additionally writing
        the requested physical packet.  Returning ``None`` keeps the normal
        producer action and lets the Graph insert a separate preparation
        dispatch.
        """
        return None

    @property
    def backend_command_plan(self):
        """Optional measured topology for a non-integrated native provider."""
        return None

    @property
    def graph_ir_node(self):
        info = self.debug_info
        name = (
            info.get("kind", type(self).__name__)
            if isinstance(info, dict)
            else type(self).__name__
        )
        return NativeCallNode(
            name=name,
            effects=tuple(self.resource_effects),
            bindings=tuple(self.runtime_arg_schema),
            temporaries=tuple(self.temporary_requirements),
            opaque=self.recordable_action is None,
        )


_UNSET_RECORDABLE_ACTION = object()


def native_action_manifest(
    executable, recordable_action=_UNSET_RECORDABLE_ACTION
):
    """Freeze and validate one executable's symbolic native-action contract."""
    if not isinstance(executable, NativeGraphExecutable):
        raise TaichiRuntimeError(
            "Native action manifests require a NativeGraphExecutable"
        )
    action = (
        executable.recordable_action
        if recordable_action is _UNSET_RECORDABLE_ACTION
        else recordable_action
    )
    command_plan = executable.backend_command_plan
    if command_plan is not None and not isinstance(
        command_plan, BackendCommandPlan
    ):
        raise TaichiRuntimeError(
            "Native Graph backend_command_plan must be a BackendCommandPlan"
        )
    if action is not None and not isinstance(action, RecordableGraphAction):
        raise TaichiRuntimeError(
            "Native Graph recordable_action must implement RecordableGraphAction"
        )

    runtime_bindings = tuple(executable.runtime_arg_schema)
    derived_runtime_bindings = tuple(
        executable.derived_runtime_arg_schema
    )
    effects = tuple(executable.resource_effects)
    temporaries = tuple(executable.temporary_requirements)
    if not all(isinstance(binding, RuntimeBinding) for binding in runtime_bindings):
        raise TaichiRuntimeError(
            "Native action runtime bindings must contain RuntimeBinding values"
        )
    if not all(
        isinstance(binding, RuntimeBinding)
        for binding in derived_runtime_bindings
    ):
        raise TaichiRuntimeError(
            "Native action derived runtime bindings must contain "
            "RuntimeBinding values"
        )
    if not all(isinstance(effect, ResourceEffect) for effect in effects):
        raise TaichiRuntimeError(
            "Native action effects must contain ResourceEffect values"
        )
    if not all(
        isinstance(temporary, TemporaryRequirement) for temporary in temporaries
    ):
        raise TaichiRuntimeError(
            "Native action temporaries must contain TemporaryRequirement values"
        )

    binding_names = tuple(binding.name for binding in runtime_bindings)
    derived_binding_names = tuple(
        binding.name for binding in derived_runtime_bindings
    )
    temporary_names = tuple(temporary.name for temporary in temporaries)
    if len(binding_names) != len(set(binding_names)):
        raise TaichiRuntimeError("Native action runtime binding names must be unique")
    if len(derived_binding_names) != len(set(derived_binding_names)):
        raise TaichiRuntimeError(
            "Native action derived runtime binding names must be unique"
        )
    if set(binding_names) & set(derived_binding_names):
        raise TaichiRuntimeError(
            "Native action public and derived runtime bindings must be "
            "disjoint"
        )
    if len(temporary_names) != len(set(temporary_names)):
        raise TaichiRuntimeError("Native action temporary names must be unique")

    ir_node = executable.graph_ir_node
    name = getattr(ir_node, "name", type(executable).__name__)
    if not isinstance(name, str) or not name:
        raise TaichiRuntimeError("Native action manifest name must be non-empty")
    ir_effects = tuple(getattr(ir_node, "effects", effects))
    if not all(isinstance(effect, ResourceEffect) for effect in ir_effects):
        raise TaichiRuntimeError(
            "Native action IR effects must contain ResourceEffect values"
        )

    if action is None:
        capabilities = RecordableActionCapabilities(backends=())
        fixed_binding_names = ()
        temporary_bindings = ()
        dispatch_count = 0
        update_policy = "opaque"
        synchronization_domain = "opaque"
        recording = None
    else:
        capabilities = action.capabilities
        if not isinstance(capabilities, RecordableActionCapabilities):
            raise TaichiRuntimeError(
                "Recordable action capabilities must be a "
                "RecordableActionCapabilities value"
            )
        fixed_binding_names = tuple(sorted(action.fixed_bindings))
        temporary_bindings = tuple(sorted(action.temporary_bindings.items()))
        dispatch_count = len(tuple(action.dispatches))
        update_policy = capabilities.update_policy
        synchronization_domain = capabilities.synchronization_domain
        recording = action.backend_command_recording
        if recording is not None and not isinstance(
            recording, BackendCommandRecording
        ):
            raise TaichiRuntimeError(
                "Backend command actions must expose a BackendCommandRecording"
            )
        if recording is not None and tuple(action.dispatches):
            raise TaichiRuntimeError(
                "A native action cannot mix backend commands and dispatches"
            )
        if recording is None and dispatch_count == 0:
            raise TaichiRuntimeError(
                "Recordable native actions require dispatches or a backend command"
            )
        if (
            recording is not None
            and recording.workspace_ownership == "provider_generation"
            and not tuple(executable.lifetime_leases)
        ):
            raise TaichiRuntimeError(
                "Provider-owned backend command workspace requires a lifetime lease"
            )

    if recording is not None:
        execution_kind = "backend_command"
        recording_kind = "native_command"
        queue = recording.queue
        stream_binding = recording.stream_binding
        barrier_policy = recording.barrier_policy
        workspace_ownership = recording.workspace_ownership
        replay_mode = recording.replay_mode
    elif action is not None:
        execution_kind = "kernel_dispatch"
        recording_kind = "cgraph_dispatch"
        queue = "compute"
        stream_binding = synchronization_domain
        barrier_policy = "declared_effects"
        workspace_ownership = "graph_temporary" if temporaries else "none"
        replay_mode = "backend_graph"
    else:
        execution_kind = "opaque_host"
        recording_kind = "opaque"
        queue = "host"
        stream_binding = "opaque"
        barrier_policy = "opaque"
        workspace_ownership = "provider"
        replay_mode = "opaque"

    return NativeActionManifest(
        schema_version=4,
        name=name,
        recordable=action is not None,
        opaque=bool(getattr(ir_node, "opaque", action is None)),
        synchronization=bool(getattr(ir_node, "synchronization", False)),
        dispatch_count=dispatch_count,
        backends=tuple(capabilities.backends),
        conditional_body_safe=bool(capabilities.conditional_body_safe),
        address_stable=bool(capabilities.address_stable) if action else False,
        update_policy=update_policy,
        synchronization_domain=synchronization_domain,
        runtime_bindings=runtime_bindings,
        derived_runtime_bindings=derived_runtime_bindings,
        effects=ir_effects,
        temporaries=temporaries,
        fixed_binding_names=fixed_binding_names,
        temporary_bindings=temporary_bindings,
        lifetime_lease_count=len(tuple(executable.lifetime_leases)),
        execution_kind=execution_kind,
        recording_kind=recording_kind,
        queue=queue,
        stream_binding=stream_binding,
        barrier_policy=barrier_policy,
        workspace_ownership=workspace_ownership,
        replay_mode=replay_mode,
        backend_command_count=(
            recording.command_count
            if recording is not None
            else (None if command_plan is None else command_plan.command_count)
        ),
        backend_command_count_exact=bool(
            recording is not None
            or (command_plan is not None and command_plan.command_count_exact)
        ),
        loose_helper_count=(
            None if command_plan is None else command_plan.helper_count
        ),
        loose_helper_count_exact=bool(
            command_plan is not None and command_plan.helper_count_exact
        ),
        backend_command_replay=bool(
            (
                recording is not None
                and recording.replay_mode in ("native_replay", "stream_capture")
            )
            or (command_plan is not None and command_plan.provider_replay)
        ),
        automatic_admissible=action is not None,
        fragmentation_reason=(
            "none"
            if action is not None
            else (
                command_plan.fragmentation_reason
                if command_plan is not None
                else "opaque_native_action"
            )
        ),
        physical_plan_id=str(
            getattr(executable, "graph_physical_plan_id", "")
        ),
    )


class NativeGraphNode:
    """Definition-time native graph node."""

    dsl_defined = True

    def compile(self):
        raise NotImplementedError


class _VulkanBufferCommandExecutable(NativeGraphExecutable):
    def __init__(self, recording):
        self._recording = recording
        self._action = BackendCommandGraphAction(recording)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(name, "ndarray")
            for name in self._recording.binding_names
        )

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {"kind": "vulkan_buffer_commands"}


class _VulkanBufferCommandNode(NativeGraphNode):
    def __init__(self, recording):
        self._recording = recording

    def compile(self):
        return _VulkanBufferCommandExecutable(self._recording)


def compile_native_graph_node(node):
    if isinstance(node, NativeGraphNode):
        return node.compile()
    to_graph_node = getattr(node, "_as_graph_native_node", None)
    if to_graph_node is not None:
        graph_node = to_graph_node()
        if isinstance(graph_node, NativeGraphNode):
            return graph_node.compile()
    raise TaichiRuntimeError(
        "Only DSL-defined native graph nodes are supported by GraphBuilder"
    )
