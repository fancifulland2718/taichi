"""Shared adapters for explicit hardware recordings in root Graphs."""

from dataclasses import replace

from taichi_forge.graph._ir import ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import (
    BackendCommandGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
    _GraphValidatedBindings,
)
from taichi_forge.hardware._retained import validate_retained_execution_contract
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang import impl


def validate_exact_bindings(recording, bindings, operation):
    """Reject missing and unexpected bindings with one stable diagnostic."""

    required = frozenset(recording.binding_names)
    provided = frozenset(bindings)
    if provided == required:
        return
    missing = sorted(required.difference(provided))
    unexpected = sorted(provided.difference(required))
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise TaichiRuntimeError(
        f"{operation} bindings do not match the recording: "
        + "; ".join(details)
    )


def graph_bindings_are_validated(bindings):
    """Whether bindings carry the owning Graph's validation certificate."""

    return isinstance(bindings, _GraphValidatedBindings)


def runtime_generation_matches(owner):
    """Returns whether a provider object belongs to the active Program."""

    return (
        impl.get_runtime().prog is owner._runtime_prog
        and int(impl.runtime_generation()) == owner._runtime_generation
    )


def validate_runtime_generation(owner, message):
    if not runtime_generation_matches(owner):
        raise TaichiRuntimeError(message)


def static_resource_effect(resource, access, *, subresource=None):
    return ResourceEffect(
        resource,
        access,
        runtime_bound=False,
        subresource=subresource,
    )


def _resolve(value, recording):
    return value(recording) if callable(value) else value


class HardwareRecordingExecutable(NativeGraphExecutable):
    def __init__(
        self,
        recording,
        *,
        runtime_bindings,
        lifetime_leases,
        debug_info,
        fixed_bindings=None,
        publish_time_binding_validation_stable=False,
    ):
        if not isinstance(publish_time_binding_validation_stable, bool):
            raise TypeError("publish_time_binding_validation_stable must be a bool")
        if publish_time_binding_validation_stable and not callable(
            getattr(recording, "validate_graph_bindings", None)
        ):
            raise ValueError(
                "stable publish-time Graph binding validation requires a "
                "validate_graph_bindings() implementation"
            )
        self._recording = recording
        self._runtime_bindings = runtime_bindings
        self._lifetime_leases = lifetime_leases
        self._debug_info = debug_info
        self.graph_publish_time_binding_validation_stable = (
            publish_time_binding_validation_stable
        )
        validate_retained_execution_contract(
            recording, tuple(_resolve(lifetime_leases, recording))
        )
        self._action = BackendCommandGraphAction(recording, fixed_bindings=fixed_bindings)

    def run(self, runtime_args):
        return self._recording.execute(runtime_args)

    def validate_graph_bindings(self, runtime_args):
        validate = getattr(self._recording, "validate_graph_bindings", None)
        if validate is not None:
            validate(runtime_args)

    @property
    def runtime_arg_schema(self):
        bindings = _resolve(self._runtime_bindings, self._recording)
        return tuple(RuntimeBinding(name, kind) for name, kind in bindings)

    @property
    def resource_effects(self):
        return self._recording.resource_effects

    @property
    def lifetime_leases(self):
        return tuple(_resolve(self._lifetime_leases, self._recording))

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return dict(_resolve(self._debug_info, self._recording))

    @property
    def graph_ir_node(self):
        node = super().graph_ir_node
        fingerprint = getattr(self._recording, "_graph_semantic_fingerprint", "")
        return replace(node, semantic_fingerprint=fingerprint) if fingerprint else node

    @property
    def graph_physical_plan_id(self):
        return getattr(self._recording, "_graph_physical_plan_id", "")

    def _freeze_graph_recipe_source(self):
        freeze = getattr(self._recording, "_freeze_graph_recipe_source", None)
        return None if freeze is None else freeze()


class HardwareRecordingNode(NativeGraphNode):
    def __init__(
        self,
        recording,
        *,
        runtime_bindings,
        lifetime_leases,
        debug_info,
        publish_time_binding_validation_stable=False,
    ):
        self._recording = recording
        self._options = {
            "runtime_bindings": runtime_bindings,
            "lifetime_leases": lifetime_leases,
            "debug_info": debug_info,
            "publish_time_binding_validation_stable": (
                publish_time_binding_validation_stable
            ),
        }

    def compile(self):
        return HardwareRecordingExecutable(self._recording, **self._options)


def native_recording_node(
    recording,
    *,
    runtime_bindings=None,
    lifetime_leases=(),
    debug_info=None,
    publish_time_binding_validation_stable=False,
):
    """Builds the common recordable-action Graph adapter.

    ``publish_time_binding_validation_stable`` is an internal certificate for
    validation that depends only on identities/types retained by one immutable
    Graph BindingVersion. It must remain false for dynamic generation, shape,
    replacement, or submission-owner checks. Vulkan graphics and fixed-storage
    FFT recordings use native handle lookup as their separate fail-closed
    lifetime boundary.
    """

    if runtime_bindings is None:
        runtime_bindings = lambda item: tuple(
            (name, "ndarray") for name in item.binding_names
        )
    if debug_info is None:
        debug_info = {}
    return HardwareRecordingNode(
        recording,
        runtime_bindings=runtime_bindings,
        lifetime_leases=lifetime_leases,
        debug_info=debug_info,
        publish_time_binding_validation_stable=publish_time_binding_validation_stable,
    )


__all__ = [
    "HardwareRecordingExecutable",
    "HardwareRecordingNode",
    "native_recording_node",
    "runtime_generation_matches",
    "static_resource_effect",
    "graph_bindings_are_validated",
    "validate_exact_bindings",
    "validate_runtime_generation",
]
