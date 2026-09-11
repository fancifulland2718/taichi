"""Pinned recordable operator generations over existing Graph ownership."""

from dataclasses import replace

from taichi_forge.graph._ir import GraphAccess, NativeCallNode, ResourceEffect
from taichi_forge.graph._native import DispatchGraphAction, NativeGraphExecutable, NativeGraphNode
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import f32


class _PinnedOperatorExecutable(NativeGraphExecutable):
    def __init__(self, snapshot, namespace):
        from taichi_forge.graph._graph import Arg, ArgKind

        self._snapshot = snapshot
        names = {}
        dispatches = []
        for kernel, symbols in snapshot.dispatches:
            renamed = []
            for symbol in symbols:
                if symbol.name not in names:
                    name = f"{namespace}_{len(names)}"
                    if symbol.tag == ArgKind.SCALAR:
                        value = Arg(ArgKind.SCALAR, name, symbol.dtype())
                    elif symbol.tag == ArgKind.NDARRAY:
                        if tuple(symbol.element_shape):
                            raise TaichiRuntimeError("Prepared operator expects scalar provider arrays")
                        value = Arg(ArgKind.NDARRAY, name, symbol.dtype(), ndim=int(symbol.field_dim))
                    else:
                        raise TaichiRuntimeError("Prepared operator actions require scalar/ndarray arguments")
                    names[symbol.name] = value
                renamed.append(names[symbol.name])
            dispatches.append((kernel, tuple(renamed)))
        self._action = DispatchGraphAction(
            dispatches,
            backends=snapshot.backends,
            conditional_body_safe=False,
            fixed_bindings={symbol.name: snapshot.bindings[old] for old, symbol in names.items()},
            update_policy="immutable",
        )

    @property
    def recordable_action(self):
        return self._action

    @property
    def resource_effects(self):
        return self._snapshot.effects

    @property
    def lifetime_leases(self):
        return (self._snapshot,)

    @property
    def graph_ir_node(self):
        # Provider Graph workspace/state lacks per-dispatch semantic effects.
        # Pinning a generation does not make it pointwise or fusion-eligible.
        return NativeCallNode(name="prepared_operator_apply", effects=self.resource_effects, opaque=True)

    @property
    def debug_info(self):
        return self._snapshot.description


class _PinnedOperatorRoot(NativeGraphExecutable):
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def recordable_root_actions(self, namespace):
        if impl.get_runtime().prog is not self._snapshot.program:
            raise TaichiRuntimeError("Prepared operator recording belongs to another runtime")
        return (_PinnedOperatorExecutable(self._snapshot, namespace),)


class _PinnedOperatorNode(NativeGraphNode):
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def compile(self):
        return _PinnedOperatorRoot(self._snapshot)


class _OperatorSnapshot:
    def __init__(self, operator, input, output, adjoint):
        from taichi_forge.graph._graph import Arg, ArgKind, GraphOwnedNdarray

        self.program = impl.get_runtime().prog
        args = (Arg(ArgKind.NDARRAY, "input", f32, ndim=1), Arg(ArgKind.NDARRAY, "output", f32, ndim=1))
        source = operator.graph_action(*args, adjoint=adjoint).compile()
        action = source.recordable_action
        if not isinstance(action, DispatchGraphAction):
            raise TaichiRuntimeError("Prepared operator requires a provider-qualified dispatch action")
        actual = {"input": input, "output": output}
        source.validate_graph_bindings(actual)
        source.validate_graph_lifetime()
        try:
            prepared = source.bind_graph_arguments(actual)
        except KeyError as exc:
            if not exc.args or exc.args[0] not in action.temporary_bindings:
                raise
            raise TaichiRuntimeError(
                "Prepared operator cannot freeze derived views of unresolved composition workspace"
            ) from exc
        bindings = dict(action.fixed_bindings)
        bindings.update(actual)
        bindings.update(prepared.replacements)
        workspace_bytes = 0
        # Reuse Graph-owned, instance-local workspace instead of allocating
        # plan-private scratch shared by independently compiled Graphs.
        requirements = {item.name: item for item in source.temporary_requirements}
        workspace = {}
        for name, requirement in requirements.items():
            if requirement.storage_kind != "f32" or requirement.alignment > 256:
                raise TaichiRuntimeError("Prepared operator requires the existing f32 Graph workspace contract")
            workspace[name] = GraphOwnedNdarray(f32, (requirement.bytes // 4,))
            workspace_bytes += requirement.bytes
        for symbol, name in action.temporary_bindings.items():
            bindings[symbol] = workspace[name]
        self.dispatches = tuple(action.dispatches)
        self.backends = action.capabilities.backends
        self.bindings = bindings
        self.owners = (source, *tuple(prepared.submission_owners))
        self.effects = tuple(
            (
                replace(effect, resource=actual[effect.resource], runtime_bound=False)
                if effect.runtime_bound and effect.resource in actual
                else effect
            )
            for effect in source.resource_effects
        ) + tuple(ResourceEffect(value, GraphAccess.READ_WRITE, runtime_bound=False) for value in workspace.values())
        self.description = {
            "kind": "prepared_operator_apply",
            "provider": operator.provider,
            "shape": tuple(operator.shape),
            "dtype": str(operator.dtype),
            "adjoint": adjoint,
            "dispatch_count": len(self.dispatches),
            "graph_workspace_bytes": workspace_bytes,
            "generation_policy": "pinned_until_reprepare",
            "dense_state_policy": "live",
            "runtime_arguments": "fixed_dense_storage",
            "recording_scope": "root_graph",
        }
        self.generations = tuple(
            tuple(owner._resource_stamp())[2:]
            for owner in prepared.submission_owners
            if hasattr(owner, "_resource_stamp")
        )
        self.coefficient_versions = tuple(
            int(owner.version) for owner in prepared.submission_owners if hasattr(owner, "version")
        )


class PreparedOperatorPlan:
    """Prepared f32 recordable operator with fixed dense input/output.

    Native kernels and provider numeric/topology snapshots are pinned without
    copying. Inputs and declared live field state may change in place. Later
    update_numeric()/coefficient publications do not alter this plan: prepare
    again to select a new generation. The dynamic graph_action() API is unchanged.

    run() uses the existing Graph binding/replay implementation. record() takes
    another root-Graph reference to the immutable action/resources; it does not
    execute a nested Graph. Each compiled Graph owns its composition workspace.
    close() retires this plan's Graph, not separately compiled recordings.
    Runtime reset and source tree retirement keep the native lifetime checks.

    Requires a provider-qualified f32 scalar-vector Graph action. This is not a
    new stored/vendor sparse route, generalized alpha/beta apply, or implicit
    field staging path. Derived subviews of unresolved composition workspace
    are not supported. Preparation performs no mathematical application.
    """

    def __init__(self, operator, input, out, *, adjoint=False):
        from taichi_forge.graph._graph import GraphBuilder

        self._program = impl.get_runtime().prog
        self._graph = self._binding = self._snapshot = None
        try:
            self._snapshot = _OperatorSnapshot(operator, input, out, adjoint)
            builder = GraphBuilder()
            builder.append_native(_PinnedOperatorNode(self._snapshot))
            self._graph = builder.compile()
            self._binding = self._graph.bind({})
            self._description = dict(self._snapshot.description)
            self._generations = self._snapshot.generations
            self._coefficient_versions = self._snapshot.coefficient_versions
        except BaseException:
            self.close()
            raise
        impl.get_runtime().register_runtime_object(self)

    def _validate_lifetime(self):
        if self._snapshot is None:
            raise TaichiRuntimeError("PreparedOperatorPlan is closed or retired")
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError("PreparedOperatorPlan belongs to another runtime")

    def run(self):
        self._validate_lifetime()
        self._graph.run(self._binding)

    def record(self):
        self._validate_lifetime()
        return _PinnedOperatorNode(self._snapshot)

    def report(self):
        return {
            **self._description,
            "provider_generations": self._generations,
            "coefficient_versions": self._coefficient_versions,
            "closed": self._snapshot is None,
            "binding": None if self._binding is None else dict(self._binding.statistics()),
        }

    def close(self):
        if self._graph is not None:
            self._graph.close()
        self._graph = self._binding = self._snapshot = self._program = None

    def _invalidate_runtime(self):
        self.close()

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["PreparedOperatorPlan"]
