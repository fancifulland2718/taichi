"""One-shot provider contributions for runtime Graph recipe assembly."""


def _retire_runtime_resources(graph):
    graph._invalidate_runtime()


class GraphRuntimeRecipeAssembly:
    """Mutable build plan populated only at the materialization boundary.

    Providers own the meaning of their fragments and install typed callbacks
    against frozen source objects. The runtime assembler executes these hooks
    without a family-name switch. This object never reaches Graph replay.
    """

    __slots__ = (
        "definition",
        "_dispatch_rewriters",
        "_map_source_groups",
        "_node_expanders",
        "_node_source_indices",
        "_node_rewriters",
        "_operation_rewriters",
        "_parallel_schedules",
        "_workspace_pair",
        "_binding_executor_factory",
        "_storage_plans",
    )

    def __init__(self, definition):
        self.definition = definition
        self._dispatch_rewriters = {}
        self._operation_rewriters = {}
        self._node_rewriters = {}
        self._node_expanders = {}
        self._node_source_indices = {}
        self._parallel_schedules = {}
        self._map_source_groups = []
        self._workspace_pair = False
        self._binding_executor_factory = None
        self._storage_plans = []

    @property
    def spec(self):
        return self.definition._runtime_spec

    @staticmethod
    def _callable(value, role):
        if not callable(value):
            raise TypeError(f"runtime Graph {role} must be callable")
        return value

    @staticmethod
    def _register(target, key, value, role):
        if key in target:
            raise ValueError(f"runtime Graph recipe selects {role} more than once")
        target[key] = value

    @staticmethod
    def find_source(sources, source_key, *, attribute="_recipe_source_key"):
        matches = tuple(
            source
            for source in sources
            if getattr(source, attribute, None) == source_key
        )
        if len(matches) != 1:
            raise ValueError(
                f"runtime Graph recipe source is unavailable: {source_key}"
            )
        return matches[0]

    def select_dispatch(self, source, rewriter):
        self._register(
            self._dispatch_rewriters,
            id(source),
            self._callable(rewriter, "dispatch rewriter"),
            "one physical dispatch source",
        )

    def dispatch_rewriter(self, source):
        if source is None:
            return None
        return self._dispatch_rewriters.get(id(source))

    def select_operation(self, source, rewriter):
        self._register(
            self._operation_rewriters,
            id(source),
            self._callable(rewriter, "operation rewriter"),
            "one physical operation source",
        )

    def operation_rewriter(self, source):
        return self._operation_rewriters.get(id(source))

    def operation_has_selection(self, operation):
        kind = operation[0]
        if kind == "dispatch":
            return any(
                self.dispatch_rewriter(source) is not None for source in operation[4:7]
            )
        if kind in ("bounded", "graph_reduction", "native"):
            return self.operation_rewriter(operation[1]) is not None
        return False

    def add_map_source_group(self, source_group):
        source_group = tuple(int(index) for index in source_group)
        if not source_group:
            raise ValueError("runtime Graph map source group must not be empty")
        if source_group in self._map_source_groups:
            raise ValueError("runtime Graph map source group is selected twice")
        self._map_source_groups.append(source_group)

    @property
    def map_source_groups(self):
        return tuple(self._map_source_groups)

    def rewrite_node(self, node_index, rewriter):
        self._register(
            self._node_rewriters,
            int(node_index),
            self._callable(rewriter, "node rewriter"),
            "one node rewrite",
        )

    def node_rewriter(self, node_index):
        return self._node_rewriters.get(int(node_index))

    def expand_node(self, node_index, expander, *, source_node_indices=None):
        self._register(
            self._node_expanders,
            int(node_index),
            self._callable(expander, "node expander"),
            "one node expansion",
        )
        if source_node_indices is not None:
            indices = tuple(int(index) for index in source_node_indices)
            if not indices or any(
                index < 0 or index >= len(self.spec.nodes) for index in indices
            ):
                raise ValueError("runtime Graph expansion has invalid source nodes")
            self._node_source_indices[int(node_index)] = indices

    def source_node_indices(self, node_index):
        return self._node_source_indices.get(int(node_index), (int(node_index),))

    def node_expander(self, node_index):
        return self._node_expanders.get(int(node_index))

    def select_parallel_schedule(self, node_index, groups, disjoint_pairs):
        self._register(
            self._parallel_schedules,
            int(node_index),
            (tuple(tuple(group) for group in groups), tuple(disjoint_pairs)),
            "one parallel schedule",
        )

    def parallel_schedule(self, node_index):
        return self._parallel_schedules.get(int(node_index), ((), ()))

    def enable_workspace_pair(self):
        if self._workspace_pair:
            raise ValueError("runtime Graph workspace pair is selected twice")
        self._workspace_pair = True

    @property
    def workspace_pair(self):
        return self._workspace_pair

    def select_binding_executor(self, factory):
        """Install one complete submission strategy, only at materialization."""
        if self._binding_executor_factory is not None:
            raise ValueError("runtime Graph binding executor is selected twice")
        self._binding_executor_factory = self._callable(
            factory, "binding executor factory"
        )

    @property
    def binding_executor_factory(self):
        return self._binding_executor_factory

    def select_storage(self, plan):
        """Select private allocation ownership, never a steady replay callback."""
        from taichi_forge.graph._recipes.runtime_storage import validate_storage_plans

        validate_storage_plans(self.spec, (*self._storage_plans, plan))
        self._storage_plans.append(plan)

    @property
    def storage_plans(self):
        return tuple(self._storage_plans)

    @property
    def executor_release(self):
        if self._binding_executor_factory is not None or self._storage_plans:
            return _retire_runtime_resources
        return None


__all__ = ["GraphRuntimeRecipeAssembly"]
