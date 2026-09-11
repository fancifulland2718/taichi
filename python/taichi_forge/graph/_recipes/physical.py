"""Backend-neutral manifests for materialized complete Graph recipes."""

from dataclasses import asdict, dataclass, field, replace

from taichi_forge.graph._ir import ResourceEffect
from taichi_forge.graph._recipes.definition import _canonical_json, _digest

_PHYSICAL_MANIFEST_SCHEMA = "taichi_forge.compiled_graph_physical_manifest.v2"
_MAX_SIGNED_BYTES = (1 << 63) - 1


class GraphPhysicalManifestError(ValueError):
    """A backend report cannot prove one complete materialized Graph."""


def _required_text(value, role):
    if not isinstance(value, str) or not value:
        raise GraphPhysicalManifestError(f"{role} must be a non-empty string")
    return value


def _nonnegative_int(value, role, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraphPhysicalManifestError(f"{role} must be an integer")
    if value < 0:
        raise GraphPhysicalManifestError(f"{role} must be non-negative")
    if value > _MAX_SIGNED_BYTES:
        raise GraphPhysicalManifestError(f"{role} exceeds signed 64-bit range")
    return value


def _positive_int(value, role):
    value = _nonnegative_int(value, role)
    if value == 0:
        raise GraphPhysicalManifestError(f"{role} must be positive")
    return value


def _normalized_strings(values, role, *, sort=False):
    values = tuple(values)
    for value in values:
        _required_text(value, role)
    if len(values) != len(set(values)):
        raise GraphPhysicalManifestError(f"{role} values must be unique")
    return tuple(sorted(values)) if sort else values


def _properties_json(value):
    return _canonical_json({} if value is None else value)


def _properties(value):
    import json

    return json.loads(value)


@dataclass(frozen=True)
class GraphPhysicalBindingManifest:
    """One concrete binding slot in the materialized Graph ABI."""

    binding_index: int
    name: str
    kinds: tuple[str, ...]
    required: bool
    scope: str = "public"

    def __post_init__(self):
        _nonnegative_int(self.binding_index, "Graph physical binding index")
        _required_text(self.name, "Graph physical binding name")
        object.__setattr__(
            self,
            "kinds",
            _normalized_strings(
                self.kinds,
                "Graph physical binding kind",
                sort=True,
            ),
        )
        if self.scope not in ("public", "fragment_internal"):
            raise GraphPhysicalManifestError("Graph physical binding scope is invalid")

    def to_dict(self):
        return {
            "binding_index": self.binding_index,
            "name": self.name,
            "kinds": self.kinds,
            "required": self.required,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class GraphPhysicalKernelManifest:
    """One actually compiled kernel, pipeline, or device artifact."""

    kernel_index: int
    kind: str
    artifact_identity: str
    pipeline_identity: str = ""
    source_identities: tuple[str, ...] = ()
    requested_grid_size: int | None = None
    selected_grid_size: int | None = None
    actual_grid_size: int | None = None
    requested_block_size: int | None = None
    selected_block_size: int | None = None
    actual_block_size: int | None = None
    static_shared_bytes: int = 0
    dynamic_shared_bytes: int = 0
    thread_local_bytes: int = 0
    abi_binding_names: tuple[str, ...] = ()
    _properties_json: str = field(default="{}", repr=False)

    @classmethod
    def create(
        cls,
        kernel_index,
        kind,
        artifact_identity,
        *,
        pipeline_identity="",
        source_identities=(),
        requested_grid_size=None,
        selected_grid_size=None,
        actual_grid_size=None,
        requested_block_size=None,
        selected_block_size=None,
        actual_block_size=None,
        static_shared_bytes=0,
        dynamic_shared_bytes=0,
        thread_local_bytes=0,
        abi_binding_names=(),
        properties=None,
    ):
        return cls(
            kernel_index=_nonnegative_int(
                kernel_index,
                "Graph physical kernel index",
            ),
            kind=_required_text(kind, "Graph physical kernel kind"),
            artifact_identity=_required_text(
                artifact_identity,
                "Graph physical kernel artifact identity",
            ),
            pipeline_identity=str(pipeline_identity or ""),
            source_identities=_normalized_strings(
                source_identities,
                "Graph physical kernel source identity",
            ),
            requested_grid_size=_nonnegative_int(
                requested_grid_size,
                "Graph requested grid size",
                optional=True,
            ),
            selected_grid_size=_nonnegative_int(
                selected_grid_size,
                "Graph selected grid size",
                optional=True,
            ),
            actual_grid_size=_nonnegative_int(
                actual_grid_size,
                "Graph actual grid size",
                optional=True,
            ),
            requested_block_size=_nonnegative_int(
                requested_block_size,
                "Graph requested block size",
                optional=True,
            ),
            selected_block_size=_nonnegative_int(
                selected_block_size,
                "Graph selected block size",
                optional=True,
            ),
            actual_block_size=_nonnegative_int(
                actual_block_size,
                "Graph actual block size",
                optional=True,
            ),
            static_shared_bytes=_nonnegative_int(
                static_shared_bytes,
                "Graph static shared bytes",
            ),
            dynamic_shared_bytes=_nonnegative_int(
                dynamic_shared_bytes,
                "Graph dynamic shared bytes",
            ),
            thread_local_bytes=_nonnegative_int(
                thread_local_bytes,
                "Graph thread-local bytes",
            ),
            abi_binding_names=_normalized_strings(
                abi_binding_names,
                "Graph physical kernel ABI binding",
            ),
            _properties_json=_properties_json(properties),
        )

    @property
    def properties(self):
        return _properties(self._properties_json)

    def to_dict(self):
        return {
            "kernel_index": self.kernel_index,
            "kind": self.kind,
            "artifact_identity": self.artifact_identity,
            "pipeline_identity": self.pipeline_identity,
            "source_identities": self.source_identities,
            "requested_grid_size": self.requested_grid_size,
            "selected_grid_size": self.selected_grid_size,
            "actual_grid_size": self.actual_grid_size,
            "requested_block_size": self.requested_block_size,
            "selected_block_size": self.selected_block_size,
            "actual_block_size": self.actual_block_size,
            "static_shared_bytes": self.static_shared_bytes,
            "dynamic_shared_bytes": self.dynamic_shared_bytes,
            "thread_local_bytes": self.thread_local_bytes,
            "abi_binding_names": self.abi_binding_names,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphPhysicalTaskManifest:
    """One node in the actual backend task DAG."""

    task_index: int
    kind: str
    depends_on: tuple[int, ...]
    region_ids: tuple[str, ...]
    kernel_indices: tuple[int, ...]
    queue: str
    pipeline_identity: str = ""

    effects: tuple[ResourceEffect, ...] = ()
    binding_names: tuple[str, ...] = ()
    temporary_bytes: int = 0
    _properties_json: str = field(default="{}", repr=False)

    @classmethod
    def create(
        cls,
        task_index,
        kind,
        *,
        depends_on=(),
        region_ids=(),
        kernel_indices=(),
        queue="default",
        pipeline_identity="",
        effects=(),
        binding_names=(),
        temporary_bytes=0,
        properties=None,
    ):
        effects = tuple(effects)
        if not all(isinstance(item, ResourceEffect) for item in effects):
            raise GraphPhysicalManifestError(
                "Graph physical task effects must be ResourceEffect values"
            )
        return cls(
            task_index=_nonnegative_int(task_index, "Graph physical task index"),
            kind=_required_text(kind, "Graph physical task kind"),
            depends_on=tuple(
                _nonnegative_int(item, "Graph physical task dependency")
                for item in depends_on
            ),
            region_ids=_normalized_strings(
                region_ids,
                "Graph physical task region",
            ),
            kernel_indices=tuple(
                _nonnegative_int(item, "Graph physical task kernel index")
                for item in kernel_indices
            ),
            queue=_required_text(queue, "Graph physical task queue"),
            pipeline_identity=(
                ""
                if not pipeline_identity
                else _required_text(
                    pipeline_identity,
                    "Graph physical task pipeline identity",
                )
            ),
            effects=effects,
            binding_names=_normalized_strings(
                binding_names,
                "Graph physical task binding",
            ),
            temporary_bytes=_nonnegative_int(
                temporary_bytes,
                "Graph physical task temporary bytes",
            ),
            _properties_json=_properties_json(properties),
        )

    @property
    def properties(self):
        return _properties(self._properties_json)

    def to_dict(self):
        return {
            "task_index": self.task_index,
            "kind": self.kind,
            "depends_on": self.depends_on,
            "region_ids": self.region_ids,
            "kernel_indices": self.kernel_indices,
            "queue": self.queue,
            "pipeline_identity": self.pipeline_identity,
            "effects": tuple(item.to_dict() for item in self.effects),
            "binding_names": self.binding_names,
            "temporary_bytes": self.temporary_bytes,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphPhysicalCommandManifest:
    """One recorded command containing one or more backend tasks."""

    command_index: int
    kind: str
    depends_on: tuple[int, ...]
    task_indices: tuple[int, ...]
    queue: str
    recording_scope: str = "graph"
    barrier_before: bool = False
    barrier_after: bool = False
    _properties_json: str = field(default="{}", repr=False)

    @classmethod
    def create(
        cls,
        command_index,
        kind,
        *,
        depends_on=(),
        task_indices=(),
        queue="default",
        recording_scope="graph",
        barrier_before=False,
        barrier_after=False,
        properties=None,
    ):
        return cls(
            command_index=_nonnegative_int(
                command_index,
                "Graph physical command index",
            ),
            kind=_required_text(kind, "Graph physical command kind"),
            depends_on=tuple(
                _nonnegative_int(item, "Graph physical command dependency")
                for item in depends_on
            ),
            task_indices=tuple(
                _nonnegative_int(item, "Graph physical command task index")
                for item in task_indices
            ),
            queue=_required_text(queue, "Graph physical command queue"),
            recording_scope=_required_text(
                recording_scope,
                "Graph physical command recording scope",
            ),
            barrier_before=bool(barrier_before),
            barrier_after=bool(barrier_after),
            _properties_json=_properties_json(properties),
        )

    @property
    def properties(self):
        return _properties(self._properties_json)

    def to_dict(self):
        return {
            "command_index": self.command_index,
            "kind": self.kind,
            "depends_on": self.depends_on,
            "task_indices": self.task_indices,
            "queue": self.queue,
            "recording_scope": self.recording_scope,
            "barrier_before": self.barrier_before,
            "barrier_after": self.barrier_after,
            "properties": self.properties,
        }


@dataclass(frozen=True)
class GraphPhysicalSubmissionManifest:
    """One actual submission partition in a materialized Graph."""

    submission_index: int
    depends_on: tuple[int, ...]
    command_indices: tuple[int, ...]
    queues: tuple[str, ...]
    recording_scope: str = "graph"
    exclusive_submission: bool = False
    replay_mode: str = "runtime_managed"

    def __post_init__(self):
        _nonnegative_int(self.submission_index, "Graph physical submission index")
        object.__setattr__(
            self,
            "depends_on",
            tuple(
                _nonnegative_int(item, "Graph physical submission dependency")
                for item in self.depends_on
            ),
        )
        object.__setattr__(
            self,
            "command_indices",
            tuple(
                _nonnegative_int(item, "Graph physical submission command index")
                for item in self.command_indices
            ),
        )
        object.__setattr__(
            self,
            "queues",
            _normalized_strings(
                self.queues,
                "Graph physical submission queue",
                sort=True,
            ),
        )
        _required_text(
            self.recording_scope,
            "Graph physical submission recording scope",
        )
        _required_text(self.replay_mode, "Graph physical submission replay mode")

    def to_dict(self):
        return {
            "submission_index": self.submission_index,
            "depends_on": self.depends_on,
            "command_indices": self.command_indices,
            "queues": self.queues,
            "recording_scope": self.recording_scope,
            "exclusive_submission": self.exclusive_submission,
            "replay_mode": self.replay_mode,
        }


@dataclass(frozen=True)
class GraphPhysicalResourceManifest:
    """One actual Graph-owned, session-owned, or external resource."""

    resource_id: str
    kind: str
    requested_bytes: int
    allocated_bytes: int
    alignment: int
    ownership: str
    lifetime: str
    scope: str = "internal"
    binding_name: str = ""
    exclusive_submission: bool = False
    allocation_members: tuple[str, ...] = ()
    allocation_count: int | None = None

    def __post_init__(self):
        _required_text(self.resource_id, "Graph physical resource ID")
        _required_text(self.kind, "Graph physical resource kind")
        members = tuple(self.allocation_members)
        for member in members:
            _required_text(member, "Graph allocation group member")
        if len(set(members)) != len(members):
            raise GraphPhysicalManifestError("Graph allocation group members must be unique")
        object.__setattr__(self, "allocation_members", tuple(sorted(members)))
        _nonnegative_int(self.allocation_count, "Graph physical allocation count", optional=True)
        _nonnegative_int(
            self.requested_bytes,
            "Graph physical resource requested bytes",
        )
        _nonnegative_int(
            self.allocated_bytes,
            "Graph physical resource allocated bytes",
        )
        if self.allocated_bytes < self.requested_bytes:
            raise GraphPhysicalManifestError(
                "Graph physical resource allocation is smaller than requested"
            )
        _positive_int(self.alignment, "Graph physical resource alignment")
        if self.alignment & (self.alignment - 1):
            raise GraphPhysicalManifestError(
                "Graph physical resource alignment must be a power of two"
            )
        if self.ownership not in (
            "fragment",
            "graph_instance",
            "shared",
            "external",
            "session",
        ):
            raise GraphPhysicalManifestError(
                "Graph physical resource ownership is invalid"
            )
        if self.lifetime not in ("task", "submission", "graph", "session"):
            raise GraphPhysicalManifestError(
                "Graph physical resource lifetime is invalid"
            )
        if self.scope not in ("internal", "public_external"):
            raise GraphPhysicalManifestError("Graph physical resource scope is invalid")
        if self.scope == "public_external":
            _required_text(
                self.binding_name,
                "Graph external physical resource binding",
            )
            if self.ownership != "external":
                raise GraphPhysicalManifestError(
                    "Graph public external resource must use external ownership"
                )
        elif self.binding_name:
            raise GraphPhysicalManifestError(
                "Graph internal physical resource cannot name a public binding"
            )

    def to_dict(self):
        return {
            "resource_id": self.resource_id,
            "kind": self.kind,
            "requested_bytes": self.requested_bytes,
            "allocated_bytes": self.allocated_bytes,
            "alignment": self.alignment,
            "ownership": self.ownership,
            "lifetime": self.lifetime,
            "scope": self.scope,
            "binding_name": self.binding_name,
            "exclusive_submission": self.exclusive_submission,
            **({"allocation_members": self.allocation_members} if self.allocation_members else {}),
            **({"allocation_count": self.allocation_count} if self.allocation_count is not None else {}),
        }


def _validate_contiguous_indices(values, attribute, role):
    actual = tuple(getattr(item, attribute) for item in values)
    expected = tuple(range(len(values)))
    if actual != expected:
        raise GraphPhysicalManifestError(
            f"{role} indices must be contiguous and ordered from zero"
        )


def _validate_dag(values, attribute, role):
    count = len(values)
    visiting = set()
    visited = set()

    def visit(index):
        if index in visited:
            return
        if index in visiting:
            raise GraphPhysicalManifestError(f"{role} dependencies contain a cycle")
        visiting.add(index)
        for dependency in getattr(values[index], attribute):
            if dependency >= count:
                raise GraphPhysicalManifestError(
                    f"{role} dependency references an absent node"
                )
            if dependency == index:
                raise GraphPhysicalManifestError(f"{role} cannot depend on itself")
            visit(dependency)
        visiting.remove(index)
        visited.add(index)

    for index in range(count):
        visit(index)


def _exact_public_abi(definition, bindings):
    expected = tuple(
        (item.name, item.kinds, item.required, item.scope)
        for item in definition.binding_abi
        if item.scope == "public"
    )
    actual = tuple(
        (item.name, item.kinds, item.required, item.scope)
        for item in bindings
        if item.scope == "public"
    )
    if actual != expected:
        raise GraphPhysicalManifestError(
            "materialized Graph public binding ABI differs from GraphDefinition"
        )


def _resource_identity_payload(resources, public_bindings):
    resource_indices = {
        resource.resource_id: index for index, resource in enumerate(resources)
    }
    return resource_indices, tuple(
        {
            "resource_index": index,
            "kind": resource.kind,
            "requested_bytes": resource.requested_bytes,
            "alignment": resource.alignment,
            "ownership": resource.ownership,
            "lifetime": resource.lifetime,
            "scope": resource.scope,
            "binding_name": (
                resource.binding_name if resource.scope == "public_external" else ""
            ),
            "exclusive_submission": resource.exclusive_submission,
            **({"allocation_members": resource.allocation_members} if resource.allocation_members else {}),
            **({"allocation_count": resource.allocation_count} if resource.allocation_count is not None else {}),
        }
        for index, resource in enumerate(resources)
    )


def _graph_resource_plan(graph):
    """Frozen allocation policy, independent of materialized cache/arena slots."""
    from taichi_forge.graph._graph import _graph_internal_storage_bytes

    spec = graph._spec
    temporary = spec.temporary_memory_plan
    plans = []
    assigned = set()
    owns_temporary = False
    for index, plan in enumerate(getattr(spec, "_storage_plans", ())):
        assigned.update(plan.binding_names)
        size = _graph_internal_storage_bytes(
            {name: spec.fixed_runtime_args[name] for name in plan.binding_names}
        )
        capacity = (
            graph._instance._temporary_arena.capacity if plan.temporary_arena else 0
        )
        if plan.temporary_arena:
            owns_temporary = True
            size += temporary.planned_peak_bytes * capacity
        size *= graph._workspace_lane_capacity
        plans.append(
            GraphPhysicalResourceManifest(
                resource_id=f"storage_plan:{index}",
                kind=plan.plan_id,
                requested_bytes=size,
                allocated_bytes=size,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                allocation_members=(
                    *plan.binding_names,
                    *(("temporary_arena",) if plan.temporary_arena else ()),
                ),
                allocation_count=(len(plan.binding_names) + capacity)
                * graph._workspace_lane_capacity,
                exclusive_submission=plan.ordered_temporary_reuse,
            )
        )
    size = _graph_internal_storage_bytes(
        {
            name: value
            for name, value in spec.fixed_runtime_args.items()
            if name not in assigned
        }
    )
    if size:
        plans.append(
            GraphPhysicalResourceManifest(
                resource_id="internal_storage",
                kind="graph_owned_storage",
                requested_bytes=size * graph._workspace_lane_capacity,
                allocated_bytes=size * graph._workspace_lane_capacity,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                allocation_count=graph._workspace_lane_capacity,
                exclusive_submission=graph._instance._exclusive_internal_storage,
            )
        )
    if temporary.planned_peak_bytes and not owns_temporary:
        arena_slots = (
            graph._instance._temporary_arena.capacity * graph._workspace_lane_capacity
        )
        plans.append(
            GraphPhysicalResourceManifest(
                resource_id="temporary_storage",
                kind="graph_temporary_slots",
                requested_bytes=temporary.planned_peak_bytes * arena_slots,
                allocated_bytes=temporary.planned_peak_bytes * arena_slots,
                alignment=1,
                ownership="graph_instance",
                lifetime="submission",
                allocation_count=temporary.slot_count * arena_slots,
            )
        )
    return tuple(plans)


def _effect_identity(effect, resource_indices, public_bindings):
    payload = effect.to_dict()
    resource = payload["resource"]
    if resource in public_bindings:
        payload["resource"] = f"public:{resource}"
    elif resource in resource_indices:
        payload["resource"] = f"internal:{resource_indices[resource]}"
    else:
        payload["resource"] = f"semantic:{resource}"
    return payload


def _without_diagnostics(value):
    """Exclude free-form reporter details from a physical identity.

    Actual artifact/pipeline IDs and the typed geometry/resource fields above
    decide identity. Diagnostic dictionaries remain in ``to_dict()`` but
    cannot manufacture a distinct route merely by changing a label.
    """

    payload = value.to_dict()
    payload.pop("properties", None)
    return payload


@dataclass(frozen=True)
class CompiledGraphPhysicalManifest:
    """Normalized actual backend facts for one complete materialized Graph."""

    materialized_physical_id: str
    backend: str
    semantic_graph_id: str
    recipe_id: str
    planned_physical_id: str
    kernels: tuple[GraphPhysicalKernelManifest, ...]
    tasks: tuple[GraphPhysicalTaskManifest, ...]
    commands: tuple[GraphPhysicalCommandManifest, ...]
    submissions: tuple[GraphPhysicalSubmissionManifest, ...]
    resources: tuple[GraphPhysicalResourceManifest, ...]
    binding_abi: tuple[GraphPhysicalBindingManifest, ...]
    task_topology_exact: bool
    command_topology_exact: bool
    allocation_topology_exact: bool
    resource_plan: tuple[GraphPhysicalResourceManifest, ...] = ()
    _provenance_json: str = field(default="{}", repr=False)

    @classmethod
    def from_graph(cls, definition, recipe, graph):
        """Observe a provider-built Forge Graph at the materialization boundary.

        This describes actual compiled work; it does not prove that an external
        provider's replacement implements the requested mathematical semantics.
        No observation is installed in Graph replay.
        """
        if graph.definition is definition:
            return observe_graph_physical_manifest(definition, recipe, graph)
        from taichi_forge.graph._recipes.providers import PROVIDER_OWNED_WHOLE_GRAPH_V1

        if recipe.assembly_protocol != PROVIDER_OWNED_WHOLE_GRAPH_V1:
            raise GraphPhysicalManifestError("replacement Graph requires whole-Graph provider assembly")
        actual_definition = graph.definition
        if actual_definition is None:
            raise GraphPhysicalManifestError("freeze the replacement Graph before compiling it")
        if _definition_binding_manifest(actual_definition) != _definition_binding_manifest(definition):
            raise GraphPhysicalManifestError("replacement Graph binding ABI differs from definition")
        baseline = actual_definition.recipe_catalog(providers=()).baseline.recipe
        observed = observe_graph_physical_manifest(actual_definition, baseline, graph)
        # Physical topology is observed on the actual replacement, while the
        # mapping to original semantics is a whole-Graph provider declaration.
        # Do not mutate either definition or install this mapping in replay.
        coverage = tuple(sorted({region_id for step in recipe.execution_steps for region_id in step.region_ids}))
        return cls.create(
            definition, recipe, backend=observed.backend,
            kernels=observed.kernels,
            tasks=tuple(replace(task, region_ids=coverage) for task in observed.tasks),
            commands=observed.commands, submissions=observed.submissions,
            resources=observed.resources, binding_abi=observed.binding_abi,
            resource_plan=observed.resource_plan,
            task_topology_exact=observed.task_topology_exact,
            command_topology_exact=observed.command_topology_exact,
            allocation_topology_exact=observed.allocation_topology_exact,
            provenance={
                **observed.provenance,
                "semantic_mapping": "provider_declared_whole_graph",
                "replacement_semantic_graph_id": actual_definition.semantic_graph_id,
            },
        )

    @classmethod
    def create(
        cls,
        definition,
        recipe,
        *,
        backend,
        kernels=(),
        tasks=(),
        commands=(),
        submissions=(),
        resources=(),
        resource_plan=None,
        binding_abi=(),
        task_topology_exact=True,
        command_topology_exact=True,
        allocation_topology_exact=True,
        provenance=None,
    ):
        if recipe.semantic_graph_id != definition.semantic_graph_id:
            raise GraphPhysicalManifestError(
                "physical manifest recipe belongs to a different semantic Graph"
            )
        if backend != definition.backend:
            raise GraphPhysicalManifestError(
                "physical manifest backend differs from GraphDefinition"
            )
        kernels = tuple(kernels)
        tasks = tuple(tasks)
        commands = tuple(commands)
        submissions = tuple(submissions)
        resources = tuple(resources)
        # Provider-supplied allocations normally describe a fixed plan. The
        # runtime observer supplies a separate plan because lazy caches and
        # backing pages are measurements, not execution choices.
        resource_plan = resources if resource_plan is None else tuple(resource_plan)
        binding_abi = tuple(binding_abi)
        typed = (
            (kernels, GraphPhysicalKernelManifest, "kernel"),
            (tasks, GraphPhysicalTaskManifest, "task"),
            (commands, GraphPhysicalCommandManifest, "command"),
            (submissions, GraphPhysicalSubmissionManifest, "submission"),
            (resources, GraphPhysicalResourceManifest, "resource"),
            (resource_plan, GraphPhysicalResourceManifest, "resource plan"),
            (binding_abi, GraphPhysicalBindingManifest, "binding"),
        )
        for values, expected_type, role in typed:
            if not all(isinstance(item, expected_type) for item in values):
                raise GraphPhysicalManifestError(
                    f"Graph physical {role} manifest has an invalid value"
                )
        _validate_contiguous_indices(kernels, "kernel_index", "Graph kernel")
        _validate_contiguous_indices(tasks, "task_index", "Graph task")
        _validate_contiguous_indices(commands, "command_index", "Graph command")
        _validate_contiguous_indices(
            submissions,
            "submission_index",
            "Graph submission",
        )
        _validate_contiguous_indices(
            binding_abi,
            "binding_index",
            "Graph binding",
        )
        _validate_dag(tasks, "depends_on", "Graph task")
        _validate_dag(commands, "depends_on", "Graph command")
        _validate_dag(submissions, "depends_on", "Graph submission")

        kernel_references = [index for task in tasks for index in task.kernel_indices]
        if any(index >= len(kernels) for index in kernel_references):
            raise GraphPhysicalManifestError(
                "Graph physical task references an absent kernel"
            )
        if set(kernel_references) != set(range(len(kernels))):
            raise GraphPhysicalManifestError(
                "every physical kernel must be referenced by a task"
            )
        task_references = [
            index for command in commands for index in command.task_indices
        ]
        if sorted(task_references) != list(range(len(tasks))):
            raise GraphPhysicalManifestError(
                "every physical task must belong to exactly one command"
            )
        command_references = [
            index for submission in submissions for index in submission.command_indices
        ]
        if sorted(command_references) != list(range(len(commands))):
            raise GraphPhysicalManifestError(
                "every physical command must belong to exactly one submission"
            )

        known_regions = {region.region_id for region in definition.regions}
        reported_regions = {
            region_id for task in tasks for region_id in task.region_ids
        }
        if not reported_regions <= known_regions:
            raise GraphPhysicalManifestError(
                "physical task references an unknown semantic region"
            )
        expected_source_regions = {
            region_id
            for step in recipe.execution_steps
            for region_id in step.region_ids
        }
        if reported_regions != expected_source_regions:
            raise GraphPhysicalManifestError(
                "physical task coverage is not the complete executable recipe"
            )

        binding_names = tuple(item.name for item in binding_abi)
        if len(binding_names) != len(set(binding_names)):
            raise GraphPhysicalManifestError(
                "Graph physical binding names must be unique"
            )
        _exact_public_abi(definition, binding_abi)
        known_binding_names = set(binding_names)
        for kernel in kernels:
            unknown = set(kernel.abi_binding_names).difference(known_binding_names)
            if unknown:
                raise GraphPhysicalManifestError(
                    "Graph physical kernel references an absent ABI binding"
                )
        for task in tasks:
            unknown = set(task.binding_names).difference(known_binding_names)
            if unknown:
                raise GraphPhysicalManifestError(
                    "Graph physical task references an absent ABI binding"
                )

        public_bindings = {item.name for item in binding_abi if item.scope == "public"}
        for resource_set in (resources, resource_plan):
            resource_ids = tuple(item.resource_id for item in resource_set)
            if len(resource_ids) != len(set(resource_ids)):
                raise GraphPhysicalManifestError(
                    "Graph physical resource IDs must be unique"
                )
            for resource in resource_set:
                if resource.scope == "public_external" and resource.binding_name not in public_bindings:
                    raise GraphPhysicalManifestError(
                        "Graph external physical resource has no public ABI binding"
                    )

        resource_indices, resource_payload = _resource_identity_payload(
            resource_plan,
            public_bindings,
        )
        physical_payload = {
            "schema": _PHYSICAL_MANIFEST_SCHEMA,
            "backend": backend,
            "semantic_graph_id": definition.semantic_graph_id,
            "kernels": tuple(_without_diagnostics(item) for item in kernels),
            "tasks": tuple(
                {
                    **_without_diagnostics(item),
                    "effects": tuple(
                        _effect_identity(
                            effect,
                            resource_indices,
                            public_bindings,
                        )
                        for effect in item.effects
                    ),
                }
                for item in tasks
            ),
            "commands": tuple(_without_diagnostics(item) for item in commands),
            "submissions": tuple(item.to_dict() for item in submissions),
            "resources": resource_payload,
            "binding_abi": tuple(item.to_dict() for item in binding_abi),
            "exactness": {
                "task_topology": bool(task_topology_exact),
                "command_topology": bool(command_topology_exact),
            },
        }
        return cls(
            materialized_physical_id=(
                f"materialized-physical:{_digest(physical_payload)}"
            ),
            backend=backend,
            semantic_graph_id=definition.semantic_graph_id,
            recipe_id=recipe.recipe_id,
            planned_physical_id=recipe.planned_physical_id,
            kernels=kernels,
            tasks=tasks,
            commands=commands,
            submissions=submissions,
            resources=resources,
            resource_plan=resource_plan,
            binding_abi=binding_abi,
            task_topology_exact=bool(task_topology_exact),
            command_topology_exact=bool(command_topology_exact),
            allocation_topology_exact=bool(allocation_topology_exact),
            _provenance_json=_properties_json(provenance),
        )

    @property
    def provenance(self):
        return _properties(self._provenance_json)

    @property
    def identity_complete(self):
        return bool(
            self.task_topology_exact
            and self.command_topology_exact
            and self.allocation_topology_exact
        )

    @property
    def persistent_requested_bytes(self):
        return sum(
            item.requested_bytes
            for item in self.resources
            if item.lifetime in ("graph", "session")
        )

    @property
    def persistent_allocated_bytes(self):
        return sum(
            item.allocated_bytes
            for item in self.resources
            if item.lifetime in ("graph", "session")
        )

    @property
    def transient_requested_bytes(self):
        return sum(
            item.requested_bytes
            for item in self.resources
            if item.lifetime in ("task", "submission")
        )

    @property
    def transient_allocated_bytes(self):
        return sum(
            item.allocated_bytes
            for item in self.resources
            if item.lifetime in ("task", "submission")
        )

    def to_dict(self):
        return {
            "schema": _PHYSICAL_MANIFEST_SCHEMA,
            "materialized_physical_id": self.materialized_physical_id,
            "backend": self.backend,
            "semantic_graph_id": self.semantic_graph_id,
            "recipe_id": self.recipe_id,
            "planned_physical_id": self.planned_physical_id,
            "kernels": tuple(item.to_dict() for item in self.kernels),
            "tasks": tuple(item.to_dict() for item in self.tasks),
            "commands": tuple(item.to_dict() for item in self.commands),
            "submissions": tuple(item.to_dict() for item in self.submissions),
            "resources": tuple(item.to_dict() for item in self.resources),
            "resource_plan": tuple(
                {key: value for key, value in item.to_dict().items() if key != "allocated_bytes"}
                for item in self.resource_plan
            ),
            "binding_abi": tuple(item.to_dict() for item in self.binding_abi),
            "exactness": {
                "task_topology": self.task_topology_exact,
                "command_topology": self.command_topology_exact,
                "allocation_topology": self.allocation_topology_exact,
                "identity_complete": self.identity_complete,
            },
            "memory": {
                "persistent_requested_bytes": self.persistent_requested_bytes,
                "persistent_allocated_bytes": self.persistent_allocated_bytes,
                "transient_requested_bytes": self.transient_requested_bytes,
                "transient_allocated_bytes": self.transient_allocated_bytes,
            },
            "provenance": self.provenance,
        }


def _definition_binding_manifest(definition):
    return tuple(
        GraphPhysicalBindingManifest(
            binding_index=index,
            name=item.name,
            kinds=item.kinds,
            required=item.required,
            scope=item.scope,
        )
        for index, item in enumerate(definition.binding_abi)
    )


def _stage_source_regions(definition, recipe, pipeline):
    source_regions = tuple(source.region_id for source in definition.sources)
    path_regions = []

    for stage in pipeline:
        parts = str(stage.get("path_id", "")).split("/")
        try:
            source_index = int(parts[1].split(":", 1)[0])
        except (IndexError, ValueError):
            path_regions = []
            break
        prefix = f"graph/{source_index}:"
        regions = tuple(
            source.region_id
            for source in definition.sources
            if source.path.startswith(prefix)
        )
        if not regions:
            path_regions = []
            break
        path_regions.append(regions)

    if path_regions and {
        region_id for regions in path_regions for region_id in regions
    } == set(source_regions):
        # One semantic source may expand to several physical dispatches (for
        # example a partial/finalize reduction), while one structured source
        # may lower into several runtime stages. Runtime stage lineage, rather
        # than physical task cardinality, is the stable coverage relation.
        return tuple(path_regions)

    counts = tuple(
        int(stage["dispatch_count"])
        + int(stage["source_native_count"])
        + int(stage["kind"] == "observation")
        for stage in pipeline
    )
    if sum(counts) != len(source_regions):
        recipe_source_regions = tuple(
            dict.fromkeys(
                region_id
                for step in recipe.execution_steps
                for region_id in step.region_ids
            )
        )
        if len(recipe_source_regions) == 1:
            return tuple(recipe_source_regions for _ in pipeline)
        raise GraphPhysicalManifestError(
            "backend pipeline source count differs from GraphDefinition"
        )
    cursor = 0
    result = []
    for count in counts:
        result.append(source_regions[cursor : cursor + count])
        cursor += count
    return tuple(result)


def observe_graph_physical_manifest(definition, recipe, graph):
    """Translate one materialized Graph into the backend-neutral manifest."""

    if graph.definition is not definition:
        raise GraphPhysicalManifestError(
            "materialized Graph does not own this GraphDefinition"
        )
    pipeline = tuple(graph._spec.pipeline_definition)
    stage_regions = getattr(graph._spec, "_recipe_node_source_regions", None)
    if stage_regions is None:
        stage_regions = _stage_source_regions(definition, recipe, pipeline)
    elif (
        len(stage_regions) != len(pipeline)
        or any(not regions for regions in stage_regions)
        or {region for regions in stage_regions for region in regions}
        != {source.region_id for source in definition.sources}
    ):
        raise GraphPhysicalManifestError(
            "materialized node lineage does not cover the frozen Graph sources"
        )
    kernels = []
    tasks = []
    commands = []
    task_topology_exact = True
    # The current native interface reports exact compiled tasks but not the
    # backend command DAG/replay partition. Do not promote our one-command-per-
    # task normalization into a false exactness claim.
    command_topology_exact = False
    static_resources = {}

    def append_task(
        kind,
        regions,
        *,
        queue="default",
        kernel=None,
        pipeline_identity="",
        properties=None,
        depends_on=None,
        command_depends_on=None,
    ):
        task_index = len(tasks)
        if depends_on is None:
            depends_on = () if task_index == 0 else (task_index - 1,)
        tasks.append(
            GraphPhysicalTaskManifest.create(
                task_index,
                kind,
                depends_on=depends_on,
                region_ids=regions,
                kernel_indices=(() if kernel is None else (kernel.kernel_index,)),
                queue=queue,
                pipeline_identity=pipeline_identity,
                properties=properties,
            )
        )
        commands.append(
            GraphPhysicalCommandManifest.create(
                len(commands),
                kind,
                depends_on=(
                    (() if not commands else (len(commands) - 1,))
                    if command_depends_on is None
                    else command_depends_on
                ),
                task_indices=(task_index,),
                queue=queue,
                properties=properties,
            )
        )
        return task_index

    for stage, regions in zip(pipeline, stage_regions):
        stage_tasks = tuple(stage["tasks"])
        native_actions = tuple(stage["native_actions"])
        external_commands = tuple(stage.get("external_commands", ()))
        if external_commands and len(external_commands) != len(native_actions):
            raise GraphPhysicalManifestError("external command positions do not cover the retained native actions")
        if external_commands:
            # Command order is observed, but vendor-internal kernel topology is
            # not. A retained library call is not one measured CUDA kernel.
            task_topology_exact = False
        next_external = 0

        def append_native_action(action, external=None):
            payload = action.to_dict(_static_resources=static_resources)
            if external is not None:
                payload = {**payload, **external}
            physical_plan_id = action.physical_plan_id or ("native-action:" + _digest(payload))
            append_task(
                "native_action",
                regions,
                queue=action.queue,
                pipeline_identity=_combined_pipeline_identity(physical_plan_id, stage_plan_identity),
                properties=payload,
            )

        stage_plan_identity = _stage_plan_identity(stage)
        parallel_groups = tuple(stage.get("parallel_dispatch_groups", ()))
        branch_by_dispatch = {
            dispatch_index: branch_index
            for branch_index, group in enumerate(parallel_groups)
            for dispatch_index in group
        }
        parallel_begin = (
            parallel_groups[0][0] if parallel_groups else None
        )
        parallel_end = (
            parallel_groups[-1][-1] + 1 if parallel_groups else None
        )
        last_task_by_dispatch = {}
        if stage_tasks:
            for raw in stage_tasks:
                while (
                    next_external < len(external_commands)
                    and external_commands[next_external]["dispatch_index"] < raw.dispatch_index
                ):
                    append_native_action(native_actions[next_external], external_commands[next_external])
                    next_external += 1
                raw_payload = asdict(raw)
                kernel = GraphPhysicalKernelManifest.create(
                    len(kernels),
                    raw.task_type,
                    raw.task_id,
                    pipeline_identity=(raw.optimization_spec_id or raw.logical_task_id),
                    source_identities=(raw.logical_task_id,),
                    requested_grid_size=raw.requested_grid_size,
                    selected_grid_size=raw.selected_grid_size,
                    actual_grid_size=raw.actual_grid_size,
                    requested_block_size=raw.requested_block_size,
                    selected_block_size=raw.selected_block_size,
                    actual_block_size=raw.actual_block_size,
                    static_shared_bytes=raw.static_shared_bytes,
                    dynamic_shared_bytes=raw.dynamic_shared_bytes,
                    thread_local_bytes=raw.thread_local_bytes,
                    properties=raw_payload,
                )
                kernels.append(kernel)
                dispatch_index = int(raw.dispatch_index)
                queue = "default"
                task_dependencies = None
                if parallel_groups:
                    branch_index = branch_by_dispatch.get(dispatch_index)
                    previous_same_dispatch = last_task_by_dispatch.get(
                        dispatch_index
                    )
                    if previous_same_dispatch is not None:
                        task_dependencies = (previous_same_dispatch,)
                    elif branch_index is not None:
                        queue = f"cuda_branch:{branch_index}"
                        group = parallel_groups[branch_index]
                        if dispatch_index == group[0]:
                            task_dependencies = (
                                ()
                                if parallel_begin == 0
                                else (
                                    last_task_by_dispatch[parallel_begin - 1],
                                )
                            )
                        else:
                            task_dependencies = (
                                last_task_by_dispatch[dispatch_index - 1],
                            )
                    elif dispatch_index == parallel_end:
                        task_dependencies = tuple(
                            last_task_by_dispatch[group[-1]]
                            for group in parallel_groups
                        )
                    elif dispatch_index > 0:
                        task_dependencies = (
                            last_task_by_dispatch[dispatch_index - 1],
                        )
                    else:
                        task_dependencies = ()
                task_index = append_task(
                    "kernel",
                    regions,
                    queue=queue,
                    kernel=kernel,
                    pipeline_identity=_combined_pipeline_identity(
                        kernel.pipeline_identity,
                        stage_plan_identity,
                    ),
                    properties={
                        "stage_kind": stage["kind"],
                        "dispatch_index": raw.dispatch_index,
                        "source_dispatch_count": raw.source_dispatch_count,
                        "parallel_branch": branch_by_dispatch.get(
                            dispatch_index
                        ),
                    },
                    depends_on=task_dependencies,
                    command_depends_on=task_dependencies,
                )
                last_task_by_dispatch[dispatch_index] = task_index
        for index in range(next_external, len(native_actions)):
            append_native_action(
                native_actions[index], external_commands[index] if external_commands else None
            )
        if not stage_tasks and not native_actions and regions:
            append_task(
                stage["kind"],
                regions,
                pipeline_identity=stage_plan_identity,
                properties={
                    "path_id": stage["path_id"],
                    "physical_dispatch_count": stage["physical_dispatch_count"],
                    "task_mapping_status": stage["task_mapping_status"],
                },
            )
            task_topology_exact = False

    execution = graph.execution_stats()
    memory = execution.memory
    concurrent_workspace_pair = bool(
        getattr(graph, "_cuda_concurrent_workspace_pair", False)
    )
    resources = []
    bounded_control_bytes = int(memory.persistent_bounded_control_bytes)
    persistent_non_bounded_bytes = int(memory.persistent_bytes) - bounded_control_bytes
    if persistent_non_bounded_bytes < 0:
        raise GraphPhysicalManifestError("Graph bounded-control memory exceeds total persistent memory")
    # These owners allocated actual Program-registered arrays. Their grouping
    # is physical identity; current/high pool backing pages are observations,
    # not requested allocations and must not perturb identity after replay.
    for index, pool in enumerate(execution.storage_pools):
        if pool.closed or pool.allocation_count == 0:
            raise GraphPhysicalManifestError("materialized Graph storage pool has no live allocation owner")
        resources.append(
            GraphPhysicalResourceManifest(
                resource_id=f"graph:storage_pool:{index}",
                kind=pool.allocator,
                requested_bytes=pool.requested_bytes,
                allocated_bytes=pool.requested_bytes,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                allocation_members=pool.allocation_members,
                allocation_count=pool.allocation_count,
                exclusive_submission=memory.internal_storage_exclusive,
            )
        )
        persistent_non_bounded_bytes -= pool.requested_bytes
    if persistent_non_bounded_bytes < 0:
        raise GraphPhysicalManifestError("Graph pool storage exceeds reported persistent allocation")
    if persistent_non_bounded_bytes:
        resources.append(
            GraphPhysicalResourceManifest(
                resource_id=("graph:persistent_non_bounded" if bounded_control_bytes else "graph:persistent"),
                kind=("graph_owned_non_bounded" if bounded_control_bytes else "graph_owned_aggregate"),
                requested_bytes=persistent_non_bounded_bytes,
                allocated_bytes=persistent_non_bounded_bytes,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                exclusive_submission=memory.internal_storage_exclusive,
            )
        )
    if bounded_control_bytes:
        resources.append(
            GraphPhysicalResourceManifest(
                resource_id="graph:bounded_control",
                kind="bounded_control_state",
                requested_bytes=bounded_control_bytes,
                allocated_bytes=bounded_control_bytes,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                exclusive_submission=True,
            )
        )
    provider_persistent = int(memory.provider_generation_known_resident_requested_bytes)
    if provider_persistent:
        resources.append(
            GraphPhysicalResourceManifest(
                resource_id="graph:provider_generation",
                kind="provider_owned_aggregate",
                requested_bytes=provider_persistent,
                allocated_bytes=provider_persistent,
                alignment=1,
                ownership="graph_instance",
                lifetime="graph",
                exclusive_submission=memory.internal_storage_exclusive,
            )
        )
    transient_requested = max(
        int(memory.planned_temporary_bytes),
        int(memory.transient_temporary_bytes),
    )
    if transient_requested:
        resources.append(
            GraphPhysicalResourceManifest(
                resource_id="graph:transient",
                kind="graph_temporary_aggregate",
                requested_bytes=transient_requested,
                allocated_bytes=transient_requested,
                alignment=1,
                ownership="graph_instance",
                lifetime="submission",
            )
        )
    submissions = (
        (
            GraphPhysicalSubmissionManifest(
                submission_index=0,
                depends_on=(),
                command_indices=tuple(range(len(commands))),
                queues=(
                    ("cuda_workspace:0", "cuda_workspace:1", "default")
                    if concurrent_workspace_pair
                    else tuple(sorted({command.queue for command in commands}))
                ),
                recording_scope="whole_graph",
                exclusive_submission=memory.internal_storage_exclusive,
                replay_mode=(
                    "cuda_concurrent_complete_graph_pair"
                    if concurrent_workspace_pair
                    else graph._instance.physical_submission_mode
                ),
            ),
        )
        if commands
        else ()
    )
    return CompiledGraphPhysicalManifest.create(
        definition,
        recipe,
        backend=definition.backend,
        kernels=kernels,
        tasks=tasks,
        commands=commands,
        submissions=submissions,
        resources=resources,
        resource_plan=_graph_resource_plan(graph),
        binding_abi=_definition_binding_manifest(definition),
        task_topology_exact=task_topology_exact,
        command_topology_exact=command_topology_exact,
        allocation_topology_exact=(
            memory.provider_generation_requested_bytes_complete
            and memory.opaque_driver_bytes in (None, 0)
        ),
        provenance={
            "core_commit": definition.compile_provenance.core_commit,
            "execution_arch": execution.arch,
            "execution_path": execution.execution_path,
            "static_layout_fingerprint": execution.static_layout_fingerprint,
            "workspace_concurrency": (
                "cuda_complete_graph_pair" if concurrent_workspace_pair else "serial"
            ),
        },
    )


observe_baseline_physical_manifest = observe_graph_physical_manifest


def _combined_pipeline_identity(*identities):
    return "|".join(str(item) for item in identities if item)


def _stage_plan_identity(stage):
    identities = []
    physical_plan_id = str(stage.get("physical_plan_id", ""))
    if physical_plan_id:
        identities.append(physical_plan_id)
    bounded = tuple(
        asdict(item["domain"]) for item in stage.get("bounded_dispatches", ())
    )
    if bounded:
        identities.append("bounded-scope:" + _digest(bounded))
    return "|".join(identities)


__all__ = [
    "CompiledGraphPhysicalManifest",
    "GraphPhysicalBindingManifest",
    "GraphPhysicalCommandManifest",
    "GraphPhysicalKernelManifest",
    "GraphPhysicalManifestError",
    "GraphPhysicalResourceManifest",
    "GraphPhysicalSubmissionManifest",
    "GraphPhysicalTaskManifest",
    "observe_baseline_physical_manifest",
    "observe_graph_physical_manifest",
]
