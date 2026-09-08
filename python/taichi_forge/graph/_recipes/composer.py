"""Whole-Graph composition of provider-owned recipe fragments."""

from dataclasses import dataclass

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.graph._recipes.fragments import GraphRecipeFragment
from taichi_forge.graph._recipes.providers import (
    PROVIDER_OWNED_WHOLE_GRAPH_V1,
    RUNTIME_GRAPH_ASSEMBLY_V1,
)

_RECIPE_SCHEMA = "taichi_forge.graph_executable_recipe.v2"
_PLANNED_RECIPE_SCHEMA = "taichi_forge.graph_recipe_planned_physical.v2"


class GraphRecipeCompositionError(ValueError):
    """One fragment set cannot form a legal complete Graph recipe."""


@dataclass(frozen=True)
class GraphRegionSelection:
    """Exact baseline-or-fragment assignment for one semantic region."""

    region_id: str
    source: str
    fragment_id: str = ""

    def to_dict(self):
        return {
            "region_id": self.region_id,
            "source": self.source,
            "fragment_id": self.fragment_id,
        }


@dataclass(frozen=True)
class GraphRecipeExecutionStep:
    """One ordered baseline or replacement step in a complete recipe."""

    step_index: int
    source: str
    region_ids: tuple[str, ...]
    depends_on: tuple[int, ...]
    fragment_id: str = ""
    task_ids: tuple[str, ...] = ()

    def to_dict(self):
        return {
            "step_index": self.step_index,
            "source": self.source,
            "region_ids": self.region_ids,
            "depends_on": self.depends_on,
            "fragment_id": self.fragment_id,
            "task_ids": self.task_ids,
        }


@dataclass(frozen=True)
class GraphExecutableRecipe:
    """A complete Graph recipe with exact semantic coverage."""

    recipe_id: str
    semantic_graph_id: str
    planned_physical_id: str
    assembly_protocol: str
    assembly_provider_namespace: str
    fragments: tuple[GraphRecipeFragment, ...]
    baseline_coverage_region_ids: tuple[str, ...]
    region_selections: tuple[GraphRegionSelection, ...]
    execution_steps: tuple[GraphRecipeExecutionStep, ...]
    queues: tuple[str, ...]
    barrier_count: int
    exclusive_submission: bool
    declared_persistent_resource_bytes: int
    declared_transient_resource_bytes: int

    @property
    def fragment_ids(self):
        return tuple(fragment.fragment_id for fragment in self.fragments)

    @property
    def executor_fragment(self):
        return next((fragment for fragment in self.fragments if fragment.submission.executor_kind), None)

    def to_dict(self):
        return {
            "schema": _RECIPE_SCHEMA,
            "recipe_id": self.recipe_id,
            "semantic_graph_id": self.semantic_graph_id,
            "planned_physical_id": self.planned_physical_id,
            "assembly": {
                "protocol": self.assembly_protocol,
                "provider_namespace": self.assembly_provider_namespace,
            },
            "fragment_ids": self.fragment_ids,
            "fragments": tuple(fragment.to_dict() for fragment in self.fragments),
            "baseline_coverage_region_ids": self.baseline_coverage_region_ids,
            "region_selections": tuple(selection.to_dict() for selection in self.region_selections),
            "execution_steps": tuple(step.to_dict() for step in self.execution_steps),
            "submission": {
                "queues": self.queues,
                "barrier_count": self.barrier_count,
                "exclusive_submission": self.exclusive_submission,
                **(
                    {
                        "executor_fragment_id": self.executor_fragment.fragment_id,
                        "executor_kind": self.executor_fragment.submission.executor_kind,
                    }
                    if self.executor_fragment is not None
                    else {}
                ),
            },
            "resources": {
                "declared_persistent_bytes": self.declared_persistent_resource_bytes,
                "declared_transient_bytes": self.declared_transient_resource_bytes,
            },
        }


def _region_paths_overlap(left, right):
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _subresources_overlap(left, right):
    return left is None or right is None or left == right


def _effect_writes(effect):
    return effect.access != GraphAccess.READ


def _effects_conflict(left, right):
    return bool(
        left.resource == right.resource
        and _subresources_overlap(left.subresource, right.subresource)
        and (_effect_writes(left) or _effect_writes(right))
    )


class GraphRecipeComposer:
    """Validate fragments and fill every uncovered region from baseline."""

    def __init__(self, definition, *, available_capabilities=()):
        self.definition = definition
        self.available_capabilities = frozenset(available_capabilities)
        self._regions = {region.region_id: region for region in definition.regions}
        self._region_indices = {
            region.region_id: index for index, region in enumerate(definition.regions)
        }
        self._public_bindings = {
            item.name: item for item in definition.binding_abi if item.scope == "public"
        }

    def _ordered_fragments(self, fragments):
        return tuple(
            sorted(
                fragments,
                key=lambda fragment: (
                    min(
                        self._region_indices[region_id]
                        for region_id in fragment.coverage_region_ids
                    ),
                    fragment.fragment_id,
                ),
            )
        )

    def _validate_fragment(self, fragment):
        if not isinstance(fragment, GraphRecipeFragment):
            raise TypeError(
                "Graph recipe composition requires GraphRecipeFragment values"
            )
        if fragment.semantic_graph_id != self.definition.semantic_graph_id:
            raise GraphRecipeCompositionError(
                "fragment semantic Graph identity does not match the definition"
            )
        unknown = set(fragment.coverage_region_ids).difference(self._regions)
        if unknown:
            raise GraphRecipeCompositionError(
                "fragment covers unknown semantic regions: "
                + ", ".join(sorted(unknown))
            )
        expected_digests = tuple(
            (region_id, self._regions[region_id].semantic_digest)
            for region_id in fragment.coverage_region_ids
        )
        if fragment.semantic_region_digests != expected_digests:
            raise GraphRecipeCompositionError(
                "fragment semantic region contract does not match the definition"
            )
        covered = set(fragment.coverage_region_ids)
        for region_id in fragment.coverage_region_ids:
            path = self._regions[region_id].path
            descendants = {
                candidate.region_id
                for candidate in self.definition.regions
                if candidate.path.startswith(path + "/")
            }
            if descendants and not descendants <= covered:
                raise GraphRecipeCompositionError(
                    "fragment coverage of a structural region must include its subtree"
                )
        source_indices = tuple(
            index
            for index, source in enumerate(self.definition.sources)
            if source.region_id in covered
        )
        if not source_indices:
            raise GraphRecipeCompositionError(
                "fragment coverage must contain an executable source region"
            )
        if source_indices and source_indices != tuple(
            range(source_indices[0], source_indices[-1] + 1)
        ):
            raise GraphRecipeCompositionError(
                "one fragment cannot skip an intermediate executable region"
            )
        if len(source_indices) > 1:
            source_paths = tuple(
                self._regions[self.definition.sources[index].region_id].path
                for index in source_indices
            )
            covered_structural_paths = tuple(
                self._regions[region_id].path
                for region_id in fragment.coverage_region_ids
                if any(
                    candidate.path.startswith(self._regions[region_id].path + "/")
                    for candidate in self.definition.regions
                )
            )
            owns_common_ancestor = any(
                all(path.startswith(structural_path + "/") for path in source_paths)
                for structural_path in covered_structural_paths
            )
            direct_parents = {path.rsplit("/", 1)[0] for path in source_paths}
            # A consecutive interval of complete sibling subtrees also forms
            # one sequential region (for example CGraph / native / CGraph).
            # Leaf parents differ there, but each covered subtree was checked
            # above and source_indices already forbids skipping an executable.
            coverage_roots = tuple(
                self._regions[region_id]
                for region_id in fragment.coverage_region_ids
                if self._regions[region_id].parent_region_id not in covered
            )
            root_parents = {region.parent_region_id for region in coverage_roots}
            root_ordinals = sorted(region.child_index for region in coverage_roots)
            sequential_subtrees = (
                len(root_parents) == 1
                and None not in root_parents
                and self._regions[next(iter(root_parents))].kind == "sequential_region"
                and root_ordinals == list(range(root_ordinals[0], root_ordinals[-1] + 1))
            )
            if (
                not owns_common_ancestor
                and len(direct_parents) != 1
                and not sequential_subtrees
            ):
                raise GraphRecipeCompositionError(
                    "multi-region fragment must share one sequential parent or "
                    "replace a complete structural subtree"
                )
        if (
            fragment.backend_requirements
            and self.definition.backend not in fragment.backend_requirements
        ):
            raise GraphRecipeCompositionError(
                f"fragment is unavailable on backend {self.definition.backend}"
            )
        missing_capabilities = set(fragment.capability_requirements).difference(
            self.available_capabilities
        )
        if missing_capabilities:
            raise GraphRecipeCompositionError(
                "fragment requires unavailable capabilities: "
                + ", ".join(sorted(missing_capabilities))
            )

        task_binding_names = {
            binding.name for task in fragment.tasks for binding in task.bindings
        }
        task_temporary_names = {
            temporary.name for task in fragment.tasks for temporary in task.temporaries
        }
        internal_names = (
            {resource.name for resource in fragment.resources}
            | task_binding_names
            | task_temporary_names
        )
        for requirement in fragment.binding_requirements:
            if requirement.scope == "fragment_internal":
                if requirement.name not in internal_names:
                    raise GraphRecipeCompositionError(
                        "fragment-internal binding has no task or resource owner: "
                        + requirement.name
                    )
                continue
            actual = self._public_bindings.get(requirement.name)
            if actual is None:
                raise GraphRecipeCompositionError(
                    "fragment requires an unknown public binding: " + requirement.name
                )
            if (
                requirement.kinds
                and "opaque" not in actual.kinds
                and not set(requirement.kinds).intersection(actual.kinds)
            ):
                raise GraphRecipeCompositionError(
                    "fragment public binding kind is incompatible: " + requirement.name
                )

        public_names = set(self._public_bindings)
        for resource in fragment.resources:
            if resource.name in public_names and resource.ownership != "external":
                raise GraphRecipeCompositionError(
                    "fragment-owned resource collides with a public binding: "
                    + resource.name
                )
        for task in fragment.tasks:
            for temporary in task.temporaries:
                if temporary.name in public_names:
                    raise GraphRecipeCompositionError(
                        "fragment temporary collides with a public binding: "
                        + temporary.name
                    )

    def _validate_coverage_compatibility(self, fragments):
        for index, left in enumerate(fragments):
            for right in fragments[index + 1 :]:
                for left_id in left.coverage_region_ids:
                    for right_id in right.coverage_region_ids:
                        if _region_paths_overlap(
                            self._regions[left_id].path,
                            self._regions[right_id].path,
                        ):
                            raise GraphRecipeCompositionError(
                                "fragments have overlapping semantic coverage"
                            )

    def _computation_fragments(self, fragments):
        executors = tuple(fragment for fragment in fragments if fragment.submission.executor_kind)
        regions = tuple(fragment for fragment in fragments if not fragment.submission.executor_kind)
        if not executors:
            return regions
        if len(executors) != 1:
            raise GraphRecipeCompositionError("one complete Graph can select only one executor")
        executor = executors[0]
        if executor.assembly_protocol != RUNTIME_GRAPH_ASSEMBLY_V1:
            raise GraphRecipeCompositionError("Graph executor requires runtime Graph assembly")
        if not {source.region_id for source in self.definition.sources} <= set(executor.coverage_region_ids):
            raise GraphRecipeCompositionError("Graph executor must cover every executable source")
        kind = executor.submission.executor_kind
        for fragment in regions:
            if kind not in fragment.submission.compatible_executor_kinds:
                raise GraphRecipeCompositionError("region fragment is not compatible with Graph executor: " + kind)
            if fragment.submission.queue != executor.submission.queue:
                raise GraphRecipeCompositionError("Graph executor and region must use the same queue")
        return regions

    def _validate_resource_compatibility(self, fragments):
        resources = {}
        temporary_owners = {}
        for fragment in fragments:
            for resource in fragment.resources:
                previous = resources.get(resource.name)
                if previous is None:
                    resources[resource.name] = resource
                    continue
                shareable = bool(
                    resource == previous
                    and resource.ownership in ("shared", "external")
                )
                if not shareable:
                    raise GraphRecipeCompositionError(
                        "fragments require incompatible resources named "
                        + resource.name
                    )
            for task in fragment.tasks:
                for temporary in task.temporaries:
                    owner = temporary_owners.setdefault(
                        temporary.name,
                        fragment.fragment_id,
                    )
                    if owner != fragment.fragment_id:
                        raise GraphRecipeCompositionError(
                            "fragments require colliding temporaries named "
                            + temporary.name
                        )

    def _validate_submission_compatibility(self, fragments):
        for index, left in enumerate(fragments):
            left_effects = tuple(
                effect for task in left.tasks for effect in task.effects
            )
            for right in fragments[index + 1 :]:
                if left.submission.queue == right.submission.queue:
                    continue
                right_effects = tuple(
                    effect for task in right.tasks for effect in task.effects
                )
                has_hazard = any(
                    _effects_conflict(left_effect, right_effect)
                    for left_effect in left_effects
                    for right_effect in right_effects
                )
                if has_hazard and not (
                    left.submission.barrier_after or right.submission.barrier_before
                ):
                    raise GraphRecipeCompositionError(
                        "cross-queue fragment effects require an explicit barrier"
                    )

    def _execution_steps(self, fragments):
        fragment_by_region = {
            region_id: fragment
            for fragment in fragments
            for region_id in fragment.coverage_region_ids
        }
        emitted_fragments = set()
        steps = []
        for source in self.definition.sources:
            fragment = fragment_by_region.get(source.region_id)
            if fragment is None:
                steps.append(
                    GraphRecipeExecutionStep(
                        step_index=len(steps),
                        source="baseline",
                        region_ids=(source.region_id,),
                        depends_on=(() if not steps else (len(steps) - 1,)),
                    )
                )
                continue
            if fragment.fragment_id in emitted_fragments:
                continue
            emitted_fragments.add(fragment.fragment_id)
            source_region_ids = tuple(
                candidate.region_id
                for candidate in self.definition.sources
                if candidate.region_id in fragment.coverage_region_ids
            )
            steps.append(
                GraphRecipeExecutionStep(
                    step_index=len(steps),
                    source="fragment",
                    region_ids=source_region_ids,
                    depends_on=(() if not steps else (len(steps) - 1,)),
                    fragment_id=fragment.fragment_id,
                    task_ids=tuple(task.task_id for task in fragment.tasks),
                )
            )
        return tuple(steps)

    def compose(self, fragments=()):
        fragments = tuple(fragments)
        if not fragments:
            selections = tuple(
                GraphRegionSelection(region.region_id, "baseline")
                for region in self.definition.regions
            )
            execution_steps = self._execution_steps(())
            return GraphExecutableRecipe(
                recipe_id=self.definition.baseline_recipe.recipe_id,
                semantic_graph_id=self.definition.semantic_graph_id,
                planned_physical_id=(
                    self.definition.baseline_recipe.planned_physical_id
                ),
                assembly_protocol=RUNTIME_GRAPH_ASSEMBLY_V1,
                assembly_provider_namespace="",
                fragments=(),
                baseline_coverage_region_ids=(
                    self.definition.baseline_recipe.coverage_region_ids
                ),
                region_selections=selections,
                execution_steps=execution_steps,
                queues=("default",),
                barrier_count=0,
                exclusive_submission=False,
                declared_persistent_resource_bytes=0,
                declared_transient_resource_bytes=0,
            )

        for fragment in fragments:
            self._validate_fragment(fragment)
        fragments = self._ordered_fragments(fragments)
        computation_fragments = self._computation_fragments(fragments)
        self._validate_coverage_compatibility(computation_fragments)
        self._validate_resource_compatibility(fragments)
        self._validate_submission_compatibility(fragments)
        assembly_protocols = {
            fragment.assembly_protocol for fragment in fragments
        }
        assembly_providers = {
            fragment.assembly_provider_namespace for fragment in fragments
        }
        if len(assembly_protocols) != 1 or len(assembly_providers) != 1:
            raise GraphRecipeCompositionError(
                "fragments require one shared assembly protocol and provider"
            )
        assembly_protocol = next(iter(assembly_protocols))
        assembly_provider_namespace = next(iter(assembly_providers))

        owner_by_region = {
            region_id: fragment.fragment_id
            for fragment in computation_fragments
            for region_id in fragment.coverage_region_ids
        }
        physical_owner_by_region = {
            region_id: fragment.planned_physical_id
            for fragment in computation_fragments
            for region_id in fragment.coverage_region_ids
        }
        baseline_coverage = tuple(
            region.region_id
            for region in self.definition.regions
            if region.region_id not in owner_by_region
        )
        if assembly_protocol == PROVIDER_OWNED_WHOLE_GRAPH_V1 and (
            len(fragments) != 1 or baseline_coverage
        ):
            raise GraphRecipeCompositionError(
                "provider-owned whole-Graph assembly requires one "
                "exact-coverage fragment"
            )
        selections = tuple(
            GraphRegionSelection(
                region.region_id,
                "fragment" if region.region_id in owner_by_region else "baseline",
                owner_by_region.get(region.region_id, ""),
            )
            for region in self.definition.regions
        )
        execution_steps = self._execution_steps(computation_fragments)
        recipe_identity_payload = {
            "schema": _RECIPE_SCHEMA,
            "semantic_graph_id": self.definition.semantic_graph_id,
            "fragment_ids": tuple(fragment.fragment_id for fragment in fragments),
            "baseline_coverage_region_ids": baseline_coverage,
            "assembly_protocol": assembly_protocol,
            "assembly_provider_namespace": assembly_provider_namespace,
        }
        recipe_id = f"graph-recipe:{_digest(recipe_identity_payload)}"
        planned_payload = {
            "schema": _PLANNED_RECIPE_SCHEMA,
            "semantic_graph_id": self.definition.semantic_graph_id,
            "baseline_planned_physical_id": (
                self.definition.baseline_recipe.planned_physical_id
            ),
            "fragment_planned_physical_ids": tuple(
                fragment.planned_physical_id for fragment in fragments
            ),
            "assembly_protocol": assembly_protocol,
            "assembly_provider_namespace": assembly_provider_namespace,
            "region_selections": tuple(
                {
                    "region_id": region.region_id,
                    "source": (
                        physical_owner_by_region.get(region.region_id, "baseline")
                    ),
                }
                for region in self.definition.regions
            ),
            "execution_steps": tuple(
                {
                    "step_index": step.step_index,
                    "source": step.source,
                    "region_ids": step.region_ids,
                    "depends_on": step.depends_on,
                    "fragment_planned_physical_id": (
                        ""
                        if not step.fragment_id
                        else next(
                            fragment.planned_physical_id
                            for fragment in fragments
                            if fragment.fragment_id == step.fragment_id
                        )
                    ),
                }
                for step in execution_steps
            ),
        }
        planned_physical_id = f"planned-physical:{_digest(planned_payload)}"
        resources_by_name = {}
        for fragment in fragments:
            for resource in fragment.resources:
                resources_by_name.setdefault(resource.name, resource)
        resources = tuple(resources_by_name.values())
        persistent_bytes = sum(
            resource.bytes
            for resource in resources
            if resource.lifetime in ("graph", "session")
        )
        transient_bytes = sum(
            resource.bytes
            for resource in resources
            if resource.lifetime in ("task", "submission")
        ) + sum(
            temporary.bytes
            for fragment in fragments
            for task in fragment.tasks
            for temporary in task.temporaries
        )
        return GraphExecutableRecipe(
            recipe_id=recipe_id,
            semantic_graph_id=self.definition.semantic_graph_id,
            planned_physical_id=planned_physical_id,
            assembly_protocol=assembly_protocol,
            assembly_provider_namespace=assembly_provider_namespace,
            fragments=fragments,
            baseline_coverage_region_ids=baseline_coverage,
            region_selections=selections,
            execution_steps=execution_steps,
            queues=tuple(sorted({fragment.submission.queue for fragment in fragments})),
            barrier_count=sum(
                int(fragment.submission.barrier_before)
                + int(fragment.submission.barrier_after)
                for fragment in fragments
            ),
            exclusive_submission=any(
                fragment.submission.exclusive_submission
                or any(resource.exclusive_submission for resource in fragment.resources)
                for fragment in fragments
            ),
            declared_persistent_resource_bytes=persistent_bytes,
            declared_transient_resource_bytes=transient_bytes,
        )
