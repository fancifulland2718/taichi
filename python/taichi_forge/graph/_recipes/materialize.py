"""Transactional materialization for complete Forge Graph recipes."""

import threading
from uuid import uuid4
from contextlib import suppress
from dataclasses import dataclass, field

from taichi_forge.graph._recipes.composer import GraphExecutableRecipe
from taichi_forge.graph._recipes.fragments import (
    GraphFragmentResourceRequirement,
    GraphRecipeFragment,
)
from taichi_forge.graph._recipes.physical import (
    CompiledGraphPhysicalManifest,
    GraphPhysicalResourceManifest,
    observe_baseline_physical_manifest,
)
from taichi_forge.graph._recipes.providers import GraphRecipeProviderSet

_MAX_SIGNED_BYTES = (1 << 63) - 1


class GraphMaterializationError(RuntimeError):
    """One candidate did not publish a complete, clean materialization."""

    def __init__(
        self,
        message,
        *,
        recipe_id="",
        phase="",
        cleanup_complete=True,
        rollback_errors=(),
    ):
        super().__init__(message)
        self.recipe_id = recipe_id
        self.phase = phase
        self.cleanup_complete = bool(cleanup_complete)
        self.rollback_errors = tuple(rollback_errors)


def _checked_add(total, value, role):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphMaterializationError(f"{role} must be a non-negative integer")
    result = total + value
    if result > _MAX_SIGNED_BYTES:
        raise GraphMaterializationError(f"{role} exceeds signed 64-bit range")
    return result


@dataclass(frozen=True)
class GraphMaterializedAllocation:
    """One allocator result registered with the current transaction."""

    value: object = field(compare=False, hash=False, repr=False)
    physical_resource: GraphPhysicalResourceManifest
    release: object = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.physical_resource, GraphPhysicalResourceManifest):
            raise TypeError(
                "Graph materialized allocation requires a physical resource manifest"
            )
        if self.release is not None and not callable(self.release):
            raise TypeError("Graph materialized allocation release must be callable")


@dataclass(frozen=True)
class GraphMaterializedFragment:
    """Opaque provider result consumed only by the whole-Graph assembler."""

    fragment_id: str
    payload: object = field(compare=False, hash=False, repr=False)

    @classmethod
    def create(cls, fragment, payload):
        if not isinstance(fragment, GraphRecipeFragment):
            raise TypeError(
                "Graph fragment materialization requires GraphRecipeFragment"
            )
        return cls(fragment_id=fragment.fragment_id, payload=payload)


@dataclass(frozen=True)
class GraphMaterializationProduct:
    """Whole-Graph executor plus its normalized physical observation."""

    executor: object = field(compare=False, hash=False, repr=False)
    manifest: CompiledGraphPhysicalManifest
    release: object = field(default=None, compare=False, hash=False, repr=False)
    # Equal execution plans do not imply interchangeable mutable resources.
    # Only providers that explicitly guarantee shared-executor semantics may
    # opt into cross-recipe instance reuse within one materialization context.
    shareable_executor: bool = False

    def __post_init__(self):
        if self.executor is None:
            raise ValueError("Graph materialization product requires an executor")
        if not isinstance(self.manifest, CompiledGraphPhysicalManifest):
            raise TypeError(
                "Graph materialization product requires CompiledGraphPhysicalManifest"
            )
        if self.release is not None and not callable(self.release):
            raise TypeError("Graph materialization product release must be callable")
        if not isinstance(self.shareable_executor, bool):
            raise TypeError("Graph executor shareability must be a boolean")


class _OwnedValue:
    __slots__ = ("label", "physical_resource", "release", "released", "value")

    def __init__(self, value, release, label, physical_resource):
        self.value = value
        self.release = release
        self.label = label
        self.physical_resource = physical_resource
        self.released = False

    def retire(self):
        if self.released:
            return
        value = self.value
        self.value = None
        self.released = True
        if self.release is not None:
            self.release(value)


class _MaterializationTransaction:
    __slots__ = ("_owners", "state")

    def __init__(self):
        self._owners = []
        self.state = "open"

    @property
    def owners(self):
        return tuple(self._owners)

    def own(self, owner):
        if self.state != "open":
            raise GraphMaterializationError(
                "materialization transaction is no longer open",
                phase="ownership",
            )
        self._owners.append(owner)

    def rollback(self):
        if self.state != "open":
            return ()
        self.state = "rolled_back"
        errors = []
        for owner in reversed(self._owners):
            try:
                owner.retire()
            # Continue retiring later owners even if one provider cleanup is
            # broken; the caller receives every failure and poisons context.
            except BaseException as error:  # noqa: BLE001
                errors.append((owner.label, error))
        self._owners.clear()
        return tuple(errors)

    def publish(self):
        if self.state != "open":
            raise GraphMaterializationError(
                "materialization transaction cannot publish twice",
                phase="publish",
            )
        self.state = "published"
        owners = tuple(self._owners)
        self._owners.clear()
        return owners


class GraphMaterializationScope:
    """Restricted provider view of one candidate build transaction."""

    __slots__ = ("_context", "_transaction", "recipe")

    def __init__(self, context, transaction, recipe):
        self._context = context
        self._transaction = transaction
        self.recipe = recipe

    @property
    def definition(self):
        return self._context.definition

    @property
    def backend(self):
        return self.definition.backend

    @property
    def available_capabilities(self):
        return self._context.available_capabilities

    @property
    def physical_resources(self):
        return tuple(
            owner.physical_resource
            for owner in self._transaction.owners
            if owner.physical_resource is not None
        )

    def own(self, value, *, release=None, label="materialized resource", physical=None):
        """Register an owner before it can escape the current transaction."""

        if release is not None and not callable(release):
            raise TypeError("Graph materialization release must be callable")
        if not isinstance(label, str) or not label:
            raise ValueError("Graph materialization owner label must be non-empty")
        if physical is not None and not isinstance(
            physical,
            GraphPhysicalResourceManifest,
        ):
            raise TypeError(
                "Graph materialization owner physical value must be a resource manifest"
            )
        self._transaction.own(_OwnedValue(value, release, label, physical))
        return value

    def own_executor(self, executor, *, release=None):
        """Enroll the executor before observing it; retire a Forge Graph explicitly.

        Other executor types retain their provider-supplied release policy.
        Re-enrolling the same executor during publication must not close it twice.
        """
        from taichi_forge.graph._graph import Graph

        for owner in self._transaction.owners:
            if owner.value is executor:
                if release is not None:
                    if owner.release is not None and owner.release != release:
                        raise ValueError("Graph executor has conflicting release owners")
                    owner.release = release
                if owner.release is None and isinstance(executor, Graph):
                    owner.release = Graph.close
                return executor
        if release is None and isinstance(executor, Graph):
            release = Graph.close
        return self.own(executor, release=release, label="whole Graph executor")

    def allocate(self, requirement):
        """Allocate and immediately enroll one fragment resource in rollback."""

        if not isinstance(requirement, GraphFragmentResourceRequirement):
            raise TypeError(
                "Graph materialization allocation requires a fragment resource"
            )
        if requirement.ownership == "external":
            raise GraphMaterializationError(
                "external fragment resources must be supplied through the binding ABI",
                recipe_id=self.recipe.recipe_id,
                phase="allocate",
            )
        allocator = self._context._resource_allocator
        if allocator is None:
            raise GraphMaterializationError(
                "Graph materialization context has no resource allocator",
                recipe_id=self.recipe.recipe_id,
                phase="allocate",
            )
        allocation = allocator(requirement)
        if not isinstance(allocation, GraphMaterializedAllocation):
            raise TypeError(
                "Graph resource allocator must return GraphMaterializedAllocation"
            )
        physical = allocation.physical_resource
        value = self.own(
            allocation.value,
            release=allocation.release,
            label=f"fragment resource {requirement.name}",
            physical=physical,
        )
        mismatched = bool(
            physical.resource_id != requirement.name
            or physical.kind != requirement.kind
            or physical.requested_bytes != requirement.bytes
            or physical.alignment < requirement.alignment
            or physical.ownership != requirement.ownership
            or physical.lifetime != requirement.lifetime
            or physical.exclusive_submission != requirement.exclusive_submission
            or physical.scope != "internal"
        )
        if mismatched:
            raise GraphMaterializationError(
                "Graph resource allocator violated the fragment requirement",
                recipe_id=self.recipe.recipe_id,
                phase="allocate",
            )
        return value


class _PublishedMaterialization:
    __slots__ = (
        "executor",
        "resource_instance_id",
        "handle_count",
        "manifest",
        "owners",
        "recipe_ids",
        "released",
        "representative_recipe_id",
    )

    def __init__(self, recipe_id, product, owners):
        self.executor = product.executor
        self.resource_instance_id = f"materialization-instance:{uuid4().hex}"
        self.manifest = product.manifest
        self.owners = owners
        self.representative_recipe_id = recipe_id
        self.recipe_ids = {recipe_id}
        self.handle_count = 0
        self.released = False

    def retire(self):
        if self.released:
            return ()
        self.released = True
        errors = []
        for owner in reversed(self.owners):
            try:
                owner.retire()
            # Published owners follow the same complete-retirement contract.
            except BaseException as error:  # noqa: BLE001
                errors.append((owner.label, error))
        self.owners = ()
        self.executor = None
        return tuple(errors)


class GraphMaterializedRecipe:
    """A live handle to one atomically published materialization."""

    __slots__ = (
        "_cache_hit",
        "_close_context_on_release",
        "_closed",
        "_context",
        "_deduplicated",
        "_state",
        "recipe_id",
    )

    def __init__(
        self,
        context,
        recipe_id,
        state,
        *,
        cache_hit,
        deduplicated,
    ):
        self._context = context
        self.recipe_id = recipe_id
        self._state = state
        self._cache_hit = bool(cache_hit)
        self._deduplicated = bool(deduplicated)
        self._closed = False
        self._close_context_on_release = False

    def _require_live(self):
        if self._closed or self._state.released:
            raise GraphMaterializationError(
                "materialized Graph recipe handle is closed",
                recipe_id=self.recipe_id,
                phase="lifecycle",
            )

    @property
    def executor(self):
        self._require_live()
        return self._state.executor

    @property
    def manifest(self):
        self._require_live()
        return self._state.manifest

    @property
    def representative_recipe_id(self):
        return self._state.representative_recipe_id

    @property
    def materialized_physical_id(self):
        return self._state.manifest.materialized_physical_id

    @property
    def resource_instance_id(self):
        """Process-local resource ownership, never a cross-process reuse key."""
        return self._state.resource_instance_id

    @property
    def deduplicated(self):
        return self._deduplicated

    @property
    def cache_hit(self):
        return self._cache_hit

    def materialization_report(self):
        return {**self.manifest.to_dict(), "resource_instance_id": self.resource_instance_id}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._context._release_handle(self.recipe_id, self._state)
        if self._close_context_on_release:
            self._context.close()

    def __enter__(self):
        self._require_live()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        with suppress(GraphMaterializationError):
            self.close()


def _default_runtime_identity():
    from taichi_forge.lang import impl

    runtime = impl.get_runtime()
    return (
        int(impl.runtime_generation()),
        str(impl.current_cfg().arch),
        id(runtime.prog),
    )


def _default_baseline_materializer(scope, definition, recipe):
    graph = definition.compile(
        workspace_lanes=scope._context.workspace_lanes,
        workspace_saturation=scope._context.workspace_saturation,
    )
    scope.own_executor(graph)
    manifest = observe_baseline_physical_manifest(definition, recipe, graph)
    return GraphMaterializationProduct(graph, manifest)


def _validate_materialized_fragment(result, scope, fragment):
    if not isinstance(result, GraphMaterializedFragment):
        raise TypeError(
            "Graph fragment materializer must return GraphMaterializedFragment"
        )
    if result.fragment_id != fragment.fragment_id:
        raise GraphMaterializationError(
            "Graph fragment materializer returned a different fragment identity",
            recipe_id=scope.recipe.recipe_id,
            phase="fragment",
        )
    return result


class GraphMaterializationContext:
    """Explicit, concurrent-safe owner for recipe build transactions.

    Stable legality and layout facts are checked here, once. Runtime resource
    generation, submission leases, replay slots, and retirement remain owned by
    the resulting Graph executor and are deliberately absent from this cache.
    """

    def __init__(
        self,
        definition,
        *,
        available_capabilities=(),
        workspace_lanes=1,
        workspace_saturation="wait",
        baseline_materializer=None,
        assembler=None,
        provider_set=None,
        resource_allocator=None,
        runtime_identity_provider=None,
    ):
        if isinstance(workspace_lanes, bool) or not isinstance(workspace_lanes, int):
            raise TypeError("Graph materialization workspace_lanes must be an integer")
        if workspace_lanes <= 0:
            raise ValueError("Graph materialization workspace_lanes must be positive")
        if workspace_saturation not in ("wait", "raise"):
            raise ValueError(
                "Graph materialization workspace_saturation must be wait or raise"
            )

        if baseline_materializer is not None and not callable(baseline_materializer):
            raise TypeError("Graph baseline materializer must be callable")
        if assembler is not None and not callable(assembler):
            raise TypeError("Graph recipe assembler must be callable")
        if provider_set is not None and not isinstance(
            provider_set,
            GraphRecipeProviderSet,
        ):
            raise TypeError(
                "Graph materialization provider_set must be GraphRecipeProviderSet"
            )
        if provider_set is not None and provider_set.definition is not definition:
            raise ValueError(
                "Graph materialization provider set belongs to another definition"
            )
        if resource_allocator is not None and not callable(resource_allocator):
            raise TypeError("Graph resource allocator must be callable")
        if runtime_identity_provider is not None and not callable(
            runtime_identity_provider
        ):
            raise TypeError("Graph runtime identity provider must be callable")
        self.definition = definition
        self.available_capabilities = frozenset(available_capabilities)
        self.workspace_lanes = workspace_lanes
        self.workspace_saturation = workspace_saturation
        self._baseline_materializer = (
            baseline_materializer or _default_baseline_materializer
        )
        self._assembler = assembler
        self.provider_set = provider_set
        self._resource_allocator = resource_allocator
        self._runtime_identity_provider = (
            runtime_identity_provider or _default_runtime_identity
        )
        self._runtime_identity = self._runtime_identity_provider()
        self._lock = threading.RLock()
        self._state = "open"
        self._active_transactions = 0
        self._recipe_states = {}
        self._recipe_handle_counts = {}
        self._physical_states = {}
        self._publication_order = []
        self._statistics = {
            "attempts": 0,
            "publications": 0,
            "failures": 0,
            "recipe_cache_hits": 0,
            "materialized_physical_deduplications": 0,
            "rollbacks": 0,
            "rollback_failures": 0,
            "releases": 0,
        }
        if runtime_identity_provider is None:
            from taichi_forge.lang import impl

            impl.get_runtime().register_runtime_object(self)

    def _require_open_locked(self):
        if self._state == "poisoned":
            raise GraphMaterializationError(
                "Graph materialization context is poisoned after incomplete cleanup",
                phase="context",
                cleanup_complete=False,
            )
        if self._state != "open":
            raise GraphMaterializationError(
                "Graph materialization context is closed",
                phase="context",
            )

    def _validate_runtime_locked(self):
        if self._runtime_identity_provider() != self._runtime_identity:
            raise GraphMaterializationError(
                "Graph materialization runtime changed; create a new context",
                phase="runtime_identity",
            )

    def _normalize_recipe(self, recipe):
        from taichi_forge.graph._recipes.composer import GraphRecipeComposer

        composer = GraphRecipeComposer(
            self.definition,
            available_capabilities=self.available_capabilities,
        )
        if recipe is None or recipe is self.definition.baseline_recipe:
            recipe = composer.compose()
        if not isinstance(recipe, GraphExecutableRecipe):
            raise TypeError(
                "Graph materialization requires a complete GraphExecutableRecipe"
            )
        if recipe.semantic_graph_id != self.definition.semantic_graph_id:
            raise GraphMaterializationError(
                "Graph recipe belongs to a different semantic definition",
                recipe_id=recipe.recipe_id,
                phase="admission",
            )
        canonical = composer.compose(recipe.fragments)
        if canonical != recipe:
            raise GraphMaterializationError(
                "Graph recipe is not the canonical composition of its fragments",
                recipe_id=recipe.recipe_id,
                phase="admission",
            )
        return recipe

    @staticmethod
    def _validate_declared_sizes(recipe):
        persistent = 0
        transient = 0
        resources = {}
        for fragment in recipe.fragments:
            for resource in fragment.resources:
                resources.setdefault(resource.name, resource)
            for task in fragment.tasks:
                for temporary in task.temporaries:
                    transient = _checked_add(
                        transient,
                        temporary.bytes,
                        "Graph temporary bytes",
                    )
        for resource in resources.values():
            if resource.lifetime in ("graph", "session"):
                persistent = _checked_add(
                    persistent,
                    resource.bytes,
                    "Graph persistent resource bytes",
                )
            else:
                transient = _checked_add(
                    transient,
                    resource.bytes,
                    "Graph transient resource bytes",
                )
        if persistent != recipe.declared_persistent_resource_bytes:
            raise GraphMaterializationError(
                "Graph recipe persistent resource total is inconsistent",
                recipe_id=recipe.recipe_id,
                phase="admission",
            )
        if transient != recipe.declared_transient_resource_bytes:
            raise GraphMaterializationError(
                "Graph recipe transient resource total is inconsistent",
                recipe_id=recipe.recipe_id,
                phase="admission",
            )

    def _make_handle_locked(
        self,
        recipe_id,
        state,
        *,
        cache_hit,
        deduplicated,
    ):
        self._recipe_handle_counts[recipe_id] = (
            self._recipe_handle_counts.get(recipe_id, 0) + 1
        )
        state.handle_count += 1
        return GraphMaterializedRecipe(
            self,
            recipe_id,
            state,
            cache_hit=cache_hit,
            deduplicated=deduplicated,
        )

    def _rollback_locked(self, transaction):
        errors = transaction.rollback()
        self._statistics["rollbacks"] += 1
        if errors:
            self._statistics["rollback_failures"] += 1
            self._state = "poisoned"
        return errors

    def _publish_locked(self, recipe, product, transaction):
        existing = self._recipe_states.get(recipe.recipe_id)
        if existing is not None:
            errors = self._rollback_locked(transaction)
            if errors:
                raise GraphMaterializationError(
                    "duplicate recipe cleanup failed",
                    recipe_id=recipe.recipe_id,
                    phase="rollback",
                    cleanup_complete=False,
                    rollback_errors=errors,
                )
            self._statistics["recipe_cache_hits"] += 1
            return self._make_handle_locked(
                recipe.recipe_id,
                existing,
                cache_hit=True,
                deduplicated=(existing.representative_recipe_id != recipe.recipe_id),
            )

        manifest = product.manifest
        duplicate = (
            self._physical_states.get(manifest.materialized_physical_id)
            if manifest.identity_complete and product.shareable_executor
            else None
        )
        if duplicate is not None:
            errors = self._rollback_locked(transaction)
            if errors:
                raise GraphMaterializationError(
                    "physical duplicate cleanup failed",
                    recipe_id=recipe.recipe_id,
                    phase="rollback",
                    cleanup_complete=False,
                    rollback_errors=errors,
                )
            duplicate.recipe_ids.add(recipe.recipe_id)
            self._recipe_states[recipe.recipe_id] = duplicate
            self._statistics["materialized_physical_deduplications"] += 1
            return self._make_handle_locked(
                recipe.recipe_id,
                duplicate,
                cache_hit=False,
                deduplicated=True,
            )

        owners = transaction.publish()
        state = _PublishedMaterialization(recipe.recipe_id, product, owners)
        self._recipe_states[recipe.recipe_id] = state
        if manifest.identity_complete and product.shareable_executor:
            self._physical_states[manifest.materialized_physical_id] = state
        self._publication_order.append(state)
        self._statistics["publications"] += 1
        return self._make_handle_locked(
            recipe.recipe_id,
            state,
            cache_hit=False,
            deduplicated=False,
        )

    def materialize(self, recipe=None):
        """Build, validate, and atomically publish one complete recipe."""

        recipe = self._normalize_recipe(recipe)
        self._validate_declared_sizes(recipe)
        with self._lock:
            self._require_open_locked()
            self._validate_runtime_locked()
            cached = self._recipe_states.get(recipe.recipe_id)
            if cached is not None:
                self._statistics["recipe_cache_hits"] += 1
                return self._make_handle_locked(
                    recipe.recipe_id,
                    cached,
                    cache_hit=True,
                    deduplicated=(cached.representative_recipe_id != recipe.recipe_id),
                )
            self._active_transactions += 1
            self._statistics["attempts"] += 1

        transaction = _MaterializationTransaction()
        scope = GraphMaterializationScope(self, transaction, recipe)
        phase = "baseline" if not recipe.fragments else "fragment"
        try:
            if recipe.fragments:
                if self.provider_set is None:
                    raise GraphMaterializationError(
                        "Graph fragment materialization requires a provider set",
                        recipe_id=recipe.recipe_id,
                        phase="fragment",
                    )
                materialized_fragments = []
                for fragment in recipe.fragments:
                    materialized_fragments.append(
                        _validate_materialized_fragment(
                            self.provider_set.materialize(scope, fragment),
                            scope,
                            fragment,
                        )
                    )
                phase = "assemble"
                if self._assembler is not None:
                    product = self._assembler(
                        scope,
                        self.definition,
                        recipe,
                        tuple(materialized_fragments),
                    )
                else:
                    product = self.provider_set.assemble(
                        scope,
                        recipe,
                        tuple(materialized_fragments),
                    )
            else:
                product = self._baseline_materializer(
                    scope,
                    self.definition,
                    recipe,
                )
            if not isinstance(product, GraphMaterializationProduct):
                raise TypeError(
                    "Graph whole-recipe materializer must return "
                    "GraphMaterializationProduct"
                )
            scope.own_executor(product.executor, release=product.release)
            manifest = product.manifest
            if (
                manifest.semantic_graph_id != self.definition.semantic_graph_id
                or manifest.recipe_id != recipe.recipe_id
                or manifest.planned_physical_id != recipe.planned_physical_id
                or manifest.backend != self.definition.backend
            ):
                raise GraphMaterializationError(
                    "Graph materializer returned a manifest for a different recipe",
                    recipe_id=recipe.recipe_id,
                    phase="observe",
                )
            missing_owned_resources = tuple(
                resource
                for resource in scope.physical_resources
                if resource not in manifest.resources
            )
            if missing_owned_resources:
                raise GraphMaterializationError(
                    "Graph physical manifest omits transaction-owned resources",
                    recipe_id=recipe.recipe_id,
                    phase="observe",
                )
            phase = "publish"
            with self._lock:
                self._require_open_locked()
                self._validate_runtime_locked()
                return self._publish_locked(recipe, product, transaction)
        except BaseException as error:
            with self._lock:
                rollback_errors = self._rollback_locked(transaction)
                self._statistics["failures"] += 1
            if isinstance(error, GraphMaterializationError) and not rollback_errors:
                raise
            message = str(error).strip() or type(error).__name__
            if rollback_errors:
                message += "; rollback did not complete"
            raise GraphMaterializationError(
                message,
                recipe_id=recipe.recipe_id,
                phase=phase,
                cleanup_complete=not rollback_errors,
                rollback_errors=rollback_errors,
            ) from error
        finally:
            with self._lock:
                self._active_transactions -= 1

    def _release_handle(self, recipe_id, state):
        with self._lock:
            count = self._recipe_handle_counts.get(recipe_id, 0)
            if count <= 0:
                return
            state.handle_count -= 1
            if count > 1:
                self._recipe_handle_counts[recipe_id] = count - 1
                return
            self._recipe_handle_counts.pop(recipe_id, None)
            self._recipe_states.pop(recipe_id, None)
            state.recipe_ids.discard(recipe_id)
            if state.recipe_ids or state.handle_count:
                return
            errors = state.retire()
            self._statistics["releases"] += 1
            physical_id = state.manifest.materialized_physical_id
            if self._physical_states.get(physical_id) is state:
                self._physical_states.pop(physical_id)
            if errors:
                self._state = "poisoned"
                self._statistics["rollback_failures"] += 1
                raise GraphMaterializationError(
                    "published Graph materialization did not retire cleanly",
                    recipe_id=recipe_id,
                    phase="retire",
                    cleanup_complete=False,
                    rollback_errors=errors,
                )

    def statistics(self):
        with self._lock:
            result = dict(self._statistics)
            states = {id(state): state for state in self._recipe_states.values()}
            result.update(
                {
                    "state": self._state,
                    "active_transactions": self._active_transactions,
                    "live_recipe_ids": len(self._recipe_states),
                    "live_physical_materializations": len(states),
                    "live_handles": sum(self._recipe_handle_counts.values()),
                    "live_owned_resources": sum(
                        len(state.owners)
                        for state in states.values()
                        if not state.released
                    ),
                }
            )
            return result

    def close(self):
        with self._lock:
            if self._state == "closed":
                return
            if self._active_transactions:
                raise GraphMaterializationError(
                    "cannot close a context with active materialization transactions",
                    phase="context",
                )
            states = []
            seen = set()
            for state in reversed(self._publication_order):
                identity = id(state)
                if identity not in seen and not state.released:
                    seen.add(identity)
                    states.append(state)
            self._state = "closed"
            self._recipe_states.clear()
            self._recipe_handle_counts.clear()
            self._physical_states.clear()
            errors = []
            for state in states:
                errors.extend(state.retire())
                self._statistics["releases"] += 1
            if errors:
                raise GraphMaterializationError(
                    "Graph materialization context did not close cleanly",
                    phase="retire",
                    cleanup_complete=False,
                    rollback_errors=tuple(errors),
                )

    def _invalidate_runtime(self):
        self.close()

    def __enter__(self):
        with self._lock:
            self._require_open_locked()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = [
    "GraphMaterializationContext",
    "GraphMaterializationError",
    "GraphMaterializationProduct",
    "GraphMaterializationScope",
    "GraphMaterializedAllocation",
    "GraphMaterializedFragment",
    "GraphMaterializedRecipe",
]
