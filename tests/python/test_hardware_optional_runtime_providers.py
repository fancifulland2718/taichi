from contextlib import nullcontext
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

import taichi_forge as ti
from taichi_forge.hardware import _amgx, _cusparselt, _cutensor
from taichi_forge.hardware import _bundled_runtime_provider as _runtime_provider
from taichi_forge.hardware._external_cuda_submission import external_cuda_submission


_MODULES = (_cusparselt, _cutensor, _amgx)


class _FakeProgram:
    def __init__(self):
        self.synchronizations = 0
        self.submissions = []

    def synchronize(self):
        self.synchronizations += 1

    def _begin_external_cuda_submission(self):
        owner = self

        class Scope:
            def _commit(self, arrays, failed=False):
                owner.submissions.append((tuple(arrays), bool(failed)))

        return Scope()


class _FakeRuntime:
    def __init__(self, execution_api):
        self.handle = ctypes.c_void_p(101)
        self.runtime_info = {
            "version_major": 2,
            "version_minor": 7,
            "version_patch": 0,
            "library_path": "fake-vendor-runtime",
            "build_version": "test-build",
        }
        self.execution_api = execution_api
        self.closed = False

    def query_execution_api(self, _api_type):
        return self.execution_api

    @staticmethod
    def check_result(result):
        assert result == 0

    def close(self):
        self.closed = True


def test_external_cuda_submission_commits_success_failure_and_no_call():
    program = _FakeProgram()
    first = SimpleNamespace(arr="first")
    second = SimpleNamespace(arr="second")

    with external_cuda_submission(program, (first, second)) as submission:
        assert submission.invoke(lambda value: value + 1, 3) == 4
    with pytest.raises(RuntimeError, match="provider failure"):
        with external_cuda_submission(program, (first,)) as submission:
            submission.invoke(
                lambda: (_ for _ in ()).throw(RuntimeError("provider failure"))
            )
    with external_cuda_submission(program, (second,)):
        pass

    assert program.submissions == [
        (("first", "second"), False),
        (("first",), True),
    ]


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_library_path_is_explicit_and_environment_owned(
    module, tmp_path, monkeypatch
):
    name = module.DEFINITION.library_names[0]
    runtime = tmp_path / name
    runtime.write_bytes(b"test-vendor-runtime")

    assert module.resolve_library_path(runtime) == str(runtime.resolve())
    assert module.resolve_library_path(tmp_path) == str(runtime.resolve())

    monkeypatch.setenv(module.DEFINITION.environment_variable, str(runtime))
    assert module.resolve_library_path() == str(runtime.resolve())

    missing = tmp_path / f"missing-{name}"
    monkeypatch.setenv(module.DEFINITION.environment_variable, str(missing))
    assert module.resolve_library_path() == str(missing)


@pytest.mark.parametrize("module", (_cusparselt, _cutensor))
def test_optional_runtime_discovers_installed_vendor_package_files(
    module, tmp_path, monkeypatch
):
    relative = Path("vendor") / "lib" / module.DEFINITION.library_names[0]
    runtime = tmp_path / relative
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"test-package-runtime")
    distribution = SimpleNamespace(
        files=(relative,), locate_file=lambda item: tmp_path / item
    )

    def find_distribution(name):
        if name == module.DEFINITION.package_distributions[0]:
            return distribution
        raise _runtime_provider.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(
        _runtime_provider.importlib.metadata, "distribution", find_distribution
    )

    assert module.resolve_library_path() == str(runtime.resolve())


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_probe_audits_without_qualifying_execution(
    module, monkeypatch
):
    definition = module.DEFINITION
    info = _runtime_provider._ProviderInfo()  # pylint: disable=W0212
    info.provider_id = definition.provider_id.encode()
    info.provider_name = definition.provider_name.encode()
    info.supported_version_family = definition.supported_version_family.encode()
    info.build_identity = b"test-forge-adapter"
    info.features = (  # pylint: disable=W0212
        _runtime_provider._REQUIRED_FEATURES | _runtime_provider._FEATURE_EXECUTION_API
    )
    info.required_symbol_count = 17
    loaded = SimpleNamespace(path="test-forge-adapter", api=SimpleNamespace(info=info))

    monkeypatch.setattr(
        _runtime_provider,
        "_bundled_provider_candidates",
        lambda _definition: ("test-forge-adapter",),
    )
    monkeypatch.setattr(
        _runtime_provider, "_query_provider", lambda _definition, _path: loaded
    )
    monkeypatch.setattr(
        _runtime_provider,
        "_probe_runtime",
        lambda _definition, _loaded, path: {
            "version_major": 2,
            "version_minor": 5,
            "version_patch": 0,
            "cuda_runtime_version": 12080,
            "library_path": path or "system-default",
            "build_version": "test-build",
        },
    )
    monkeypatch.setattr(_runtime_provider, "_binary_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(
        _runtime_provider,
        "_vendor_dll_directories",
        lambda _definition, _path: nullcontext(),
    )

    result = module.probe_provider("vendor-runtime")

    assert result["provider_id"] == definition.provider_id
    assert result["discovery"] == "available"
    assert result["provider_version"] == "2.5.0"
    assert result["native_facts"]["required_symbol_count"] == 17
    assert result["native_facts"]["library_loaded_transiently"] is True
    assert result["native_facts"]["execution_api_available"] is True
    assert result["native_facts"]["execution_qualified"] is False
    assert result["native_facts"]["execution_resource_created"] is False


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_probe_fails_closed_when_vendor_runtime_is_missing(
    module, monkeypatch
):
    monkeypatch.setattr(
        _runtime_provider,
        "_bundled_provider_candidates",
        lambda _definition: ("test-forge-adapter",),
    )
    monkeypatch.setattr(
        _runtime_provider,
        "_query_provider",
        lambda _definition, path: SimpleNamespace(path=path),
    )

    def missing_runtime(_definition, _loaded, _path):
        raise _runtime_provider._ProviderRuntimeError(  # pylint: disable=W0212
            _runtime_provider._RUNTIME_UNAVAILABLE,  # pylint: disable=W0212
            "test vendor runtime missing",
        )

    monkeypatch.setattr(_runtime_provider, "_probe_runtime", missing_runtime)
    monkeypatch.setattr(
        _runtime_provider,
        "_vendor_dll_directories",
        lambda _definition, _path: nullcontext(),
    )

    result = module.probe_provider("missing-vendor-runtime")

    assert result["discovery"] == "missing"
    assert result["unavailable_reason"] == "external_library_not_found"
    assert result["failure_scope"] == "provider"
    assert result["native_facts"]["library_loaded_transiently"] is False


@pytest.mark.parametrize("module", _MODULES)
def test_optional_runtime_passive_status_never_loads_a_library(module, monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive status must not load an optional runtime")

    monkeypatch.setattr(_runtime_provider, "_load_library", unexpected_load)
    status = module.passive_status()

    assert status["provider_id"] == module.DEFINITION.provider_id
    assert status["library_loaded"] is False
    assert status["native_facts"]["external_component_probed"] is False
    assert status["native_facts"]["execution_api_available"] is False


def test_optional_runtime_passive_status_observes_retained_runtime_without_loading(
    monkeypatch,
):
    definition = _cutensor.DEFINITION
    loaded = SimpleNamespace(api=SimpleNamespace(destroy_runtime=lambda _handle: 0))
    runtime_info = {
        "version_major": 2,
        "version_minor": 7,
        "version_patch": 0,
        "cuda_runtime_version": 12080,
        "library_path": "retained-cutensor-runtime",
        "build_version": "test-build",
    }
    monkeypatch.setattr(
        _runtime_provider,
        "_load_library",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("passive status must not load a library")
        ),
    )

    runtime = _runtime_provider.BundledRuntime(
        definition,
        loaded,
        ctypes.c_void_p(505),
        runtime_info,
    )
    status = _cutensor.passive_status()

    assert status["library_loaded"] is True
    assert status["provider_version"] == "2.7.0"
    assert status["native_facts"]["execution_api_available"] is True
    assert status["native_facts"]["active_runtime_count"] == 1
    assert status["native_facts"]["loaded_runtime_identities"] == (
        {
            "library_path": "retained-cutensor-runtime",
            "version": "2.7.0",
            "build_version": "test-build",
            "lease_count": 1,
        },
    )

    runtime.close()
    status = _cutensor.passive_status()
    assert status["library_loaded"] is False
    assert status["native_facts"]["active_runtime_count"] == 0


def test_optional_runtime_probe_capabilities_remain_non_executing_observation_calls():
    for provider_id in ("cusparselt", "cutensor", "amgx"):
        descriptor = ti.hardware.capability(f"runtime.probe.{provider_id}")
        assert descriptor.provider_id == provider_id
        assert descriptor.scopes == ("python",)
        assert descriptor.execution_kind == "external_library"
        assert descriptor.graph_integration == "unsupported"
        assert descriptor.hardware_acceleration == "none"
        assert descriptor.hardware_route == "none"
        assert descriptor.implementation_status == "internal_foundation"
        assert "kernel intrinsic" in descriptor.notes[1]


def test_optional_runtime_execution_capabilities_are_explicit_host_plan_apis():
    operation_ids = (
        "tensor.matmul.cusparselt",
        "tensor.contract.cutensor",
        "linalg.solve.amgx",
    )
    for operation_id in operation_ids:
        descriptor = ti.hardware.capability(operation_id)
        is_recordable = operation_id in ("tensor.contract.cutensor", "tensor.matmul.cusparselt")
        assert descriptor.scopes == (
            ("python", "graph") if is_recordable else ("python",)
        )
        assert descriptor.execution_kind == "external_library"
        assert descriptor.graph_integration == (
            "root_ordered" if is_recordable else "unsupported"
        )
        assert descriptor.hardware_acceleration == "implementation_defined"
        assert descriptor.activation_mode == "explicit_hardware_api"
        assert descriptor.lifetime_policy == "provider_plan"
        assert "automatic" in " ".join(descriptor.notes).lower()


@pytest.mark.parametrize("module", _MODULES)
def test_public_hardware_probe_keeps_probe_only_provider_disabled(module, monkeypatch):
    definition = module.DEFINITION
    monkeypatch.setattr(
        module,
        "probe_provider",
        lambda _path=None: {
            "provider_id": definition.provider_id,
            "external_component_probed": True,
            "discovery": "available",
            "unavailable_reason": "none",
            "provider_abi": definition.provider_abi_name,
            "provider_version": "2.5.0",
            "last_error": None,
            "failure_scope": None,
            "native_facts": {
                "execution_qualified": False,
                "execution_api_available": True,
            },
        },
    )

    report = ti.hardware.probe(definition.provider_id)
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == f"runtime.probe.{definition.provider_id}"
    )

    assert report.external_components_probed is True
    assert operation.discovery == "available"
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"
    assert operation.native_facts["execution_qualified"] is False


def test_optional_runtime_provider_filenames_are_platform_specific():
    for module in _MODULES:
        filename = _runtime_provider._adapter_filename(
            module.DEFINITION
        )  # pylint: disable=W0212
        assert Path(filename).suffix == (".dll" if os.name == "nt" else ".so")


def test_cutensor_provider_owns_plan_lifetime_and_native_destroy(monkeypatch):
    calls = []

    def create(_runtime, _desc, handle, info):
        handle._obj.value = 202
        info._obj.workspace_estimate_bytes = 0
        info._obj.workspace_required_bytes = 0
        return 0

    execution = SimpleNamespace(
        execution_abi_version=1,
        create_contraction_plan=create,
        execute_contraction=lambda _plan, _desc: calls.append("execute") or 0,
        destroy_contraction_plan=lambda plan: calls.append(("destroy", plan.value))
        or 0,
    )
    runtime = _FakeRuntime(execution)
    program = _FakeProgram()
    monkeypatch.setattr(_cutensor, "_require_cuda_program", lambda _name: program)
    monkeypatch.setattr(_cutensor, "_open_runtime", lambda _definition, _path: runtime)
    monkeypatch.setattr(_cutensor.impl, "runtime_generation", lambda: 7)
    monkeypatch.setattr(_cutensor, "validate_runtime_generation", lambda *_args: None)
    monkeypatch.setattr(_cutensor, "runtime_generation_matches", lambda _owner: True)
    monkeypatch.setattr(_cutensor, "_validate_array", lambda *_args: None)
    monkeypatch.setattr(_cutensor, "_device_pointer", lambda value: id(value))

    provider = _cutensor.CutensorProvider()
    plan = provider.contraction_plan(
        (2, 3), "ik", (3, 4), "kj", (2, 4), "ij", (2, 4), "ij"
    )
    arrays = tuple(SimpleNamespace(arr=object()) for _ in range(4))
    plan.execute(*arrays)
    with pytest.raises(ti.TaichiRuntimeError, match="plans are live"):
        provider.close()
    plan.close()
    provider.close()

    assert calls == ["execute", ("destroy", 202)]
    assert runtime.closed is True
    assert program.synchronizations == 2
    assert len(program.submissions) == 1
    assert program.submissions[0][1] is False


def test_cusparselt_provider_retains_owned_buffers_until_plan_close(monkeypatch):
    calls = []

    def create(_runtime, _desc, handle, info):
        handle._obj.value = 303
        info._obj.compressed_bytes = 64
        info._obj.compression_buffer_bytes = 32
        info._obj.workspace_bytes = 16
        return 0

    execution = SimpleNamespace(
        execution_abi_version=1,
        create_matmul_plan=create,
        compress_sparse_a=lambda _plan, _desc: calls.append("compress") or 0,
        execute_matmul=lambda _plan, _desc: calls.append("execute") or 0,
        destroy_matmul_plan=lambda plan: calls.append(("destroy", plan.value)) or 0,
    )
    runtime = _FakeRuntime(execution)
    program = _FakeProgram()
    monkeypatch.setattr(_cusparselt, "_require_cuda_program", lambda _name: program)
    monkeypatch.setattr(
        _cusparselt, "_open_runtime", lambda _definition, _path: runtime
    )
    monkeypatch.setattr(_cusparselt.impl, "runtime_generation", lambda: 7)
    monkeypatch.setattr(_cusparselt, "validate_runtime_generation", lambda *_args: None)
    monkeypatch.setattr(_cusparselt, "runtime_generation_matches", lambda _owner: True)
    monkeypatch.setattr(
        _cusparselt,
        "ScalarNdarray",
        lambda _dtype, shape: SimpleNamespace(shape=shape, arr=object()),
    )
    monkeypatch.setattr(_cusparselt, "_validate_array", lambda *_args: None)
    monkeypatch.setattr(_cusparselt, "_device_pointer", lambda value: id(value))

    provider = _cusparselt.CusparseLtProvider()
    plan = provider.matmul_plan(16, 16, 16)
    arrays = tuple(SimpleNamespace(arr=object()) for _ in range(4))
    plan.compress(arrays[0]).execute(*arrays[1:])
    with pytest.raises(ti.TaichiRuntimeError, match="plans are live"):
        provider.close()
    plan.close()
    provider.close()

    assert calls == ["compress", "execute", ("destroy", 303)]
    assert runtime.closed is True
    assert len(program.submissions) == 2
    assert all(not item[1] for item in program.submissions)


def test_amgx_provider_executes_host_buffers_and_blocks_early_close(monkeypatch):
    calls = []

    def create(_runtime, _desc, handle):
        handle._obj.value = 404
        return 0

    def solve(_solver, desc, info):
        solve_desc = desc._obj
        ctypes.memmove(
            solve_desc.solution, solve_desc.rhs, 3 * ctypes.sizeof(ctypes.c_double)
        )
        info._obj.solve_status = 0
        info._obj.iterations = 4
        info._obj.residual_norm = 1e-12
        return 0

    execution = SimpleNamespace(
        execution_abi_version=1,
        create_solver=create,
        replace_coefficients=lambda _solver, _values, _nonzeros: calls.append("replace")
        or 0,
        solve=solve,
        destroy_solver=lambda solver: calls.append(("destroy", solver.value)) or 0,
    )
    runtime = _FakeRuntime(execution)
    program = _FakeProgram()
    monkeypatch.setattr(_amgx, "_require_cuda_program", lambda _name: program)
    monkeypatch.setattr(_amgx, "_open_runtime", lambda _definition, _path: runtime)
    monkeypatch.setattr(_amgx.impl, "runtime_generation", lambda: 7)
    monkeypatch.setattr(_amgx, "validate_runtime_generation", lambda *_args: None)
    monkeypatch.setattr(_amgx, "runtime_generation_matches", lambda _owner: True)

    provider = _amgx.AmgxProvider()
    solver = provider.solver(
        np.array([0, 1, 2, 3], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.ones(3, dtype=np.float64),
        "config_version=2, solver=PCG",
    )
    with pytest.raises(ti.TaichiRuntimeError, match="solver resources are live"):
        provider.close()
    solution, info = solver.solve(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    solver.replace_coefficients(np.full(3, 2.0, dtype=np.float64))
    solver.close()
    provider.close()

    np.testing.assert_array_equal(solution, np.array([1.0, 2.0, 3.0]))
    assert info["converged"] is True
    assert info["iterations"] == 4
    assert calls == ["replace", ("destroy", 404)]
    assert runtime.closed is True
