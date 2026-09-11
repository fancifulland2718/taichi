import ctypes
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _optix
from taichi_forge.lang import impl
from tests import test_utils


_OPTIX_VERSION_BY_ABI = {
    93: 80100,
    105: 90000,
    118: 90100,
}


class _FakeOptixLibrary:
    def __init__(
        self,
        *,
        optix_abi=93,
        query_result=0,
        context_result=0,
        runtime_probe_result=0,
    ):
        self.calls = {
            "query": 0,
            "create_context": 0,
            "destroy_context": 0,
            "create_scene": 0,
            "update_scene": 0,
            "trace": 0,
            "destroy_scene": 0,
            "probe_runtime": [],
            "create_context_runtime_path": [],
        }
        self._last_error = b"fake provider failure"

        @_optix._CreateContext
        def create_context(desc, out_context):
            self.calls["create_context"] += 1
            self.calls["create_context_runtime_path"].append(
                desc.contents.runtime_library_path
            )
            if context_result:
                self._last_error = b"fake OptiX runtime unavailable"
                return context_result
            out_context[0] = ctypes.c_void_p(0x1234)
            return 0

        @_optix._DestroyContext
        def destroy_context(_context):
            self.calls["destroy_context"] += 1
            return 0

        @_optix._CreateScene
        def create_scene(_context, _desc, out_scene):
            self.calls["create_scene"] += 1
            out_scene[0] = ctypes.c_void_p(0x5678)
            return 0

        @_optix._UpdateScene
        def update_scene(_scene, _desc):
            self.calls["update_scene"] += 1
            return 0

        @_optix._Trace
        def trace(_scene, _desc):
            self.calls["trace"] += 1
            return 0

        @_optix._GetSceneMemory
        def get_scene_memory(_scene, out_memory):
            memory = out_memory.contents
            memory.gas_bytes = 4096
            memory.ias_bytes = 2048
            memory.build_update_scratch_bytes = 1024
            memory.instance_bytes = 80
            memory.launch_params_bytes = 24
            memory.shared_pipeline_sbt_bytes = 192
            return 0

        @_optix._DestroyScene
        def destroy_scene(_scene):
            self.calls["destroy_scene"] += 1
            return 0

        @_optix._GetLastError
        def get_last_error(destination, destination_size):
            required = len(self._last_error) + 1
            if destination and destination_size:
                copied = min(len(self._last_error), int(destination_size) - 1)
                ctypes.memmove(destination, self._last_error, copied)
                destination[copied] = b"\0"
            return required

        @_optix._ProbeRuntime
        def probe_runtime(path):
            self.calls["probe_runtime"].append(path)
            if runtime_probe_result:
                self._last_error = b"fake OptiX ABI unsupported"
            return runtime_probe_result

        self._callbacks = (
            probe_runtime,
            create_context,
            destroy_context,
            create_scene,
            update_scene,
            trace,
            get_scene_memory,
            destroy_scene,
            get_last_error,
        )
        api = _optix._ProviderApi()
        api.struct_size = ctypes.sizeof(_optix._ProviderApi)
        api.provider_abi_version = _optix.PROVIDER_ABI_VERSION
        api.info.struct_size = ctypes.sizeof(_optix._ProviderInfo)
        api.info.provider_abi_version = _optix.PROVIDER_ABI_VERSION
        api.info.optix_abi_version = optix_abi
        api.info.optix_version = _OPTIX_VERSION_BY_ABI.get(optix_abi, 0)
        api.info.features = _optix._REQUIRED_FEATURES | (1 << 5)
        api.info.provider_name = b"fake-optix"
        api.info.build_identity = b"fake-provider-abi1"
        api.probe_runtime = probe_runtime
        api.create_context = create_context
        api.destroy_context = destroy_context
        api.create_triangle_scene = create_scene
        api.update_triangle_scene = update_scene
        api.trace = trace
        api.get_scene_memory = get_scene_memory
        api.destroy_triangle_scene = destroy_scene
        api.get_last_error = get_last_error
        self._api = api

        query_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_size_t,
            ctypes.POINTER(_optix._ProviderApi),
        )

        @query_type
        def query(requested_abi, api_size, out_api):
            self.calls["query"] += 1
            if query_result:
                return query_result
            if requested_abi != _optix.PROVIDER_ABI_VERSION:
                return 2
            ctypes.memmove(
                out_api,
                ctypes.byref(self._api),
                min(int(api_size), ctypes.sizeof(self._api)),
            )
            return 0

        self.taichi_forge_optix_provider_query = query
        self._query = query


def test_optix_device_pointer_resolves_ndarray_data_not_allocation_descriptor(
    monkeypatch,
):
    allocation = object()

    class FakeProgram:
        def get_ndarray_data_ptr_as_int(self, value):
            assert value is allocation
            return 0x12345678

    class FakeRuntime:
        prog = FakeProgram()

    class FakeNdarray:
        arr = allocation

    monkeypatch.setattr(_optix.impl, "get_runtime", FakeRuntime)

    assert _optix._device_pointer(FakeNdarray()) == 0x12345678


@pytest.mark.parametrize(
    ("optix_abi", "provider_version"),
    ((93, "8.1.0"), (105, "9.0.0"), (118, "9.1.0")),
)
def test_optix_probe_is_runtime_only_and_abi_versioned(
    monkeypatch, optix_abi, provider_version
):
    fake = _FakeOptixLibrary(optix_abi=optix_abi)
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("fake-adapter.dll",)
    )
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider()

    assert result["discovery"] == "present"
    assert result["unavailable_reason"] == "execution_not_qualified"
    assert result["provider_abi"] == _optix.PROVIDER_ABI_NAME
    assert result["provider_version"] == provider_version
    assert result["native_facts"]["optix_abi_version"] == optix_abi
    assert result["native_facts"]["runtime_probe_only"] is True
    assert fake.calls["query"] == 1
    assert fake.calls["create_context"] == 0


@pytest.mark.parametrize("optix_abi", (92, 106, 117, 119))
def test_optix_probe_rejects_unqualified_sdk_abi_without_initializing(
    monkeypatch, optix_abi
):
    fake = _FakeOptixLibrary(optix_abi=optix_abi)
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("future-adapter.dll",)
    )
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider()

    assert result["discovery"] == "incompatible"
    assert result["unavailable_reason"] == "no_compatible_optix_provider"
    assert "outside Forge's bundled adapter range" in result["last_error"]
    assert result["failure_scope"] == "provider"
    assert fake.calls["create_context"] == 0


def test_optix_probe_handles_query_failure_before_error_callback_exists(monkeypatch):
    fake = _FakeOptixLibrary(query_result=2)
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("abi-mismatch-adapter.dll",)
    )
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)

    result = _optix.probe_provider()

    assert result["discovery"] == "incompatible"
    assert result["unavailable_reason"] == "no_compatible_optix_provider"
    assert "provider query failed with result 2" in result["last_error"]
    assert fake.calls["create_context"] == 0


def test_optix_passive_status_and_missing_probe_do_not_load_a_library(monkeypatch):
    def unexpected_load(_path):
        raise AssertionError("passive OptiX reporting must not load a plugin")

    monkeypatch.delenv("TAICHI_FORGE_OPTIX_LIBRARY", raising=False)
    monkeypatch.setattr(_optix, "_bundled_provider_candidates", lambda: ())
    monkeypatch.setattr(_optix, "_load_library", unexpected_load)

    assert _optix.passive_status()["library_loaded"] is False
    result = _optix.probe_provider()
    assert result["external_component_probed"] is False
    assert result["unavailable_reason"] == "bundled_provider_adapter_not_installed"


def test_optix_library_path_is_always_the_vendor_runtime(monkeypatch):
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("bundled-adapter.dll",)
    )
    monkeypatch.setenv(
        "TAICHI_FORGE_OPTIX_LIBRARY", str(Path("env/nvoptix.dll").resolve())
    )

    candidates, runtime_path, provider_source = _optix._provider_and_runtime_candidates(
        "looks-like-an-adapter.dll"
    )

    assert candidates == ("bundled-adapter.dll",)
    assert runtime_path == str(Path("looks-like-an-adapter.dll").resolve())
    assert provider_source == "forge_runtime_wheel"


def test_optix_probe_selects_newest_driver_compatible_bundled_adapter(monkeypatch):
    abi118 = _FakeOptixLibrary(optix_abi=118, runtime_probe_result=4)
    abi105 = _FakeOptixLibrary(optix_abi=105)
    abi93 = _FakeOptixLibrary(optix_abi=93)
    libraries = {"abi118.dll": abi118, "abi105.dll": abi105, "abi93.dll": abi93}

    monkeypatch.setattr(
        _optix,
        "_bundled_provider_candidates",
        lambda: tuple(libraries),
    )
    monkeypatch.setattr(
        _optix,
        "_load_library",
        lambda path: libraries[Path(path).name],
    )
    report = _optix.probe_provider("vendor/nvoptix.dll")

    assert report["discovery"] == "present"
    assert report["native_facts"]["optix_abi_version"] == 105
    assert report["native_facts"]["vendor_runtime_abi_compatible"] is True
    assert len(report["native_facts"]["rejected_newer_candidates"]) == 1
    expected = _optix._runtime_library_argument(
        str(Path("vendor/nvoptix.dll").resolve())
    )
    assert abi118.calls["probe_runtime"] == [expected]
    assert abi105.calls["probe_runtime"] == [expected]
    assert abi93.calls["probe_runtime"] == []


def test_optix_adapters_are_one_pinned_runtime_wheel_component():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "cmake" / "TaichiOptixProvider.cmake").read_text(encoding="utf-8")
    abi = (root / "taichi" / "optix" / "forge_optix_provider.h").read_text(
        encoding="utf-8"
    )
    provider = (root / "taichi" / "optix" / "provider" / "provider.cpp").read_text(
        encoding="utf-8"
    )

    runtime_project = (root / "packaging" / "runtime" / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert '"Build the pinned OptiX provider set for the runtime wheel" OFF' in cmake
    assert "set(_ti_optix_supported_abis 93 105 118)" in cmake
    assert "install(TARGETS ${target_name}" in cmake
    assert "COMPONENT runtime" in cmake
    assert "taichi_forge_optix_provider_abi1_optix${_header_abi}" in cmake
    assert "50021ea0af6d41609a97777ceebbdf1e1d34efe7" in cmake
    assert "fff65c2a7c592f1ea5f1661ad7d2381cf965f9bd" in cmake
    assert "f1f6dd803f3159992d248178f6e09421c6eb8b6d" in cmake
    runtime_build_targets = next(
        line
        for line in runtime_project.splitlines()
        if line.startswith("build.targets")
    )
    assert '"taichi_runtime"' in runtime_build_targets
    assert '"taichi_forge_optix_providers"' in runtime_build_targets
    assert 'TI_BUILD_BUNDLED_OPTIX_PROVIDERS = "ON"' in runtime_project
    assert 'TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT = "OFF"' in runtime_project
    assert "TI_ALLOW_UNQUALIFIED_OPTIX_PTX_TOOLKIT" in cmake
    assert 'set(_ti_optix_expected_ptx_version "8.5")' in cmake
    assert "--gpu-architecture=compute_75" in cmake
    assert '"-DEXPECTED_PTX_TARGET=sm_75"' in cmake
    assert "TI_FORGE_OPTIX_PROVIDER_ABI_VERSION 1u" in abi
    assert "taichi_forge_optix_provider_set_runtime_library" not in abi
    assert "taichi_forge_optix_provider_probe_runtime" not in abi
    assert "#include <optix" not in abi.lower()
    assert _optix.SUPPORTED_OPTIX_ABIS == (93, 105, 118)
    assert "OPTIX_ABI_VERSION == 118" in provider
    assert "pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams)" in provider


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_provider_load_failure_has_explicit_phase(monkeypatch):
    def fail_load(_path):
        raise OSError("injected provider load failure")

    monkeypatch.setattr(_optix, "_load_library", fail_load)
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("missing-adapter.dll",)
    )

    with pytest.raises(RuntimeError, match="provider load failure") as error:
        ti.hardware.ray.load_optix_provider()
    assert error.value._taichi_forge_hardware_failure_phase == "provider_load_failure"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_load_falls_back_only_when_newer_runtime_abi_is_unavailable(
    monkeypatch,
):
    abi118 = _FakeOptixLibrary(optix_abi=118, context_result=_optix._OPTIX_UNAVAILABLE)
    abi105 = _FakeOptixLibrary(optix_abi=105)
    libraries = {"abi118.dll": abi118, "abi105.dll": abi105}
    monkeypatch.setattr(
        _optix,
        "_bundled_provider_candidates",
        lambda: tuple(libraries),
    )
    monkeypatch.setattr(
        _optix,
        "_load_library",
        lambda path: libraries[Path(path).name],
    )

    provider = ti.hardware.ray.load_optix_provider()

    assert provider.identity["optix_abi_version"] == 105
    assert provider.identity["provider_source"] == "forge_runtime_wheel"
    assert abi118.calls["create_context"] == 1
    assert abi105.calls["create_context"] == 1
    assert abi118.calls["create_context_runtime_path"] == [None]
    assert abi105.calls["create_context_runtime_path"] == [None]
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_fake_optix_provider_scene_graph_lifetime_and_memory(monkeypatch):
    fake = _FakeOptixLibrary(optix_abi=93)
    monkeypatch.setattr(
        _optix, "_bundled_provider_candidates", lambda: ("fake-adapter.dll",)
    )
    monkeypatch.setattr(_optix, "_load_library", lambda _path: fake)
    vertices = ti.ndarray(ti.f32, shape=(3, 3))
    indices = ti.ndarray(ti.i32, shape=(1, 3))
    rays = ti.ndarray(ti.f32, shape=(2, 8))
    hits = ti.ndarray(ti.f32, shape=(2, 4))

    provider = ti.hardware.ray.load_optix_provider()
    program = impl.get_runtime().prog
    native_before = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    scene = provider.triangle_scene(vertices, indices)
    native_after_scene = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    recording = scene.record(2)
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording, admission="explicit")
    graph = builder.compile()
    graph.run({"rays": rays, "hits": hits})
    scene.refit(vertices)
    native_after_execution = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]

    report = scene.memory_report()
    assert report.known_resident_requested_bytes == 4096 + 2048 + 1024 + 80 + 24 + 192
    assert report.opaque_component_count == 1
    assert fake.calls["trace"] == 1
    assert fake.calls["update_scene"] == 1
    assert native_after_scene == native_before + 1
    assert native_after_execution == native_after_scene + 2
    with pytest.raises(RuntimeError, match="while triangle scenes are live"):
        provider.close()
    scene.close()
    with pytest.raises(RuntimeError, match="OptixTriangleScene has been closed"):
        graph.run({"rays": rays, "hits": hits})
    provider.close()
    assert fake.calls["destroy_scene"] == 1
    assert fake.calls["destroy_context"] == 1


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_optix_refit_refreshes_instance_bounds_in_graph():
    status = _optix.probe_provider()
    if status["discovery"] != "present":
        pytest.skip(f"OptiX runtime unavailable: {status['unavailable_reason']}")
    vertices = ti.ndarray(ti.f32, (3, 3))
    vertices.from_numpy(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32))
    indices = ti.ndarray(ti.i32, (1, 3))
    indices.from_numpy(np.array([[0, 1, 2]], np.int32))
    rays = ti.ndarray(ti.f32, (2, 8))
    rays.from_numpy(
        np.array(
            [[0.2, 0.3, 1, 0, 0, 0, -1, 10], [8.2, 0.3, 1, 0, 0, 0, -1, 10]], np.float32
        )
    )
    hits = ti.ndarray(ti.f32, (2, 4))

    @ti.kernel
    def relocate(v: ti.types.ndarray(ti.f32, ndim=2), offset: ti.f32):
        for i in range(3):
            v[i, 0] = offset + ti.cast(i == 1, ti.f32)

    with ti.hardware.ray.load_optix_provider() as provider:
        with provider.triangle_scene(vertices, indices) as scene:
            builder = ti.graph.GraphBuilder()
            builder.dispatch(
                relocate,
                ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vertices", ti.f32, ndim=2),
                ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.f32),
            )
            builder.append_native(scene.record_refit(), admission="explicit")
            builder.append_native(scene.record(2), admission="explicit")
            graph = builder.compile()
            before_bytes = scene.memory_report().known_resident_requested_bytes
            try:
                for offset in (8.0, 0.0, 8.0):
                    ticket = graph.submit(
                        {
                            "vertices": vertices,
                            "offset": offset,
                            "rays": rays,
                            "hits": hits,
                        }
                    )
                    ticket.wait()
                    actual = hits.to_numpy()
                    selected = int(offset != 0)
                    assert actual[selected].tolist() == [1.0, 0.0, 0.0, 1.0]
                    assert actual[1 - selected].tolist() == [-1.0, -1.0, -1.0, 0.0]
                assert (
                    scene.memory_report().known_resident_requested_bytes == before_bytes
                )
            finally:
                graph.close()
