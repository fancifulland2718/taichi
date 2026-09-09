import hashlib
import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils
from taichi_forge.hardware._cub_source_provider import load_cub_source_provider
from taichi_forge.hardware._retained import retained_execution_contract
from taichi_forge.hardware._source_provider import (
    SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
    SourceProviderManifestError,
    load_source_provider_manifest,
)
from taichi_forge.lang import impl


def _manifest(binary_name, binary_hash):
    return {
        "schema_version": SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION,
        "build_profile": {
            "schema_version": 1,
            "kind": "cuda-toolkit-addon",
            "abi_boundary": "provider-c-abi",
            "driver_contract": {
                "minimum_api_version": 13000,
                "ptx_api_version": 13020,
                "basis": "explicit-test-contract",
            },
        },
        "provider_id": "cub_reference",
        "provider_abi": "taichi-forge-cub-source-provider-c-abi1",
        "provider_abi_version": 1,
        "binary": {"path": binary_name, "sha256": binary_hash},
        "toolchain": {
            "cuda_toolkit": "13.2",
            "nvcc": "13.2.51",
            "compiler_components": [
                {"name": "nvcc", "version": "13.2.51", "sha256": "c" * 64},
                {"name": "ptxas", "version": "13.2.51", "sha256": "d" * 64},
            ],
            "host_compiler": "MSVC 19.44",
            "cxx_abi": "msvc-19",
            "build_flags": ["--shared", "-O3"],
            "target_code": ["sm_89", "compute_89"],
            "source_dependencies": [
                {"name": "cccl/cub", "version": "3.2.0", "sha256": "b" * 64}
            ],
        },
        "runtime_dependencies": [
            {
                "name": "cudart",
                "linkage": "static",
                "version": "13.2",
                "sha256": None,
            }
        ],
        "source_identity": {"kind": "sha256", "value": "a" * 64},
        "specializations": [
            {
                "operation": "radix_sort_pairs",
                "key_dtype": "u32",
                "value_dtype": "u32",
                "temporary_storage": "caller_owned",
            }
        ],
    }


def _write_manifest(tmp_path, document):
    path = tmp_path / "provider.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def test_source_provider_manifest_binds_binary_toolchain_and_target(tmp_path):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    path = _write_manifest(tmp_path, _manifest(binary.name, binary_hash))

    manifest = load_source_provider_manifest(
        path,
        expected_provider_id="cub_reference",
        expected_provider_abi="taichi-forge-cub-source-provider-c-abi1",
    )

    assert manifest.binary_path == binary.resolve()
    assert manifest.binary_sha256 == binary_hash
    assert manifest.identity["provider_id"] == "cub_reference"
    assert manifest.supports_cuda_compute_capability(89)
    assert manifest.supports_cuda_compute_capability(90)
    assert not manifest.supports_cuda_compute_capability(80)
    assert manifest.runtime_dependencies[0].linkage == "static"
    assert manifest.toolchain["source_dependencies"][0].name == "cccl/cub"
    assert manifest.specializations[0]["temporary_storage"] == "caller_owned"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda item: item.update(extra=True), "fields are not exact"),
        (lambda item: item["binary"].update(path="../escape.dll"), "escapes"),
        (lambda item: item["binary"].update(sha256="0" * 64), "SHA-256 mismatch"),
        (lambda item: item["toolchain"].update(target_code=["all"]), "sm_NN"),
        (lambda item: item.update(provider_abi_version=0), "positive integer"),
        (
            lambda item: item["build_profile"].update(kind="portable-runtime"),
            "CUDA addon",
        ),
        (
            lambda item: item["build_profile"]["driver_contract"].update(
                minimum_api_version=True
            ),
            "positive integer",
        ),
        (
            lambda item: item["build_profile"]["driver_contract"].update(
                ptx_api_version=12000
            ),
            "runtime floor",
        ),
    ],
)
def test_source_provider_manifest_fails_closed(tmp_path, mutation, match):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    document = _manifest(binary.name, binary_hash)
    mutation(document)
    path = _write_manifest(tmp_path, document)

    with pytest.raises(SourceProviderManifestError, match=match):
        load_source_provider_manifest(path)


def test_source_provider_manifest_rejects_provider_identity_mismatch(tmp_path):
    binary = tmp_path / "cub_provider.dll"
    binary.write_bytes(b"provider-binary")
    path = _write_manifest(
        tmp_path,
        _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest()),
    )

    with pytest.raises(SourceProviderManifestError, match="does not match"):
        load_source_provider_manifest(path, expected_provider_id="mathdx_reference")


def test_source_provider_build_report_is_path_and_source_commit_independent(tmp_path):
    binary = tmp_path / "provider.dll"
    binary.write_bytes(b"same artifact")
    document = _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest())
    first = load_source_provider_manifest(_write_manifest(tmp_path, document))
    document["source_identity"] = {"kind": "git", "value": "another-commit"}
    document["toolchain"]["build_flags"] += [
        "--compiler-bindir",
        "another-installation",
    ]
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / binary.name).write_bytes(binary.read_bytes())
    second = load_source_provider_manifest(_write_manifest(moved, document))
    assert first.manifest_sha256 != second.manifest_sha256
    assert first.build_report() == second.build_report()
    report = first.build_report()
    assert json.loads(json.dumps(report)) == report
    assert {item["name"] for item in report["components"]} == {
        "nvcc",
        "ptxas",
        "cccl/cub",
        "cudart",
    }
    assert report["profile"]["kind"] == "cuda-toolkit-addon"
    with pytest.raises(TypeError):
        first.build_profile["driver_contract"]["minimum_api_version"] = 0
    document["build_profile"]["driver_contract"]["minimum_api_version"] = 13010
    third = load_source_provider_manifest(_write_manifest(moved, document))
    assert third.build_report()["build_identity"] != report["build_identity"]
    document["toolchain"]["compiler_components"][1]["version"] = "13.3.73"
    fourth = load_source_provider_manifest(_write_manifest(moved, document))
    assert (
        fourth.build_report()["build_identity"]
        != third.build_report()["build_identity"]
    )


@pytest.mark.parametrize(
    "targets,capability,driver,path,reason",
    [
        (["sm_80", "compute_80"], 89, 13000, "sass", None),
        (["sm_89", "compute_89"], 89, 12090, "sass", "driver_api_too_old"),
        (["sm_89", "compute_89"], 120, 13000, "ptx_jit", "ptx_jit_driver_too_old"),
        (["sm_89", "compute_89"], 120, 13020, "ptx_jit", None),
        (["sm_89"], 90, 13020, None, "target_code_unavailable"),
        (["sm_89", "compute_89"], 80, 13020, None, "target_code_unavailable"),
        (["sm_89"], 89, None, "sass", "driver_version_unavailable"),
    ],
)
def test_source_provider_distinguishes_sass_from_ptx_driver_contract(
    tmp_path, targets, capability, driver, path, reason
):
    binary = tmp_path / "provider.dll"
    binary.write_bytes(b"artifact")
    document = _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest())
    document["toolchain"]["target_code"] = targets
    manifest = load_source_provider_manifest(_write_manifest(tmp_path, document))
    result = manifest.cuda_compatibility(capability, driver)
    assert result["code_path"] == path
    assert result["unavailable_reason"] == reason
    assert result["eligible"] is (reason is None)
    assert result["execution_qualified"] is False


def test_source_provider_accepts_legacy_schema_without_inventing_driver_floor(tmp_path):
    binary = tmp_path / "provider.dll"
    binary.write_bytes(b"legacy")
    document = _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest())
    document["schema_version"] = 2
    del document["build_profile"]
    del document["toolchain"]["compiler_components"]
    manifest = load_source_provider_manifest(_write_manifest(tmp_path, document))
    assert manifest.build_report()["profile"] is None
    result = manifest.cuda_compatibility(89, None)
    assert result["eligible"]
    assert not result["driver_contract_declared"]
    assert result["required_driver_api_version"] is None


def test_cub_source_provider_rejects_incompatible_driver_before_library_load(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from taichi_forge.hardware import _cub_source_provider as cub

    binary = tmp_path / "provider.dll"
    binary.write_bytes(b"must not load")
    document = _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest())
    path = _write_manifest(tmp_path, document)
    monkeypatch.setattr(cub.impl, "get_runtime", lambda: SimpleNamespace(prog=object()))
    monkeypatch.setattr(cub, "active_backend", lambda: "cuda")
    monkeypatch.setattr(cub.impl, "get_cuda_compute_capability", lambda: 120)
    monkeypatch.setattr(cub._ti_core, "cuda_driver_api_version", lambda: 13000)
    monkeypatch.setattr(
        cub, "_load_process_library", lambda _: pytest.fail("ineligible DLL loaded")
    )
    with pytest.raises(ti.TaichiRuntimeError, match="ptx_jit_driver_too_old"):
        load_cub_source_provider(path)


def test_source_provider_builder_checks_emitted_sass_and_ptx(monkeypatch):
    from taichi_forge.hardware.source_providers.cub import build

    assert build._MANIFEST_SCHEMA_VERSION == SOURCE_PROVIDER_MANIFEST_SCHEMA_VERSION
    outputs = {
        "--list-elf": "ELF file 1: provider.1.sm_80.cubin\n",
        "--list-ptx": "PTX file 1: provider.1.sm_80.ptx\n",
    }
    monkeypatch.setattr(
        build._shared, "_run_output", lambda command: outputs[command[1]]
    )
    build._audit_target_code("cuobjdump", "provider.dll", ("sm_80", "compute_80"))
    outputs["--list-ptx"] = ""
    with pytest.raises(RuntimeError, match="emitted device code differs"):
        build._audit_target_code("cuobjdump", "provider.dll", ("sm_80", "compute_80"))


def test_source_provider_binary_audit_uses_shared_dependency_policy(
    tmp_path, monkeypatch
):
    from scripts import validate_source_provider as validator

    binary = tmp_path / "provider.dll"
    binary.write_bytes(b"artifact")
    document = _manifest(binary.name, hashlib.sha256(binary.read_bytes()).hexdigest())
    path = _write_manifest(tmp_path, document)
    calls = []
    monkeypatch.setattr(
        validator, "_validate_binary_dependencies", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(validator, "_binary_imports", lambda *args: {"KERNEL32.dll"})
    report = validator.audit(path, "windows")
    assert calls == [(binary.resolve(), "windows")]
    assert report["build"]["profile"]["kind"] == "cuda-toolkit-addon"
    assert report["binary_imports"] == ["KERNEL32.dll"]
    assert not report["execution_qualified"]


def test_cub_optional_scan_capability_does_not_break_old_addon_abi():
    from types import SimpleNamespace
    from taichi_forge.hardware._cub_source_provider import CubSourcePlan

    old = SimpleNamespace(_library=SimpleNamespace(info=SimpleNamespace(features=0xF)))
    with pytest.raises(ti.TaichiRuntimeError, match="does not provide"):
        CubSourcePlan(old, "segmented_inclusive_scan_u32", 32)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype", [ti.i32, ti.u32])
@pytest.mark.parametrize("inclusive", [True, False])
def test_cub_segmented_scan_reset_monoid_matches_modular_integer_reference(
    dtype, inclusive
):
    manifest_path = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not manifest_path:
        pytest.skip("a user-built CUB addon was not supplied")
    provider = load_cub_source_provider(manifest_path)
    if provider._library.info.features & 0x30 != 0x30:
        pytest.skip("this ABI-compatible addon predates reset-monoid scan")
    rng = np.random.default_rng(60809)
    mode = "inclusive" if inclusive else "exclusive"
    # Word, warp, block and multi-tile boundaries; empty and singleton
    # segments; and one long segment extending across many scan tiles.
    layouts = [
        (1, [0, 0, 1]),
        (33, [0, 1, 31, 32, 32, 33]),
        (4097, list(range(4098))),
        (65537, [0, 0, 1, 32, 129, 129, 65281, 65537]),
        (1000003, [0, 1000003]),
    ]
    for n, offsets in layouts:
        words = np.zeros((n + 31) // 32, dtype=np.uint32)
        for begin in offsets[:-1]:
            if begin < n:
                words[begin >> 5] |= np.uint32(1 << (begin & 31))
        data = rng.integers(0, 2**32, size=n, dtype=np.uint32)
        values, output = ti.ndarray(dtype, n), ti.ndarray(dtype, n)
        heads = ti.ndarray(ti.u32, len(words))
        heads.from_numpy(words)
        values.from_numpy(data.view(np.int32) if dtype == ti.i32 else data)
        plan = provider.plan(f"segmented_{mode}_scan_u32", n)
        plan.run(input=values, heads=heads, output=output)
        ti.sync()
        expected = np.empty(n, dtype=np.uint32)
        for begin, end in zip(offsets, offsets[1:]):
            if begin == end:
                continue
            partial = np.cumsum(data[begin:end], dtype=np.uint32)
            if inclusive:
                expected[begin:end] = partial
            else:
                expected[begin] = 0
                expected[begin + 1 : end] = partial[:-1]
        np.testing.assert_array_equal(output.to_numpy().view(np.uint32), expected)
        # A second input must reset tile state rather than reuse old carry.
        values.fill(0)
        plan.run(input=values, heads=heads, output=output)
        ti.sync()
        np.testing.assert_array_equal(output.to_numpy(), np.zeros(n, dtype=np.uint32))


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("n", [257, 65537])
def test_cub_source_provider_executes_explicit_primitives_and_graph(n, monkeypatch):
    manifest_path = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not manifest_path:
        pytest.skip("a user-built CUB source-provider manifest was not supplied")

    provider = load_cub_source_provider(manifest_path)
    assert load_cub_source_provider(manifest_path)._library is provider._library
    rng = np.random.default_rng(20260827)
    keys_u32_values = rng.integers(0, 19, size=n, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)
    keys_u32 = ti.ndarray(ti.u32, shape=n)
    values_in = ti.ndarray(ti.u32, shape=n)
    keys_u32_out = ti.ndarray(ti.u32, shape=n)
    values_out = ti.ndarray(ti.u32, shape=n)
    keys_u32.from_numpy(keys_u32_values)
    values_in.from_numpy(values)
    sort_u32 = provider.plan("radix_sort_pairs_u32", n)
    program = impl.get_runtime().prog
    native_before = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    sort_u32.run(
        keys_in=keys_u32,
        values_in=values_in,
        keys_out=keys_u32_out,
        values_out=values_out,
    )
    native_after = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    assert native_after == native_before + 1
    ti.sync()
    stable_order = np.argsort(keys_u32_values, kind="stable")
    np.testing.assert_array_equal(
        keys_u32_out.to_numpy(), keys_u32_values[stable_order]
    )
    np.testing.assert_array_equal(values_out.to_numpy(), values[stable_order])

    keys_u64_values = (
        rng.integers(0, 31, size=n, dtype=np.uint64) << np.uint64(33)
    ) | rng.integers(0, 37, size=n, dtype=np.uint64)
    keys_u64 = ti.ndarray(ti.u64, shape=n)
    keys_u64_out = ti.ndarray(ti.u64, shape=n)
    keys_u64.from_numpy(keys_u64_values)
    sort_u64 = provider.plan("radix_sort_pairs_u64", n)
    sort_u64.run(
        keys_in=keys_u64,
        values_in=values_in,
        keys_out=keys_u64_out,
        values_out=values_out,
    )
    ti.sync()
    stable_order = np.argsort(keys_u64_values, kind="stable")
    np.testing.assert_array_equal(
        keys_u64_out.to_numpy(), keys_u64_values[stable_order]
    )
    np.testing.assert_array_equal(values_out.to_numpy(), values[stable_order])

    scan_values = rng.integers(0, 7, size=n, dtype=np.uint32)
    scan_in = ti.ndarray(ti.u32, shape=n)
    scan_out = ti.ndarray(ti.u32, shape=n)
    scan_in.from_numpy(scan_values)
    scan = provider.plan("exclusive_scan_u32", n)
    scan.run(input=scan_in, output=scan_out)
    ti.sync()
    expected_scan = np.zeros(n, dtype=np.uint32)
    expected_scan[1:] = np.cumsum(scan_values[:-1], dtype=np.uint32)
    np.testing.assert_array_equal(scan_out.to_numpy(), expected_scan)

    flags_values = rng.integers(0, 2, size=n, dtype=np.uint32)
    flags = ti.ndarray(ti.u32, shape=n)
    selected = ti.ndarray(ti.u32, shape=n)
    selected_count = ti.ndarray(ti.u32, shape=1)
    flags.from_numpy(flags_values)
    select = provider.plan("select_flagged_u32", n)
    select.run(input=values_in, flags=flags, output=selected, count=selected_count)
    ti.sync()
    expected_selected = values[flags_values != 0]
    assert int(selected_count.to_numpy()[0]) == len(expected_selected)
    np.testing.assert_array_equal(
        selected.to_numpy()[: len(expected_selected)], expected_selected
    )

    scan_out.fill(0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(scan)
    graph = builder.compile()
    for _ in range(3):
        graph.run({"input": scan_in, "output": scan_out})
    ti.sync()
    np.testing.assert_array_equal(scan_out.to_numpy(), expected_scan)
    assert graph._debug_info["native_count"] == 1
    assert graph._spec.lifetime_leases
    physical_plans = graph.definition.to_dict()["planned_physical_manifest"][
        "execution"
    ]["native_physical_plans"]
    assert any(scan._graph_physical_plan_id == item[2] for item in physical_plans)

    # Profile/eligibility work belongs to explicit load/plan boundaries, not
    # an already bound replay. Also cross CTA boundaries in the larger case.
    with monkeypatch.context() as replay_patch:
        replay_patch.setattr(
            type(provider.manifest),
            "build_report",
            lambda _: pytest.fail("replay built a profile"),
        )
        replay_patch.setattr(
            type(provider.manifest),
            "cuda_compatibility",
            lambda *args: pytest.fail("replay checked driver eligibility"),
        )
        for _ in range(32):
            graph.run({"input": scan_in, "output": scan_out})
    ti.sync()
    np.testing.assert_array_equal(scan_out.to_numpy(), expected_scan)

    retained = retained_execution_contract(scan)
    assert retained.identity.provider_id == "cub_reference"
    assert retained.automatic_selection_policy == "forbidden"
    assert retained.identity.to_dict()["problem_scope"] == {
        "num_items": n,
        "operation": "exclusive_scan_u32",
    }
    assert tuple(item.name for item in retained.cost_model.fixed_costs) == (
        "manifest_and_binary_validation",
        "provider_library_load",
        "workspace_query_and_allocation",
        "ctypes_dispatch",
        "submission_registration",
    )
    assert retained.cost_model.scale_costs[0].dimensions == ("num_items",)
    assert scan.workspace_bytes > 0

    with pytest.raises(RuntimeError, match="must not alias"):
        scan.run(input=scan_in, output=scan_in)
    wrong = ti.ndarray(ti.u64, shape=n)
    with pytest.raises(RuntimeError, match="compact scalar"):
        scan.run(input=wrong, output=scan_out)
