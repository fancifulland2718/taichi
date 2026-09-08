from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from scripts import validate_runtime_wheel
from scripts import validate_shim_wheel
from scripts import repair_runtime_wheel


def _wheel_with_member(path: Path, member: str) -> ZipFile:
    with ZipFile(path, "w") as zf:
        zf.writestr(member, b"native binary placeholder")
    return ZipFile(path)


def test_strict_elf_export_audit_matches_final_binary(monkeypatch, tmp_path):
    wheel = _wheel_with_member(
        tmp_path / "runtime.whl",
        "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
    )
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "TAICHI_FORGE_RUNTIME_PRIVATE_1 A 0 0\n"
                "taichi_runtime_anchor@@TAICHI_FORGE_RUNTIME_PRIVATE_1 T 10 4\n"
            ),
        ),
    )
    try:
        validate_runtime_wheel._strict_binary_exports(
            wheel,
            "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
            "manylinux",
            ["TAICHI_FORGE_RUNTIME_PRIVATE_1", "taichi_runtime_anchor"],
        )
    finally:
        wheel.close()


@pytest.mark.parametrize("provider", ("cudss", "vkfft"))
def test_windows_provider_optional_exports_are_owner_scoped(
    provider, monkeypatch, tmp_path
):
    required = f"taichi_forge_{provider}_provider_query"
    stem = (
        validate_runtime_wheel.VKFFT_ADAPTER_STEM
        if provider == "vkfft"
        else "taichi_forge_cudss_provider_abi1_cudss080"
    )
    member = f"taichi_forge_runtime/_lib/hardware_providers/{stem}.dll"
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    exports = set()
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="\n".join(
                f"    {index} 0 00001000 {symbol}"
                for index, symbol in enumerate(sorted(exports), 1)
            ),
        ),
    )
    with _wheel_with_member(tmp_path / "runtime.whl", member) as wheel:
        def audit():
            validate_runtime_wheel._strict_provider_exports(
                wheel, {member: required}, "windows"
            )

        exports.add(required)
        audit()  # Previous adapters do not need the new extension symbols.
        extensions = validate_runtime_wheel.OPTIONAL_PROVIDER_EXPORTS[required]
        for extension in sorted(extensions):
            exports.add(extension)
            audit()
        exports.remove(required)
        with pytest.raises(RuntimeError, match="requires"):
            audit()
        exports.add(required)
        exports.add("taichi_forge_other_provider_query")
        with pytest.raises(RuntimeError, match="permits only"):
            audit()
        if provider == "vkfft":
            exports.remove("taichi_forge_other_provider_query")
            exports.add("unintended_shader_compiler_export")
            with pytest.raises(RuntimeError, match="permits only"):
                audit()


def test_strict_elf_export_audit_rejects_repair_drift(monkeypatch, tmp_path):
    wheel = _wheel_with_member(
        tmp_path / "runtime.whl",
        "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
    )
    monkeypatch.setattr(validate_runtime_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_runtime_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="taichi_runtime_anchor T 10 4\nLLVMContextCreate T 20 4\n",
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="differ from the audited manifest"):
            validate_runtime_wheel._strict_binary_exports(
                wheel,
                "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
                "manylinux",
                ["taichi_runtime_anchor"],
            )
    finally:
        wheel.close()


def test_strict_shim_audit_requires_dependency_and_relative_runpath(
    monkeypatch, tmp_path
):
    wheel = _wheel_with_member(
        tmp_path / "shim.whl",
        "taichi_forge/_lib/core/taichi_python.so",
    )
    monkeypatch.setattr(validate_shim_wheel.shutil, "which", lambda name: name)
    monkeypatch.setattr(
        validate_shim_wheel.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "(NEEDED) Shared library: [libtaichi_runtime.so]\n"
                "(RUNPATH) Library runpath: [$ORIGIN/../../../"
                "taichi_forge_runtime/_lib/runtime_native]\n"
            ),
        ),
    )
    try:
        validate_shim_wheel._strict_dynamic_contract(
            wheel,
            "taichi_forge/_lib/core/taichi_python.so",
            "manylinux",
        )
    finally:
        wheel.close()


def test_manylinux_normalization_requires_canonical_primary_runtime(tmp_path):
    canonical = tmp_path / "canonical.whl"
    with ZipFile(canonical, "w") as zf:
        zf.writestr(
            "taichi_forge_runtime/_lib/runtime_native/libtaichi_runtime.so",
            b"runtime",
        )
    repair_runtime_wheel.normalize_manylinux_wheel(canonical)

    hashed = tmp_path / "hashed.whl"
    with ZipFile(hashed, "w") as zf:
        zf.writestr(
            "taichi_forge_runtime/_lib/runtime_native/"
            "libtaichi_runtime-deadbeef.so",
            b"runtime",
        )
    with pytest.raises(SystemExit, match="must preserve.*primary runtime"):
        repair_runtime_wheel.normalize_manylinux_wheel(hashed)
