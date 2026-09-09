"""Build the optional CUTLASS addon; requires caller-owned Toolkit and headers."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

_spec = importlib.util.spec_from_file_location(
    "_forge_cuda_addon_build", Path(__file__).parents[1] / "_cuda_build.py"
)
_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build)


def build_cutlass_source_provider(
    output_directory, *, cutlass_root, target_code, nvcc=None, host_compiler=None
):
    root = Path(cutlass_root).resolve(strict=True)
    include = root / "include"
    version_header = (include / "cutlass/version.h").read_text(encoding="utf-8")
    version = ".".join(
        re.search(rf"^#define CUTLASS_{key}\s+(\d+)", version_header, re.M).group(1)
        for key in ("MAJOR", "MINOR", "PATCH")
    )
    # Hash the header tree, not merely version.h: patched upstream is allowed
    # but must produce a different reusable recipe identity.
    digest = hashlib.sha256()
    for path in sorted(p for p in include.rglob("*") if p.is_file()):
        digest.update(path.relative_to(include).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(_build._sha256(path)))
    headers_identity = digest.hexdigest()
    compiler = Path(nvcc or shutil.which("nvcc") or "").resolve(strict=True)
    if not compiler.is_file():
        raise ValueError("Pass the NVCC executable explicitly or configure PATH")
    toolkit = compiler.parent.parent
    cuda_version, cudart_version = _build._toolkit_versions(toolkit)
    nvcc_version = _build._nvcc_version(compiler)
    targets = _build._target_code(target_code)
    cl, cl_identity, cxx_abi = _build._host_compiler(host_compiler)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = output / (
        "taichi_forge_cutlass_abi1" + (".dll" if os.name == "nt" else ".so")
    )
    source = Path(__file__).with_name("provider.cu")
    flags = [
        "--shared",
        "-O3",
        "--std=c++17",
        "--cudart=static",
        "--compiler-bindir",
        str(cl.parent),
        "-I",
        str(include),
        *_build._gencode_flags(targets),
    ]
    flags += (
        ["-Xcompiler=/Zc:preprocessor", "-Xcompiler=/MT"]
        if os.name == "nt"
        else ["-Xcompiler=-fPIC"]
    )
    subprocess.run([str(compiler), str(source), *flags, "-o", str(binary)], check=True)
    suffix = ".exe" if os.name == "nt" else ""
    _build._audit_target_code(compiler.with_name("cuobjdump" + suffix), binary, targets)
    components = [
        {"name": "nvcc", "version": nvcc_version, "sha256": _build._sha256(compiler)}
    ]
    if any(code.startswith("sm_") for code in targets):
        ptxas = compiler.with_name("ptxas" + suffix)
        components.append(
            {
                "name": "ptxas",
                "version": _build._nvcc_version(ptxas),
                "sha256": _build._sha256(ptxas),
            }
        )
    license_path = output / "CUTLASS-LICENSE.txt"
    shutil.copyfile(root / "LICENSE.txt", license_path)
    document = {
        "schema_version": 3,
        "provider_id": "cutlass_matmul",
        "provider_abi": "taichi-forge-cutlass-matmul-c-abi1",
        "provider_abi_version": 1,
        "binary": {"path": binary.name, "sha256": _build._sha256(binary)},
        "build_profile": {
            "schema_version": 1,
            "kind": "cuda-toolkit-addon",
            "abi_boundary": "provider-c-abi",
            "driver_contract": {
                "minimum_api_version": int(cudart_version.split(".")[0]) * 1000,
                "ptx_api_version": int(nvcc_version.split(".")[0]) * 1000
                + int(nvcc_version.split(".")[1]) * 10,
                "basis": "cuda-minor-compatibility-for-sass; compiler-release-driver-for-ptx-jit",
            },
        },
        "toolchain": {
            "cuda_toolkit": cuda_version,
            "nvcc": nvcc_version,
            "host_compiler": cl_identity,
            "cxx_abi": cxx_abi,
            "build_flags": flags,
            "target_code": list(targets),
            "compiler_components": components,
            "source_dependencies": [
                {"name": "cutlass", "version": version, "sha256": headers_identity}
            ],
        },
        "runtime_dependencies": [
            {
                "name": "cudart",
                "linkage": "static",
                "version": cudart_version,
                "sha256": _build._sha256(_build._static_cudart(toolkit)),
            }
        ],
        "source_identity": {"kind": "sha256", "value": _build._sha256(source)},
        "specializations": [
            {
                "operation": "matmul",
                "dtype": "f32",
                "batch_count": 1,
                "numeric_policy": "simt-f32-no-tf32",
                "strategy": strategy,
                "epilogue": "identity-or-relu",
                "temporary_storage": "graph_owned",
            }
            for strategy in (
                "direct_fused",
                "split_k_reduce_fused",
                "split_k_wide_reduce_fused",
            )
        ],
    }
    manifest = output / "cutlass_source_provider.json"
    manifest.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutlass-root", required=True)
    parser.add_argument("--target-code", required=True)
    parser.add_argument("--nvcc")
    parser.add_argument("--host-compiler")
    args = parser.parse_args()
    print(
        build_cutlass_source_provider(
            args.output,
            cutlass_root=args.cutlass_root,
            target_code=args.target_code,
            nvcc=args.nvcc,
            host_compiler=args.host_compiler,
        )
    )
