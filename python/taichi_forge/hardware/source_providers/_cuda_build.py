"""Cold standalone CUDA addon build utilities; no runtime imports."""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_output(command):
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _nvcc_version(nvcc):
    output = _run_output([str(nvcc), "--version"])
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", output)
    if not match:
        raise RuntimeError("could not determine the NVCC version")
    return match.group(1)


def _static_cudart(toolkit_root):
    candidates = (
        toolkit_root / "lib" / "x64" / "cudart_static.lib",
        toolkit_root / "lib64" / "libcudart_static.a",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("could not locate the CUDART static library")


def _audit_target_code(cuobjdump, binary, targets):
    # Verify emitted code, not just the requested -gencode flags. Listing PTX
    # reports sm_NN names too, so preserve the container kind explicitly.
    sass = _run_output([str(cuobjdump), "--list-elf", str(binary)])
    ptx = _run_output([str(cuobjdump), "--list-ptx", str(binary)])
    observed = {f"sm_{item}" for item in re.findall(r"\.sm_(\d+)\.cubin\b", sass)}
    observed.update(f"compute_{item}" for item in re.findall(r"\.sm_(\d+)\.ptx\b", ptx))
    if observed != set(targets):
        raise RuntimeError(
            f"emitted device code differs from requested targets: {sorted(observed)} vs {targets}"
        )


def _toolkit_versions(toolkit_root):
    version_path = toolkit_root / "version.json"
    if not version_path.is_file():
        raise RuntimeError(f"CUDA Toolkit version manifest is missing: {version_path}")
    document = json.loads(version_path.read_text(encoding="utf-8"))

    def version(component):
        try:
            value = document[component]["version"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"CUDA Toolkit version manifest lacks {component}"
            ) from exc
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"CUDA Toolkit {component} version is invalid")
        return value

    return version("cuda"), version("cuda_cudart")


def _find_msvc_cl():
    direct = shutil.which("cl.exe")
    if direct:
        return Path(direct).resolve()
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    output = _run_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
    ).strip()
    if not output:
        return None
    tools = Path(output) / "VC" / "Tools" / "MSVC"
    versions = sorted((item for item in tools.iterdir() if item.is_dir()), reverse=True)
    for version in versions:
        candidate = version / "bin" / "Hostx64" / "x64" / "cl.exe"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _host_compiler(explicit):
    if explicit:
        compiler = Path(explicit).resolve()
    elif os.name == "nt":
        compiler = _find_msvc_cl()
    else:
        candidate = shutil.which("c++") or shutil.which("g++")
        compiler = None if candidate is None else Path(candidate).resolve()
    if compiler is None or not compiler.is_file():
        raise RuntimeError(
            "could not find the NVCC host compiler; pass --host-compiler explicitly"
        )
    identity = f"{compiler.name}:sha256:{_sha256(compiler)}"
    cxx_abi = (
        "msvc-x64"
        if os.name == "nt"
        else f"{platform.system().lower()}-{platform.machine()}"
    )
    return compiler, identity, cxx_abi


def _target_code(value):
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("--target-code must contain unique sm_NN/compute_NN entries")
    for item in result:
        if re.fullmatch(r"(?:sm|compute)_\d+", item) is None:
            raise ValueError("--target-code entries must use sm_NN or compute_NN")
    return result


def _gencode_flags(target_code):
    result = []
    for code in target_code:
        capability = code.split("_", 1)[1]
        result.extend(("-gencode", f"arch=compute_{capability},code={code}"))
    return result
