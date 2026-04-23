# -*- coding: utf-8 -*-

# -- stdlib --
from pathlib import Path
import os
from os.path import join
import json
import platform
import shutil
import tempfile
import sys
import subprocess

# -- third party --
# -- own --
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, error, get_cache_home, warn
from .tinysh import powershell


def grep(value, target):
    for line in target.split("\n"):
        if value in line:
            return line


# -- code --
@banner("Setup Clang")
def setup_clang(as_compiler=True) -> None:
    """
    Setup Clang.
    """
    u = platform.uname()
    if u.system == "Linux":
        for v in ("", "-14", "-13", "-12", "-11", "-10"):
            clang = shutil.which(f"clang{v}")
            if clang is not None:
                clangpp = shutil.which(f"clang++{v}")
                assert clangpp
                break
        else:
            error("Could not find clang of any version")
            return
    elif u.system == "Darwin":
        brew_config = subprocess.check_output(["brew", "config"]).decode("utf-8")
        print("brew_config", brew_config)
        brew_prefix = grep("HOMEBREW_PREFIX", brew_config).split()[1]
        print("brew_prefix", brew_prefix)
        clang = join(brew_prefix, "opt", "llvm@15", "bin", "clang")
        clangpp = join(brew_prefix, "opt", "llvm@15", "bin", "clang++")
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / "clang-14-v2"
        url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/clang-14.0.6-win-complete.zip"
        download_dep(url, out, force=True)
        clang = str(out / "bin" / "clang++.exe").replace("\\", "\\\\")
        clangpp = clang
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    cmake_args["CLANG_EXECUTABLE"] = clang

    if as_compiler:
        cc = os.environ.get("CC")
        cxx = os.environ.get("CXX")
        if cc:
            warn(f"Explicitly specified compiler via environment variable CC={cc}, not configuring clang.")
        else:
            cmake_args["CMAKE_C_COMPILER"] = clang

        if cxx:
            warn(f"Explicitly specified compiler via environment variable CXX={cxx}, not configuring clang++.")
        else:
            cmake_args["CMAKE_CXX_COMPILER"] = clangpp


ENV_EXTRACT_SCRIPT = """
param ([string]$DevShell, [string]$VsPath, [string]$OutFile)
$WarningPreference = 'SilentlyContinue'
Import-Module $DevShell
Enter-VsDevShell -VsInstallPath $VsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64"
Get-ChildItem env:* | ConvertTo-Json -Depth 1 | Out-File $OutFile
"""


def _vs_devshell(vs):
    dll = vs / "Common7" / "Tools" / "Microsoft.VisualStudio.DevShell.dll"

    if not dll.exists():
        error("Could not find Visual Studio DevShell")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "extract.ps1"
        with script.open("w") as f:
            f.write(ENV_EXTRACT_SCRIPT)
        outfile = tmp / "env.json"
        powershell(
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-DevShell",
            str(dll),
            "-VsPath",
            str(vs),
            "-OutFile",
            str(outfile),
        )
        with outfile.open(encoding="utf-16") as f:
            envs = json.load(f)

    for v in envs:
        os.environ[v["Key"]] = v["Value"]


@banner("Setup MSVC")
def setup_msvc() -> None:
    assert platform.system() == "Windows"

    # Prefer vswhere so we can transparently pick up VS 2026 (or any
    # future major). Fall back to the legacy hard-coded paths only if
    # vswhere is unavailable.
    vs: Path | None = None
    vswhere_candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio\\Installer\\vswhere.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio\\Installer\\vswhere.exe",
    ]
    vswhere = next((p for p in vswhere_candidates if p.exists()), None)
    if vswhere is not None:
        try:
            out = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-prerelease",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
            ).strip()
            if out:
                vs = Path(out)
        except subprocess.CalledProcessError:
            vs = None

    # Legacy fallback — enumerate the layouts used by the VS 2026 installer.
    if vs is None:
        legacy_roots = [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
            Path(r"C:\Program Files\Microsoft Visual Studio"),
        ]
        for base in legacy_roots:
            for ver in ("2026",):
                for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
                    candidate = base / ver / edition
                    if candidate.exists():
                        vs = candidate
                        break
                if vs:
                    break
            if vs:
                break

    if vs is not None:
        # Always use Ninja + cl.exe for local builds. Ninja works
        # identically with any MSVC toolchain (2022, 2026, future), avoids
        # version-pinned generators like `Visual Studio 17 2022`, and
        # lets sccache wrap cl.exe transparently.
        _vs_devshell(vs)
        cmake_args["CMAKE_C_COMPILER"] = "cl.exe"
        cmake_args["CMAKE_CXX_COMPILER"] = "cl.exe"
        os.environ["CMAKE_GENERATOR"] = "Ninja"
        os.environ.pop("TAICHI_USE_MSBUILD", None)
        return

    # Nothing found — install VS 2026 Build Tools as a last resort.
    url = "https://aka.ms/vs/18/release/vs_BuildTools.exe"
    out = Path(r"C:\Program Files (x86)\Microsoft Visual Studio") / "2026" / "BuildTools"
    download_dep(
        url,
        out,
        elevate=True,
        args=[
            "--passive",
            "--wait",
            "--norestart",
            "--includeRecommended",
            "--add",
            "Microsoft.VisualStudio.Workload.VCTools",
        ],
    )
    warn("Please restart build.py after Visual Studio Build Tools is installed.")
    sys.exit(1)
