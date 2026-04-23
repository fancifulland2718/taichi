# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .bootstrap import get_cache_home
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home, is_manylinux2014


# -- code --
# LLVM 19.1.x (stock upstream, built with specific CMake flags — no custom
# source patches).
#
# This fork has diverged from upstream Taichi and no longer ships a
# public LLVM 19 prebuilt. The recommended flow is:
#
#   1. Run `scripts/build_llvm19_local.ps1` once to produce
#      `dist/taichi-llvm-19/` (takes ~15-25 minutes on a modern desktop).
#   2. Export `LLVM_DIR=<repo>/dist/taichi-llvm-19/lib/cmake/llvm`
#      (or just the install prefix — `setup_llvm` accepts either).
#   3. Run `python build.py ...` normally.
#
# Alternatively, set `LLVM19_WIN_URL` / `LLVM19_LINUX_URL` / etc. to a zip
# URL you have hosted yourself. If none of the above is set, `setup_llvm`
# falls back to the legacy LLVM 15 prebuilts on Linux/macOS and raises a
# clear error on Windows.

_LLVM19_WIN_URL = os.environ.get("LLVM19_WIN_URL", "")
_LLVM19_LINUX_URL = os.environ.get("LLVM19_LINUX_URL", "")
_LLVM19_LINUX_MANYLINUX_URL = os.environ.get("LLVM19_LINUX_MANYLINUX_URL", "")
_LLVM19_LINUX_AMDGPU_URL = os.environ.get("LLVM19_LINUX_AMDGPU_URL", "")
_LLVM19_MAC_ARM64_URL = os.environ.get("LLVM19_MAC_ARM64_URL", "")
_LLVM19_MAC_X64_URL = os.environ.get("LLVM19_MAC_X64_URL", "")


def _repo_root() -> str:
    """Return the absolute path to the Taichi repo root (…/.github/workflows/scripts/ti_build/llvm.py → …)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _find_local_llvm19() -> str:
    """Look for a local LLVM 19 install produced by build_llvm19_local.ps1.

    Returns the path to the lib/cmake/llvm subdirectory because CMake's
    find_package(LLVM CONFIG) expects LLVM_DIR to point directly at the
    directory containing LLVMConfig.cmake.
    """
    prefix = os.path.join(_repo_root(), "dist", "taichi-llvm-19")
    cmake_dir = os.path.join(prefix, "lib", "cmake", "llvm")
    if os.path.isfile(os.path.join(cmake_dir, "LLVMConfig.cmake")):
        return cmake_dir
    return ""


@banner("Setup LLVM")
def setup_llvm() -> None:
    """
    Download and install LLVM 19 (falling back to LLVM 15 where no LLVM 19
    prebuilt is available yet).

    Respects the LLVM_DIR environment variable: if it is set to an existing
    directory this function is a no-op, so developers with a local LLVM
    install can skip the download entirely.
    """
    existing = os.environ.get("LLVM_DIR", "")
    if existing and os.path.isdir(existing):
        return

    # Auto-detect a local build produced by scripts/build_llvm19_local.ps1
    local = _find_local_llvm19()
    if local:
        os.environ["LLVM_DIR"] = local
        print(f":: Using local LLVM 19 install at {local}")
        return

    u = platform.uname()
    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            url = _LLVM19_LINUX_AMDGPU_URL
            out = get_cache_home() / "llvm19-amdgpu"
            legacy_url = "https://github.com/GaleSeLee/assets/releases/download/v0.0.5/taichi-llvm-15.0.0-linux.zip"
            legacy_out = get_cache_home() / "llvm15-amdgpu-005"
        elif is_manylinux2014():
            url = _LLVM19_LINUX_MANYLINUX_URL
            out = get_cache_home() / "llvm19-manylinux2014"
            legacy_url = "https://github.com/ailzhang/torchhub_example/releases/download/0.3/taichi-llvm-15-linux.zip"
            legacy_out = get_cache_home() / "llvm15-manylinux2014"
        else:
            url = _LLVM19_LINUX_URL
            out = get_cache_home() / "llvm19"
            legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-linux.zip"
            legacy_out = get_cache_home() / "llvm15"
        if url:
            download_dep(url, out, strip=1)
        else:
            download_dep(legacy_url, legacy_out, strip=1)
            out = legacy_out
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        url = _LLVM19_MAC_ARM64_URL
        out = get_cache_home() / "llvm19-m1"
        if url:
            download_dep(url, out, strip=1)
        else:
            legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/taichi-llvm-15-m1-nozstd.zip"
            out = get_cache_home() / "llvm15-m1-nozstd"
            download_dep(legacy_url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "x86_64"):
        url = _LLVM19_MAC_X64_URL
        out = get_cache_home() / "llvm19-mac"
        if url:
            download_dep(url, out, strip=1)
        else:
            legacy_url = "https://github.com/taichi-dev/taichi_assets/releases/download/llvm15/llvm-15-mac10.15.zip"
            out = get_cache_home() / "llvm15-mac"
            download_dep(legacy_url, out, strip=1)
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        if not _LLVM19_WIN_URL:
            raise RuntimeError(
                "LLVM 19 for Windows is not available for download and no local "
                "install was found.\n\n"
                "Please build it once locally:\n"
                "    pwsh -File scripts/build_llvm19_local.ps1\n\n"
                "Then re-run `python build.py`. The script installs to "
                "dist/taichi-llvm-19 which setup_llvm auto-detects.\n\n"
                "Alternatively, set LLVM19_WIN_URL to a zip URL you host "
                "yourself, or set LLVM_DIR to an existing install prefix."
            )
        out = get_cache_home() / "llvm19"
        download_dep(_LLVM19_WIN_URL, out, strip=0)
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
