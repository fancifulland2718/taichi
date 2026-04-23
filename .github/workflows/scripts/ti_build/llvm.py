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
# source patches). The Windows binary is produced by
# .github/workflows/build_llvm19_windows.yml and published as a release
# asset on this repo. Linux/macOS prebuilts are not yet re-produced for
# LLVM 19 — if LLVM19_OVERRIDE_URL is not provided, those platforms fall
# back to the legacy LLVM 15 prebuilt (deprecated, to be removed once
# platform-specific LLVM 19 zips exist).
#
# If you already have LLVM 19 installed locally, set LLVM_DIR to point at
# <prefix>/lib/cmake/llvm and this downloader will be bypassed.

_LLVM19_WIN_URL = os.environ.get(
    "LLVM19_WIN_URL",
    # Placeholder. Replace with the release asset URL produced by
    # build_llvm19_windows.yml once the first build is published.
    "https://github.com/taichi-dev/taichi_assets/releases/download/llvm19/taichi-llvm-19-msvc2022.zip",
)
_LLVM19_LINUX_URL = os.environ.get("LLVM19_LINUX_URL", "")
_LLVM19_LINUX_MANYLINUX_URL = os.environ.get("LLVM19_LINUX_MANYLINUX_URL", "")
_LLVM19_LINUX_AMDGPU_URL = os.environ.get("LLVM19_LINUX_AMDGPU_URL", "")
_LLVM19_MAC_ARM64_URL = os.environ.get("LLVM19_MAC_ARM64_URL", "")
_LLVM19_MAC_X64_URL = os.environ.get("LLVM19_MAC_X64_URL", "")


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
        out = get_cache_home() / "llvm19"
        download_dep(_LLVM19_WIN_URL, out, strip=0)
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
