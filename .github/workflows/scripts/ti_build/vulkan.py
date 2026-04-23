# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .dep import download_dep
from .misc import banner, get_cache_home, path_prepend
from .python import path_prepend


# -- code --
# Vulkan SDK pinned version. 1.3.296.0 -> 1.4.304.1 as part of the Phase A
# toolchain refresh. The SDK layout and component IDs we use are stable
# across minor versions, so the only thing that changes is the URL / path.
_VULKAN_VERSION = os.environ.get("VULKAN_SDK_VERSION", "1.4.304.1")


@banner("Setup Vulkan {version}")
def setup_vulkan(version: str = _VULKAN_VERSION):
    u = platform.uname()
    if u.system == "Linux":
        url = f"https://sdk.lunarg.com/sdk/download/{version}/linux/vulkansdk-linux-x86_64-{version}.tar.xz"
        prefix = get_cache_home() / f"vulkan-{version}"
        download_dep(url, prefix, strip=1)
        sdk = prefix / "x86_64"
        os.environ["VULKAN_SDK"] = str(sdk)
        path_prepend("PATH", sdk / "bin")
        path_prepend("LD_LIBRARY_PATH", sdk / "lib")
        os.environ["VK_LAYER_PATH"] = str(sdk / "share" / "vulkan" / "explicit_layer.d")
    # elif (u.system, u.machine) == ("Darwin", "arm64"):
    # elif (u.system, u.machine) == ("Darwin", "x86_64"):
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        # LunarG renamed the Windows installer to "VulkanSDK-<ver>-Installer.exe"
        # some time around the 1.4 series; the older "vulkansdk-windows-X64-*.exe"
        # naming returns 404 for all versions now.
        url = f"https://sdk.lunarg.com/sdk/download/{version}/windows/VulkanSDK-{version}-Installer.exe"
        prefix = get_cache_home() / f"vulkan-{version}"
        download_dep(
            url,
            prefix,
            elevate=True,
            args=[
                "--accept-licenses",
                "--default-answer",
                "--confirm-command",
                "--root",
                prefix,
                "install",
                "com.lunarg.vulkan.sdl2",
                "com.lunarg.vulkan.glm",
                "com.lunarg.vulkan.volk",
                "com.lunarg.vulkan.vma",
                # 'com.lunarg.vulkan.debug',
            ],
        )
        os.environ["VULKAN_SDK"] = str(prefix)
        os.environ["VK_SDK_PATH"] = str(prefix)
        os.environ["VK_LAYER_PATH"] = str(prefix / "Bin")
        path_prepend("PATH", prefix / "Bin")
    else:
        return
