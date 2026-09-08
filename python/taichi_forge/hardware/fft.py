"""Public optional CUDA and Vulkan FFT plans.

``is_available()`` and ``cache_statistics()`` retain their cuFFT-only meaning.
Use ``ti.hardware.probe('vkfft', library_path=...)`` to inspect the Vulkan addon.
"""

from taichi_forge.hardware._vulkan_fft import VulkanFftPlan
from taichi_forge.hardware._vulkan_fft_recipe import VulkanFftRecipeProvider

from taichi_forge.hardware._fft_recipe import FftRecipeProvider
from taichi_forge.hardware._fft import (
    CufftLayout,
    CufftPlan1D,
    CufftPlanCacheStatistics,
    CufftPlanND,
    CufftRecording,
    cache_statistics,
    is_available,
)

__all__ = [
    "VulkanFftPlan",
    "VulkanFftRecipeProvider",
    "FftRecipeProvider",
    "CufftLayout",
    "CufftPlan1D",
    "CufftPlanND",
    "CufftPlanCacheStatistics",
    "CufftRecording",
    "cache_statistics",
    "is_available",
]
