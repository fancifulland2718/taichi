"""Explicit optional sorting plans. Ordinary ``ti.algorithms.sort`` is unchanged."""

from taichi_forge.hardware._parallel_sort import VulkanParallelSortPlan

__all__ = ["VulkanParallelSortPlan"]
