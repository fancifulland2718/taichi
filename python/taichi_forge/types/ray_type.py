"""Kernel argument annotations for hardware acceleration structures."""


class _AccelerationStructureResource:
    """Resource-free marker for provider-owned, typed Graph bindings."""


class AccelerationStructureType:
    """A read-only Vulkan top-level acceleration structure kernel argument."""


acceleration_structure = AccelerationStructureType

__all__ = ["acceleration_structure"]
