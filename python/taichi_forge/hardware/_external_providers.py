"""Internal registry for optional hardware runtime providers.

The registry describes how Forge observes and probes a provider.  It does not
install a package, select an algorithm, or keep an execution resource alive.
Those actions remain with the domain API that owns the corresponding plan or
scene.
"""

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple


@dataclass(frozen=True)
class ExternalProviderSpec:
    provider_id: str
    adapter_kind: str
    install_owner: str
    library_path_policy: str
    process_handle_policy: str
    runtime_resource_policy: str
    transitive_dependencies: Tuple[str, ...]
    python_adapter_module: Optional[str] = None
    native_path_resolver: Optional[str] = None

    @property
    def supports_library_path(self):
        return self.library_path_policy in ("optional", "required")


_EXTERNAL_PROVIDER_SPECS = (
    ExternalProviderSpec(
        provider_id="cusolverdn",
        adapter_kind="python_dynamic_symbols",
        install_owner="user_cuda_environment",
        library_path_policy="optional",
        process_handle_policy="process_resident",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cuda_runtime", "cublas", "cusparse"),
        python_adapter_module="taichi_forge.hardware._cusolverdn",
    ),
    ExternalProviderSpec(
        provider_id="vkfft",
        adapter_kind="optional_jit_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("vulkan_loader",),
        python_adapter_module="taichi_forge.hardware._vulkan_fft",
    ),
    ExternalProviderSpec(
        provider_id="cublas",
        adapter_kind="native_symbols",
        install_owner="user_cuda_environment",
        library_path_policy="implicit",
        process_handle_policy="process_resident",
        runtime_resource_policy="runtime_generation",
        transitive_dependencies=(),
    ),
    ExternalProviderSpec(
        provider_id="cublaslt",
        adapter_kind="python_dynamic_symbols",
        install_owner="user_cuda_environment",
        library_path_policy="optional",
        process_handle_policy="process_resident",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cuda_runtime",),
        python_adapter_module="taichi_forge.hardware._cublaslt",
    ),
    ExternalProviderSpec(
        provider_id="cusparse",
        adapter_kind="native_symbols",
        install_owner="user_cuda_environment",
        library_path_policy="implicit",
        process_handle_policy="process_resident",
        runtime_resource_policy="runtime_generation",
        transitive_dependencies=(),
    ),
    ExternalProviderSpec(
        provider_id="cufft",
        adapter_kind="native_symbols",
        install_owner="user_cuda_environment",
        library_path_policy="implicit",
        process_handle_policy="process_resident",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=(),
    ),
    ExternalProviderSpec(
        provider_id="cudss",
        adapter_kind="bundled_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cublas",),
        python_adapter_module="taichi_forge.hardware._cudss",
    ),
    ExternalProviderSpec(
        provider_id="optix",
        adapter_kind="bundled_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_context",
        transitive_dependencies=("cuda_driver", "optix_driver_runtime"),
        python_adapter_module="taichi_forge.hardware._optix",
    ),
    ExternalProviderSpec(
        provider_id="cusparselt",
        adapter_kind="bundled_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cuda_runtime",),
        python_adapter_module="taichi_forge.hardware._cusparselt",
    ),
    ExternalProviderSpec(
        provider_id="cutensor",
        adapter_kind="bundled_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cuda_runtime",),
        python_adapter_module="taichi_forge.hardware._cutensor",
    ),
    ExternalProviderSpec(
        provider_id="amgx",
        adapter_kind="bundled_provider_c_abi",
        install_owner="forge_runtime_wheel",
        library_path_policy="optional",
        process_handle_policy="provider_object",
        runtime_resource_policy="provider_plan",
        transitive_dependencies=("cuda_runtime", "cublas", "cusparse"),
        python_adapter_module="taichi_forge.hardware._amgx",
    ),
)

_EXTERNAL_PROVIDER_SPECS_BY_ID = MappingProxyType({spec.provider_id: spec for spec in _EXTERNAL_PROVIDER_SPECS})


def external_provider_specs():
    return _EXTERNAL_PROVIDER_SPECS


def external_provider_ids():
    return tuple(spec.provider_id for spec in _EXTERNAL_PROVIDER_SPECS)


def external_provider_spec(provider_id):
    if not isinstance(provider_id, str) or not provider_id:
        raise TypeError("provider_id must be a nonempty string")
    try:
        return _EXTERNAL_PROVIDER_SPECS_BY_ID[provider_id]
    except KeyError as exc:
        raise KeyError(f"unknown external hardware provider: {provider_id}") from exc


def probe_external_provider(provider_id, library_path=None):
    """Transiently inspect one provider without selecting or retaining it."""

    spec = external_provider_spec(provider_id)
    if library_path is not None and not spec.supports_library_path:
        raise ValueError(f"library_path is not supported for {provider_id} provider probes")
    if spec.python_adapter_module is not None:
        adapter = importlib.import_module(spec.python_adapter_module)
        return dict(adapter.probe_provider(library_path))

    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

    if spec.native_path_resolver is not None:
        raise RuntimeError(f"unknown native path resolver for {provider_id}: " f"{spec.native_path_resolver}")
    return dict(_ti_core.probe_cuda_external_library(provider_id))


def passive_external_provider_status(provider_id):
    """Read already-loaded state without opening a provider library."""

    spec = external_provider_spec(provider_id)
    if spec.python_adapter_module is not None:
        adapter = importlib.import_module(spec.python_adapter_module)
        return dict(adapter.passive_status())

    from taichi_forge._lib import core as _ti_core  # pylint: disable=C0415

    status = getattr(_ti_core, "cuda_external_library_status", None)
    if status is None:
        return {
            "provider_id": provider_id,
            "library_loaded": False,
            "provider_abi": None,
            "provider_version": None,
            "native_facts": {
                "status_policy": "passive_status_unavailable",
                "external_component_probed": False,
                "provider_enablement_changed": False,
                "provider_selection_changed": False,
            },
        }
    return dict(status(provider_id))


def external_provider_library_loaded(provider_id):
    status = passive_external_provider_status(provider_id)
    return bool(status["library_loaded"])


__all__ = [
    "ExternalProviderSpec",
    "external_provider_ids",
    "external_provider_library_loaded",
    "external_provider_spec",
    "external_provider_specs",
    "passive_external_provider_status",
    "probe_external_provider",
]
