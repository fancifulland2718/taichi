"""Private FFT materialization: batch tails, scratch reuse and retained commands."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware import _vulkan_fft
from tests import test_utils
from tests.python.test_hardware_vulkan_fft import _adapter, _complex, _input


@pytest.mark.parametrize(
    "dimensions,batches,tile", [((16, 8), 5, 2), ((4, 4, 4), 4, 1)]
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_partition_preserves_batch_semantics_and_retains_commands(
    dimensions, batches, tile, monkeypatch
):
    data, values = _input(dimensions, batches)
    other, _ = _input(dimensions, batches)
    options = dict(batch_count=batches, adapter_path=_adapter())
    baseline = _vulkan_fft.VulkanFftPlan(data, dimensions, **options)
    with _vulkan_fft.VulkanFftPlan(
        other, dimensions, _batch_tile=batches, **options
    ) as unpartitioned:
        # The optional table must not manufacture a new physical identity
        # when the complete transform is the same as legacy ABI1 creation.
        assert unpartitioned._physical_id == baseline._physical_id
    partition = _vulkan_fft.VulkanFftPlan(data, dimensions, _batch_tile=tile, **options)
    equivalent = _vulkan_fft.VulkanFftPlan(
        other, dimensions, _batch_tile=tile, **options
    )
    inverse = _vulkan_fft.VulkanFftPlan(
        data,
        dimensions,
        _batch_tile=tile,
        direction="inverse",
        normalization="inverse",
        **options
    )
    try:
        facts = partition.statistics()
        assert facts["recipe_extension_abi"] == 1
        assert facts["application_count"] == (2 if batches % tile else 1)
        assert facts["batch_tile"] == tile
        assert facts["dispatch_count"] > baseline.statistics()["dispatch_count"]
        assert (
            facts["dispatch_fingerprint"]
            != baseline.statistics()["dispatch_fingerprint"]
        )
        assert (
            partition._semantic_id == baseline._semantic_id == equivalent._semantic_id
        )
        assert partition._physical_id != baseline._physical_id
        assert partition._physical_id == equivalent._physical_id
        assert facts["shader_module_count"] > 0
        expected = np.fft.fftn(_complex(values), axes=tuple(range(-len(dimensions), 0)))

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "FFT replay must not prepare, observe or discover plans"
            )

        monkeypatch.setattr(partition, "statistics", forbidden)
        monkeypatch.setattr(partition, "memory_report", forbidden)
        monkeypatch.setattr(partition, "validate_graph_lifetime", forbidden)
        monkeypatch.setattr(_vulkan_fft, "_adapter_path", forbidden)
        monkeypatch.setattr(_vulkan_fft, "probe_provider", forbidden)
        partition.run()
        np.testing.assert_allclose(
            _complex(data.to_numpy()), expected, atol=3e-5, rtol=3e-5
        )
        inverse.run()
        # Retirement, not the Python open-plan registry, owns pending commands.
        inverse.close()
        partition.close()
        np.testing.assert_allclose(data.to_numpy(), values, atol=3e-6, rtol=4e-5)
        # The second plan must be independent of the first plan's retirement.
        equivalent.run()
        np.testing.assert_allclose(
            _complex(other.to_numpy()), expected, atol=3e-5, rtol=3e-5
        )
        with pytest.raises(RuntimeError, match="closed"):
            partition.run()
    finally:
        for plan in (baseline, partition, equivalent, inverse):
            plan.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_partition_accounts_workspace_and_normalizes_multistage_fft():
    dimensions, batches, tile = (262144,), 5, 2
    data, values = _input(dimensions, batches)
    options = dict(batch_count=batches, adapter_path=_adapter())
    with _vulkan_fft.VulkanFftPlan(data, dimensions, **options) as baseline:
        before = baseline.statistics()
        with _vulkan_fft.VulkanFftPlan(
            data, dimensions, _batch_tile=tile, **options
        ) as partition:
            facts = partition.statistics()
            assert facts["application_count"] == 2
            assert facts["dispatch_count"] > before["dispatch_count"]
            assert facts["temporary_buffer_bytes"] <= before["temporary_buffer_bytes"]
            if before["temporary_buffer_bytes"]:
                assert (
                    facts["temporary_buffer_bytes"] < before["temporary_buffer_bytes"]
                )
            report = partition.memory_report()
            assert (
                report.known_resident_requested_bytes
                == facts["persistent_allocation_bytes"]
            )
            assert not report.resident_requested_bytes_complete
            partition.run()
            np.testing.assert_allclose(
                _complex(data.to_numpy()),
                np.fft.fft(_complex(values)),
                atol=0.0004,
                rtol=0.0001,
            )
        with _vulkan_fft.VulkanFftPlan(
            data,
            dimensions,
            _batch_tile=tile,
            direction="inverse",
            normalization="inverse",
            **options
        ) as inverse:
            inverse.run()
            np.testing.assert_allclose(data.to_numpy(), values, atol=3e-6, rtol=5e-5)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_fft_partition_rejects_invalid_configuration_before_native_creation():
    data, _ = _input((8,), 3)
    # No adapter discovery or shader compilation is needed for these failures.
    for tile in (0, -1, True, 1.5, 4):
        with pytest.raises(ValueError, match="batch tile"):
            _vulkan_fft.VulkanFftPlan(data, (8,), batch_count=3, _batch_tile=tile)
