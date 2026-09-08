"""Frozen private cuDSS configuration, resource facts, and rollback contracts."""

import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware._cudss_config import configured_plan
from tests import test_utils


def test_cudss_factor_statistics_distinguish_missing_failed_and_zero():
    from taichi_forge.hardware._cudss_config import _factor_statistics

    assert _factor_statistics({})["status"] == "not_queried"
    assert _factor_statistics({"graph_owned": 1})["status"] == "unavailable"
    facts = dict(graph_owned=1, factor_statistics_queries=1, factor_statistics_result=0)
    for name, size in (
        ("factor_lu_nonzeros", 8),
        ("factor_superpanels", 4),
        ("factor_flops", 8),
    ):
        facts.update({name: 0, f"{name}_status": 0, f"{name}_written_bytes": size})
    # A zero is a reported value, not a substitute for unavailable evidence.
    assert _factor_statistics(facts)["factor_flops"]["value"] == 0
    facts["factor_lu_nonzeros"] = -1  # Adapter rejected a vendor size mismatch.
    partial = _factor_statistics(facts)
    assert partial["status"] == "partial"
    assert partial["lu_nonzeros"]["value"] is None
    facts["factor_statistics_result"] = 6
    failed = _factor_statistics(facts)
    assert failed["status"] == "unavailable"
    assert failed["factor_flops"]["value"] is None
    assert failed["factor_flops"]["vendor_status"] is None


def _problem():
    library = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not library:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to a cuDSS 0.8.x runtime")
    # An explicitly configured environment must fail, not silently skip, if
    # its adapter or dependency chain is broken.
    assert ti.hardware.linalg.cudss_is_available(library_path=library)
    row = ti.ndarray(ti.i32, 5)
    col = ti.ndarray(ti.i32, 10)
    values = ti.ndarray(ti.f32, 10)
    row.from_numpy(np.array([0, 2, 5, 8, 10], dtype=np.int32))
    col.from_numpy(np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int32))
    values.from_numpy(np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32))
    matrix = ti.linalg.SparsePattern.csr(4, 4, row, col).matrix(values)
    dense = np.array(
        [[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]],
        np.float32,
    )
    return matrix, values, dense, library


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_frozen_configuration_restore_and_cached_estimates():
    matrix, _values, dense, library = _problem()
    rhs = ti.ndarray(ti.f32, 4)
    solution = ti.ndarray(ti.f32, 4)
    rhs_np = np.arange(1, 5, dtype=np.float32)
    rhs.from_numpy(rhs_np)
    for reordering, solve in (
        ("default", "default"),
        ("amd", "general"),
        ("natural", "default"),
    ):
        spec = {"version": 1, "reordering": reordering, "solve": solve}
        with configured_plan(
            matrix, spec, matrix_type="spd", matrix_view="full", library_path=library
        ) as plan:
            before = plan._configuration_report()
            assert before["configuration"] == spec
            assert before["analysis_memory_estimates"]["status"] == "not_queried"
            assert before["preparation_factor_statistics"]["status"] == "not_queried"
            plan.compute().solve(rhs, solution)
            ti.sync()
            np.testing.assert_allclose(dense @ solution.to_numpy(), rhs_np, rtol=1e-5)
            report = json.loads(json.dumps(plan._configuration_report()))
            estimates = report["analysis_memory_estimates"]
            assert estimates["kind"] == "vendor_estimate_not_observation"
            assert estimates["status"] in ("available", "unavailable")
            if estimates["status"] == "available":
                assert estimates["device_peak_bytes"] >= 0
                assert estimates["written_bytes"] >= 48
            assert report["observed_device_peak_bytes"] is None
            assert report["resolved_default_algorithm"] is None
            assert report["preparation_factor_statistics"]["collection_count"] == 0
            rhs.from_numpy(rhs_np * 2)
            plan.solve(rhs, solution)
            ti.sync()
            assert plan._configuration_report() == report
            np.testing.assert_allclose(
                dense @ solution.to_numpy(), rhs_np * 2, rtol=1e-5
            )
            rhs.from_numpy(rhs_np)
        # Restoration recreates analysis/factors, not serialized vendor state.
        with configured_plan(
            matrix,
            report["configuration"],
            matrix_type="spd",
            matrix_view="full",
            library_path=library,
        ) as restored:
            assert not restored.statistics()["analyzed"]
            restored.compute().solve(rhs, solution)
            ti.sync()
            np.testing.assert_allclose(dense @ solution.to_numpy(), rhs_np, rtol=1e-5)
    # The old explicit constructor still uses ABI1 without the new query path.
    with ti.hardware.linalg.CudssPlan(matrix, library_path=library) as legacy:
        legacy.compute().solve(rhs, solution)
        ti.sync()
        np.testing.assert_allclose(dense @ solution.to_numpy(), rhs_np, rtol=1e-5)
        report = legacy._configuration_report()
        assert report["configuration"] is None
        assert report["analysis_memory_estimates"]["status"] == "not_queried"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_configuration_rejection_and_failed_restore_cleanup(monkeypatch):
    from taichi_forge.hardware._linalg import CudssPlan
    from taichi_forge.lang import impl

    matrix, _values, _dense, library = _problem()
    spec = {"version": 1, "reordering": "amd", "solve": "general"}
    for invalid in (dict(spec, version=True), dict(spec, solve="multiblock")):
        with pytest.raises(ValueError):
            configured_plan(matrix, invalid, library_path=library)
    with pytest.raises(RuntimeError, match="private configuration"):
        CudssPlan._create_configured(matrix, (77, 0), library_path=library)

    original_report = CudssPlan._configuration_report
    handles = []

    def drift(plan):
        handles.append(plan._handle)
        report = original_report(plan)
        report["configuration"]["reordering"] = "natural"
        return report

    with monkeypatch.context() as changed:
        changed.setattr(CudssPlan, "_configuration_report", drift)
        with pytest.raises(ValueError, match="configuration drifted"):
            configured_plan(matrix, spec, library_path=library)
    assert len(handles) == 1
    with pytest.raises(RuntimeError, match="stale or closed"):
        impl.get_runtime().prog._cuda_cudss_plan_statistics(handles[0])
    with configured_plan(matrix, spec, library_path=library) as plan:
        assert plan._configuration_report()["configuration"] == spec
