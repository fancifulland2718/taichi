"""Private cuDSS recipe preparation; no raw policy in the public search API."""

from collections.abc import Mapping

_REORDERINGS = ("default", "amd", "nested_dissection", "natural")
_SOLVES = ("default", "general")


def canonical_configuration(configuration):
    if not isinstance(configuration, Mapping) or set(configuration) != {
        "version",
        "reordering",
        "solve",
    }:
        raise ValueError("cuDSS configuration needs version, reordering and solve")
    if type(configuration["version"]) is not int or configuration["version"] != 1:
        raise ValueError("Unsupported cuDSS private configuration version")
    if configuration["reordering"] not in _REORDERINGS:
        raise ValueError("Unsupported cuDSS reordering policy")
    if configuration["solve"] not in _SOLVES:
        raise ValueError("Unsupported cuDSS solve policy")
    return {
        "version": 1,
        "reordering": configuration["reordering"],
        "solve": configuration["solve"],
    }


def configured_plan(matrix, configuration, *, _graph_owned=False, **kwargs):
    from taichi_forge.hardware._linalg import CudssPlan

    configuration = canonical_configuration(configuration)
    encoded = (
        _REORDERINGS.index(configuration["reordering"]),
        _SOLVES.index(configuration["solve"]),
    )
    plan = CudssPlan._create_configured(
        matrix, encoded, _graph_owned=_graph_owned, **kwargs
    )
    try:
        if plan._configuration_report()["configuration"] != configuration:
            raise ValueError("cuDSS frozen configuration drifted during preparation")
        return plan
    except BaseException:
        plan.close()
        raise


def configuration_report(facts):
    """Translate cached analysis estimates, never manufacture observed memory."""
    facts = dict(facts)
    configuration = None
    if facts["configuration_abi"] == 1:
        configuration = {
            "version": 1,
            "reordering": _REORDERINGS[facts["reordering"]],
            "solve": _SOLVES[facts["solve"]],
        }
    estimates = {
        name: None if facts[f"estimated_{name}"] < 0 else facts[f"estimated_{name}"]
        for name in (
            "device_persistent_bytes",
            "device_peak_bytes",
            "host_persistent_bytes",
            "host_peak_bytes",
        )
    }
    status = facts["memory_estimates_status"]
    return {
        "configuration": configuration,
        "resolved_default_algorithm": None,
        "analysis_memory_estimates": {
            "kind": "vendor_estimate_not_observation",
            "scope": "analyzed_pattern_frozen_configuration_single_rhs",
            "status": (
                "not_queried"
                if status < 0
                else (
                    "available"
                    if all(v is not None for v in estimates.values())
                    else "unavailable"
                )
            ),
            "vendor_status": None if status < 0 else status,
            "written_bytes": facts["memory_estimates_written_bytes"],
            **estimates,
        },
        "observed_device_peak_bytes": None,
        "graph_owned": bool(facts.get("graph_owned", 0)),
        "graph_allocator": (
            {
                "kind": "requested_payload_not_driver_residency",
                "live_bytes": facts["allocator_live_requested_bytes"],
                "peak_live_bytes": facts["allocator_peak_requested_bytes"],
                "snapshot_bytes": facts["graph_snapshot_bytes"],
                "sealed_allocation_rejections": facts["sealed_allocation_rejections"],
            }
            if facts.get("graph_owned", 0)
            else None
        ),
    }
