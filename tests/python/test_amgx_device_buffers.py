"""Optional real-vendor numeric checks; unit ABI fixtures cannot replace these."""

import json
import os

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.skipif(
    not os.environ.get("TI_FORGE_TEST_AMGX_LIBRARY_PATH"),
    reason="set TI_FORGE_TEST_AMGX_LIBRARY_PATH for real AmgX device-buffer tests",
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("rows,dtype", [(1024, np.float32), (32768, np.float64)])
def test_amgx_device_producers_updates_and_solution_consumer(rows, dtype):
    scalar = ti.f32 if dtype == np.float32 else ti.f64
    offsets = np.empty(rows + 1, dtype=np.int32)
    offsets[0] = 0
    offsets[1:-1] = 3 * np.arange(1, rows, dtype=np.int32) - 1
    offsets[-1] = 3 * rows - 2
    columns = np.concatenate([np.arange(max(0, i - 1), min(rows, i + 2), dtype=np.int32) for i in range(rows)])
    values = ti.ndarray(scalar, columns.size)
    rhs = ti.ndarray(scalar, rows)
    solution = ti.ndarray(scalar, rows)
    output = ti.ndarray(scalar, rows)

    @ti.kernel
    def produce(values: ti.types.ndarray(), rhs: ti.types.ndarray(), diagonal: ti.f64, scale: ti.f64):
        for i in range(rows):
            start = 3 * i - ti.cast(i > 0, ti.i32)
            center = start + ti.cast(i > 0, ti.i32)
            values[center] = diagonal
            # The manufactured solution must not round through default_fp=f32
            # before promotion by scale, especially for the f64 vendor check.
            step = ti.cast(1, ti.f64) / 1000
            x = scale * (1 + step * (i % 31))
            b = diagonal * x
            if i > 0:
                values[start] = -1
                b -= scale * (1 + step * ((i - 1) % 31))
            if i + 1 < rows:
                values[center + 1] = -1
                b -= scale * (1 + step * ((i + 1) % 31))
            rhs[i] = b

    @ti.kernel
    def consume(solution: ti.types.ndarray(), output: ti.types.ndarray()):
        for i in range(rows):
            output[i] = solution[i] * 2

    config = json.dumps(
        {
            "config_version": 2,
            "solver": {
                "solver": "PCG",
                "preconditioner": {"solver": "NOSOLVER"},
                "max_iters": 100,
                "monitor_residual": 1,
                "convergence": "RELATIVE_INI",
                "norm": "L2",
                "tolerance": 1e-6 if dtype == np.float32 else 1e-11,
                "print_solve_stats": 0,
                "obtain_timings": 0,
            },
        }
    )
    produce(values, rhs, 4.0, 1.0)
    solution.fill(0)
    with ti.hardware.linalg.AmgxProvider(os.environ["TI_FORGE_TEST_AMGX_LIBRARY_PATH"]) as provider:
        with provider.solver(offsets, columns, values, config) as solver:
            binding = solver.bind_device(rhs, solution, values=values, zero_initial_guess=False)
            for iteration in range(3):
                diagonal, scale = 4.0 + iteration, 1.0 + iteration
                produce(values, rhs, diagonal, scale)
                if iteration:
                    binding.replace_coefficients()
                result, info = binding.solve()
                assert result is solution and info["converged"]
                consume(solution, output)  # No user synchronization between vendor output and GPU consumer.
                expected = 2 * scale * (1.0 + 0.001 * (np.arange(rows) % 31))
                tolerance = 2e-5 if dtype == np.float32 else 1e-9
                np.testing.assert_allclose(output.to_numpy(), expected, rtol=tolerance, atol=tolerance)
        with pytest.raises(ti.TaichiRuntimeError, match="closed"):
            binding.solve()
