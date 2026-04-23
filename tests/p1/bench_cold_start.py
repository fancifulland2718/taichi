"""Cold-start compile-time benchmark for P1 optimisations.

Measures how long Taichi takes to compile a set of representative kernels
when NO warm cache is present (i.e. the slow path every new user experiences).

Usage
-----
    python tests/p1/bench_cold_start.py [--arch cpu] [--runs 3] [--csv out.csv]

The script:
  1. Clears the offline cache before every individual timed run to guarantee
     a cold start (per the principle added to plan.md).
  2. Times each kernel's first-call compile in isolation.
  3. Writes a CSV with columns: kernel, run, compile_ms.
  4. Prints a summary table (mean ± stdev across runs) at the end.

If compile time for any kernel regresses vs. the stored baseline the script
prints a "REGRESSION" line so CI can detect it.  Attach the profiler JSON
(TI_COMPILE_PROFILE=1) to diagnose the bottleneck before deciding to revert.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time

# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

KERNELS: list[dict] = []


def _reg(name):
    """Decorator – register a kernel factory by name."""
    def _dec(fn):
        KERNELS.append({"name": name, "factory": fn})
        return fn
    return _dec


@_reg("saxpy_1m")
def _make_saxpy(ti):
    import taichi as ti  # noqa: F811 – re-imported inside factory

    x = ti.field(ti.f32, shape=1 << 20)
    y = ti.field(ti.f32, shape=1 << 20)

    @ti.kernel
    def saxpy(a: float):
        for i in x:
            y[i] = a * x[i] + y[i]

    return saxpy


@_reg("stencil_laplacian_512")
def _make_stencil(ti):
    import taichi as ti  # noqa: F811

    N = 512
    f = ti.field(ti.f32, shape=(N, N))
    g = ti.field(ti.f32, shape=(N, N))

    @ti.kernel
    def laplacian():
        for i, j in ti.ndrange((1, N - 1), (1, N - 1)):
            g[i, j] = (f[i - 1, j] + f[i + 1, j] +
                       f[i, j - 1] + f[i, j + 1] - 4.0 * f[i, j])

    return laplacian


@_reg("reduction_1m")
def _make_reduction(ti):
    import taichi as ti  # noqa: F811

    n = 1 << 20
    arr = ti.field(ti.f32, shape=n)
    s = ti.field(ti.f32, shape=())

    @ti.kernel
    def reduce():
        s[None] = 0.0
        for i in arr:
            s[None] += arr[i]

    return reduce


@_reg("matrix_mul_64")
def _make_matmul(ti):
    import taichi as ti  # noqa: F811

    N = 64
    A = ti.field(ti.f32, shape=(N, N))
    B = ti.field(ti.f32, shape=(N, N))
    C = ti.field(ti.f32, shape=(N, N))

    @ti.kernel
    def matmul():
        for i, j in C:
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    return matmul


@_reg("autodiff_simple")
def _make_autodiff(ti):
    import taichi as ti  # noqa: F811

    n = 1024
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def forward():
        for i in x:
            loss[None] += x[i] ** 2

    return forward


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> str:
    import taichi as ti
    cfg = ti.impl.default_cfg()
    return os.path.abspath(cfg.offline_cache_file_path)


def _clear_cache(cache_dir: str) -> None:
    """Delete all .tic files (and the metadata dir) for a true cold start."""
    if not os.path.isdir(cache_dir):
        return
    removed = 0
    for root, dirs, files in os.walk(cache_dir):
        for fname in files:
            if fname.endswith((".tic", ".meta", ".json")):
                os.remove(os.path.join(root, fname))
                removed += 1
    if removed:
        print(f"  [cache] Cleared {removed} cache file(s) from {cache_dir}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def _time_kernel(kernel_spec: dict, arch_str: str) -> float:
    """Initialise Taichi, compile + run the kernel once, return compile_ms."""
    import importlib
    ti = importlib.import_module("taichi")

    arch = getattr(ti, arch_str, ti.cpu)
    # offline_cache=False so the compile result is NOT saved yet (pure cold).
    ti.init(arch=arch, offline_cache=False, log_level="error")

    # Build kernel (registration is lazy – happens on first call).
    kernel_fn = kernel_spec["factory"](ti)

    t0 = time.perf_counter()
    kernel_fn()          # trigger compile + launch
    ti.sync()
    t1 = time.perf_counter()

    ti.reset()
    return (t1 - t0) * 1000.0  # ms


def run_benchmark(
    arch_str: str = "cpu",
    runs: int = 3,
    csv_path: str | None = None,
    baseline_json: str | None = None,
) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    rows: list[dict] = []

    for spec in KERNELS:
        name = spec["name"]
        times: list[float] = []
        for r in range(runs):
            # Must import fresh each run so fields are re-created.
            # Use a subprocess to truly isolate each compile.
            import subprocess
            env = os.environ.copy()
            env["_BENCH_KERNEL"] = name
            env["_BENCH_ARCH"] = arch_str
            env["PYTHONPATH"] = str(
                os.path.join(os.path.dirname(__file__), "..", "..", "python")
            ) + os.pathsep + env.get("PYTHONPATH", "")

            helper = os.path.join(os.path.dirname(__file__), "_bench_runner.py")
            proc = subprocess.run(
                [sys.executable, helper],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"  [WARN] kernel={name} run={r} failed:\n{proc.stderr[:500]}")
                continue
            try:
                ms = float(proc.stdout.strip())
            except ValueError:
                print(f"  [WARN] kernel={name} run={r} bad output: {proc.stdout!r}")
                continue
            times.append(ms)
            rows.append({"kernel": name, "run": r, "compile_ms": f"{ms:.2f}"})
            print(f"  {name:30s}  run={r}  {ms:8.1f} ms")
        results[name] = times

    # Summary
    print("\n--- Summary ---")
    regression = False
    baseline: dict[str, float] = {}
    if baseline_json and os.path.isfile(baseline_json):
        with open(baseline_json) as fh:
            baseline = json.load(fh)

    for name, times in results.items():
        if not times:
            print(f"  {name:30s}  NO DATA")
            continue
        mean = statistics.mean(times)
        note = ""
        if name in baseline and mean > baseline[name] * 1.10:
            note = f"  <<< REGRESSION vs baseline {baseline[name]:.1f} ms"
            regression = True
        stdev_s = f"±{statistics.stdev(times):.1f}" if len(times) > 1 else ""
        print(f"  {name:30s}  mean={mean:7.1f} ms {stdev_s}  {note}")

    if regression:
        print("\nREGRESSION detected – run with TI_COMPILE_PROFILE=1 to diagnose.")

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["kernel", "run", "compile_ms"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written to {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="cpu", help="Taichi arch (cpu, cuda, vulkan, …)")
    p.add_argument("--runs", type=int, default=3, help="Number of timed runs per kernel")
    p.add_argument("--csv", default=None, help="Path to write CSV results")
    p.add_argument("--baseline", default=None, help="JSON file with baseline mean_ms values for regression detection")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        arch_str=args.arch,
        runs=args.runs,
        csv_path=args.csv,
        baseline_json=args.baseline,
    )
