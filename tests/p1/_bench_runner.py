"""Subprocess helper for bench_cold_start.py.

Reads env vars _BENCH_KERNEL and _BENCH_ARCH, runs the named kernel once,
and prints the elapsed milliseconds to stdout.  Die with returncode != 0 on
error so the parent process can detect failures.

This helper must be a separate file so each invocation gets a fresh Python
interpreter (and fresh Taichi state / LLVM JIT — no cross-contamination).
"""

import importlib
import os
import sys
import time

# ------------------------------------------------------------------ main ---
kernel_name = os.environ.get("_BENCH_KERNEL", "")
arch_str = os.environ.get("_BENCH_ARCH", "cpu")

if not kernel_name:
    print("_BENCH_KERNEL not set", file=sys.stderr)
    sys.exit(1)

# Locate the kernel factories defined in bench_cold_start.py
sys.path.insert(0, os.path.dirname(__file__))
import bench_cold_start as _bcs  # noqa: E402

spec = next((s for s in _bcs.KERNELS if s["name"] == kernel_name), None)
if spec is None:
    print(f"Unknown kernel: {kernel_name!r}", file=sys.stderr)
    sys.exit(1)

ti = importlib.import_module("taichi")
arch = getattr(ti, arch_str, ti.cpu)

# No offline cache → guaranteed cold compile.
ti.init(arch=arch, offline_cache=False, log_level="error")

kernel_fn = spec["factory"](ti)

t0 = time.perf_counter()
kernel_fn()
ti.sync()
t1 = time.perf_counter()

print(f"{(t1 - t0) * 1000.0:.3f}")
