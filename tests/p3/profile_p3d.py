"""Profile the Python-side compile path to locate real FFI / AST-transformer hotspots.

Drives a cold compile of a moderately complex kernel (many ops + nested loops) and
captures cProfile stats. The aim is to decide whether P3.d (FFI batching) has a
non-trivial target before writing any code.
"""
from __future__ import annotations

import cProfile
import pstats
import io
import os
import time

# Ensure cold offline_cache
os.environ.setdefault("TI_OFFLINE_CACHE", "0")

import taichi as ti


def _build_kernel(x):
    @ti.kernel
    def k():
        # Many statements so AST transform is meaningful but kernel still compiles fast.
        for i in range(x.shape[0]):
            a = x[i] + 1.0
            b = a * 2.0 + x[i]
            c = b - a * 0.5
            d = c + b * 0.25
            e = d - c * 0.1
            f = e + d * 0.05
            g = f * 2.0 - e
            h = g + f - a
            x[i] = a + b + c + d + e + f + g + h
    return k


def run_once(label: str):
    ti.init(arch=ti.cpu, offline_cache=False)
    x = ti.field(ti.f32, shape=1024)
    k = _build_kernel(x)
    t0 = time.perf_counter()
    k()          # forces compile
    dt = time.perf_counter() - t0
    ti.reset()
    print(f"[{label}] cold compile+launch = {dt*1000:.1f} ms")
    return dt


def main():
    # Warm once so top-level imports / prog.materialize() don't pollute the cProfile.
    run_once("warmup")

    pr = cProfile.Profile()
    pr.enable()
    run_once("profiled")
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(40)
    out1 = buf.getvalue()

    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=buf2).sort_stats("tottime")
    ps2.print_stats(40)
    out2 = buf2.getvalue()

    with open("tests/p3/profile_p3d.txt", "w", encoding="utf-8") as f:
        f.write("# cProfile cumulative\n")
        f.write(out1)
        f.write("\n\n# cProfile tottime\n")
        f.write(out2)
    # Print short summary
    print(out2.splitlines()[0:45][-40:])


if __name__ == "__main__":
    main()
