"""Larger profile: N kernels, each with many statements. Targets the 'many kernel cold
compile' scenario where P3.d FFI batch was hypothesized to help.
"""
from __future__ import annotations
import cProfile, pstats, io, os, time

import taichi as ti


def _build_many(x, n_kernels: int, body_stmts: int):
    kernels = []
    for j in range(n_kernels):
        def make(j=j):
            @ti.kernel
            def k():
                for i in range(x.shape[0]):
                    a = x[i] + 1.0
                    # body_stmts-1 chained ops
                    for _ in range(1):
                        pass
                    b = a * 2.0 + x[i]
                    c = b - a * 0.5
                    d = c + b * 0.25
                    e = d - c * 0.1
                    f = e + d * 0.05
                    g = f * 2.0 - e
                    h = g + f - a
                    i1 = h + a * 1.25
                    i2 = i1 - h * 0.8
                    i3 = i2 + i1 * 0.4
                    i4 = i3 - i2 * 0.2
                    i5 = i4 + i3 * 0.1
                    i6 = i5 - i4 * 0.05
                    x[i] = a + b + c + d + e + f + g + h + i1 + i2 + i3 + i4 + i5 + i6 + float(j)
            return k
        kernels.append(make())
    return kernels


def run_once(label: str, n_kernels: int):
    ti.init(arch=ti.cpu, offline_cache=False)
    x = ti.field(ti.f32, shape=1024)
    ks = _build_many(x, n_kernels, 16)
    t0 = time.perf_counter()
    for k in ks:
        k()
    dt = time.perf_counter() - t0
    ti.reset()
    print(f"[{label}] {n_kernels} kernels cold compile+launch = {dt*1000:.1f} ms")
    return dt


def main():
    # warm
    run_once("warm", 2)

    pr = cProfile.Profile()
    pr.enable()
    run_once("profiled", 16)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(30)
    out_tt = buf.getvalue()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
    out_cu = buf.getvalue()

    with open("tests/p3/profile_p3d.txt", "w", encoding="utf-8") as f:
        f.write("## tottime\n"); f.write(out_tt)
        f.write("\n## cumulative\n"); f.write(out_cu)

    for line in out_tt.splitlines()[:35]:
        print(line)


if __name__ == "__main__":
    main()
