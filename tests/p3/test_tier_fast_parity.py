"""Correctness test for compile_tier="fast" on LLVM backends.

Verifies that tier=fast (which caps llvm_opt_level at 0) produces
numerically close results to tier=balanced (default O3) across the
P3.c matrix paths and a basic multi-kernel workload.
"""
import math
import sys
from pathlib import Path

import taichi as ti


def run_matrix(tier):
    ti.init(arch=ti.cpu, offline_cache=False, compile_tier=tier, random_seed=0)
    M = ti.Matrix.field(3, 3, ti.f64, shape=())
    @ti.kernel
    def init():
        for i, j in ti.static(ti.ndrange(3, 3)):
            M[None][i, j] = ti.cast(i * 3 + j + 1, ti.f64) * 0.1
    @ti.kernel
    def sq() -> ti.f64:
        N = M[None] @ M[None]
        s = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            s += N[i, j]
        return s
    init()
    return sq()


def run_seq(tier, N):
    ti.init(arch=ti.cpu, offline_cache=False, compile_tier=tier, random_seed=0)
    x = ti.field(ti.f64, shape=128)

    def make(idx):
        s = float(idx)
        @ti.kernel
        def k():
            for i in x:
                a = ti.f64(i) + s
                x[i] += a * 1.1 + 0.25
        return k
    ks = [make(i) for i in range(N)]
    for k in ks:
        k()
    return [x[i] for i in range(4)]


if __name__ == "__main__":
    a = run_matrix("balanced")
    b = run_matrix("fast")
    print(f"matrix: balanced={a!r}  fast={b!r}  abs_delta={abs(a-b):.3e}")
    assert abs(a - b) < 1e-9, "matrix path: tier=fast produced different result"

    a = run_seq("balanced", 8)
    b = run_seq("fast", 8)
    delta = max(abs(x - y) for x, y in zip(a, b))
    print(f"seq(8):  balanced head={a!r}")
    print(f"         fast     head={b!r}")
    print(f"         max abs delta={delta:.3e}")
    assert delta < 1e-6, "seq path: tier=fast drifted too much"

    print("OK — tier=fast numerically compatible with tier=balanced on these kernels.")
