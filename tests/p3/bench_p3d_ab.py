"""A/B bench for P3.d: cold compile time for N kernels. Compare pre/post-fix."""
from __future__ import annotations
import os, sys, time, subprocess, json


CHILD = r"""
import sys, time, taichi as ti
n = int(sys.argv[1])
ti.init(arch=ti.cpu, offline_cache=False)
x = ti.field(ti.f32, shape=1024)
ks = []
for j in range(n):
    def make(j=j):
        @ti.kernel
        def k():
            for i in range(x.shape[0]):
                a = x[i] + 1.0
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
    ks.append(make())
t0 = time.perf_counter()
for k in ks:
    k()
dt = time.perf_counter() - t0
print(f"N={n} dt={dt:.4f}")
"""


def one(n: int) -> float:
    p = subprocess.run([sys.executable, "tests/p3/_p3d_child.py", str(n)], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr); raise SystemExit(p.returncode)
    for line in p.stdout.splitlines():
        if line.startswith("N="):
            return float(line.split("dt=")[1])
    raise RuntimeError(p.stdout)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "A"
    rows = []
    for n in (4, 8, 16, 32):
        # 2 cold runs, take min for stability
        times = [one(n) for _ in range(2)]
        rows.append((n, min(times), max(times)))
        print(f"  n={n} min={min(times):.4f} max={max(times):.4f}")
    with open(f"tests/p3/bench_p3d_{label}.txt", "w", encoding="utf-8") as f:
        f.write(f"# label={label}\n")
        for n, mn, mx in rows:
            f.write(f"N={n} min={mn:.4f} max={mx:.4f}\n")


if __name__ == "__main__":
    main()
