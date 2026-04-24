"""A/B compile-time bench on CUDA: tier=balanced vs tier=fast."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent

SRC = r"""
import sys, time
import taichi as ti
tier = sys.argv[1]; N = int(sys.argv[2])
ti.init(arch=ti.cuda, offline_cache=False, compile_tier=tier)
x = ti.field(ti.f64, shape=256)
def make(idx):
    s = float(idx)
    @ti.kernel
    def k():
        for i in x:
            a = ti.f64(i) + s
            b = a * 1.0001 + 0.5
            c = b * b - a
            d = c * 2.0 + b
            e = d - a * 0.5
            f = e * 1.3
            g = f + c - a
            h = g * 0.7 + d
            x[i] += h
    return k
t0 = time.perf_counter()
ks = [make(i) for i in range(N)]
for k in ks: k()
dt = time.perf_counter() - t0
print(f"DT={dt:.4f} HEAD={[x[j] for j in range(4)]}")
"""


def run_one(tier, N):
    tmp = HERE / "_tier_cuda_child.py"
    tmp.write_text(SRC)
    p = subprocess.run([sys.executable, str(tmp), tier, str(N)],
                       capture_output=True, text=True, timeout=300)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("DT=")]
    if not lines:
        print("CHILD STDOUT:", p.stdout)
        print("CHILD STDERR:", p.stderr)
        raise RuntimeError("child failed")
    last = lines[-1]
    dt = float(last.split()[0].split("=")[1])
    head = last.split("HEAD=", 1)[1]
    return dt, head


if __name__ == "__main__":
    N = 16
    res = {}
    for tier in ["balanced", "fast"]:
        times = []; head = None
        for _ in range(2):
            dt, head = run_one(tier, N)
            times.append(dt)
            print(f"  CUDA tier={tier} dt={dt:.4f}")
        res[tier] = (min(times), head)
    bal, fast = res["balanced"][0], res["fast"][0]
    save = (bal - fast) / bal * 100
    print(f"\nCUDA N={N} balanced={bal:.4f}s fast={fast:.4f}s save={save:+.1f}%")
    print(f"balanced head={res['balanced'][1]}")
    print(f"fast     head={res['fast'][1]}")
