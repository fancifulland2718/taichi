"""A/B bench: compile_tier="balanced" (default, O3) vs "fast" (capped at O0).
CPU backend. N=16 kernels, 2 runs each, min wall time.
Also checks numeric drift (first 4 values).
"""
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
CHILD = HERE / "_optlvl_child.py"

# ensure child exists
if not CHILD.exists():
    CHILD.write_text((HERE / "_p3d_child.py").read_text())


def run_one(tier: str, N: int) -> tuple[float, str]:
    # We use _p3d_child.py but inject TI_TIER env var. Simpler: new tiny child.
    src = """
import sys, time
import taichi as ti
tier = sys.argv[1]
N = int(sys.argv[2])
ti.init(arch=ti.cpu, offline_cache=False, compile_tier=tier)

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
for k in ks:
    k()
dt = time.perf_counter() - t0
print(f"DT={dt:.4f} HEAD={[x[j] for j in range(4)]}")
"""
    tmp = HERE / "_tier_child.py"
    tmp.write_text(src)
    p = subprocess.run([sys.executable, str(tmp), tier, str(N)],
                       capture_output=True, text=True, timeout=300)
    last = [ln for ln in p.stdout.splitlines() if ln.startswith("DT=")][-1]
    dt = float(last.split()[0].split("=")[1])
    head = last.split("HEAD=", 1)[1]
    return dt, head


if __name__ == "__main__":
    N = 16
    results = {}
    for tier in ["balanced", "fast"]:
        times = []
        head = None
        for _ in range(2):
            dt, h = run_one(tier, N)
            times.append(dt)
            head = h
            print(f"  tier={tier} dt={dt:.4f}")
        best = min(times)
        results[tier] = (best, head)
        print(f"tier={tier} best={best:.4f} head={head}")
    bal, fast = results["balanced"][0], results["fast"][0]
    save = (bal - fast) / bal * 100
    print(f"\nN={N} balanced={bal:.4f}s fast={fast:.4f}s save={save:+.1f}%")
    print(f"balanced head={results['balanced'][1]}")
    print(f"fast     head={results['fast'][1]}")
