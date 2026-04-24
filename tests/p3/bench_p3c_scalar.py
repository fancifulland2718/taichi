"""P3.c — early-exit scalarize bench.

Pure scalar kernels (no TensorType stmts) should now skip all five scalarize
sub-passes. Measures cold-compile wall-clock on a saxpy-class kernel across
3 kernel sizes, fresh subprocess per row.
"""
import subprocess
import sys
import time
import os
import tempfile


RUNNER = r'''
import time
import taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)

N = 1 << 20
a = ti.field(ti.f32, shape=N)
b = ti.field(ti.f32, shape=N)
c = ti.field(ti.f32, shape=N)

@ti.kernel
def saxpy(alpha: ti.f32):
    # Scalar-only — no matrices. Each static(for) expansion simply emits
    # extra scalar arithmetic so the IR grows but stays TensorType-free.
    for i in a:
        x = a[i]
        y = b[i]
        z = ti.f32(0.0)
        for k in ti.static(range({UNROLL})):
            z += alpha * x + float(k) * y
        c[i] = z

t0 = time.perf_counter()
saxpy(1.5)
ti.sync()
print(f"DT={{time.perf_counter()-t0:.4f}}")
'''


def timed(unroll):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                      encoding="utf-8") as f:
        f.write(RUNNER.format(UNROLL=unroll))
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", path],
                           capture_output=True, text=True, encoding="utf-8")
        for line in p.stdout.splitlines():
            if line.startswith("DT="):
                return float(line[3:])
        print("STDERR:", p.stderr[:800])
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return None


def main():
    print(f"{'unroll':>7} {'compile dt (s)':>16}")
    for N in [32, 128, 512]:
        dt = timed(N)
        print(f"{N:>7} {dt:>16.4f}")


if __name__ == "__main__":
    main()
