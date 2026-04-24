"""Quick A/B: LLVM opt level impact on cold compile time (CPU, N kernels).
No code changes — just drives ti.init(llvm_opt_level=N).
"""
import sys, subprocess, time
CHILD_TMPL = """
import sys, time, taichi as ti
n, lvl = int(sys.argv[1]), int(sys.argv[2])
ti.init(arch=ti.cpu, offline_cache=False, llvm_opt_level=lvl)
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
# final value sanity
ti.sync()
print(f"N={n} lvl={lvl} dt={time.perf_counter()-t0:.4f} head={x.to_numpy()[:4].tolist()}")
"""


def one(n, lvl):
    with open("tests/p3/_optlvl_child.py","w") as f: f.write(CHILD_TMPL)
    import subprocess
    p = subprocess.run([sys.executable, "tests/p3/_optlvl_child.py", str(n), str(lvl)],
                       capture_output=True, text=True, timeout=180)
    if p.returncode != 0:
        print(p.stdout); print(p.stderr); raise SystemExit(p.returncode)
    for line in p.stdout.splitlines():
        if line.startswith("N="):
            return line
    raise RuntimeError(p.stdout)


def main():
    for lvl in (3, 2, 1, 0):
        for n in (8, 16, 32):
            # two runs, take min
            r = [one(n, lvl) for _ in range(2)]
            print(f"  lvl={lvl} {r[0]} | {r[1]}")


if __name__ == "__main__":
    main()
