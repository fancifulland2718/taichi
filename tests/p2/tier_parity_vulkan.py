"""P1.d numeric parity on Vulkan: fast/balanced/full must produce identical
output from a math-heavy kernel (sin/cos/sqrt + rational) that would expose
any semantics-breaking spv optimizer pass."""
import os, shutil, sys


def run_one(tier: str):
    home = os.path.expanduser("~")
    p = os.path.join(home, ".cache", "taichi", "ticache")
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    import taichi as ti
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False, compile_tier=tier,
            random_seed=42, default_fp=ti.f32, log_level=ti.WARN)
    N = 1024
    x = ti.field(ti.f32, shape=N)

    @ti.kernel
    def compute():
        for i in x:
            a = ti.f32(i) * 0.1234
            b = ti.sin(a) * ti.cos(a * 2.0) + ti.sqrt(ti.abs(a) + 1.0)
            c = b * b + a / (1.0 + a * a)
            x[i] = c + ti.f32(i) * 0.5

    compute()
    ti.sync()
    arr = x.to_numpy()
    ti.reset()
    return arr


outs = {t: run_one(t) for t in ("fast", "balanced", "full")}
ref = outs["balanced"]
print(f"balanced head: {ref[:4]}  tail: {ref[-2:]}")
ok = True
for tier in ("fast", "full"):
    diff = abs(outs[tier] - ref).max()
    print(f"{tier:9s}  max|Delta| vs balanced = {diff:.3e}")
    if diff > 1e-5:
        ok = False
print("OK" if ok else "MISMATCH")
