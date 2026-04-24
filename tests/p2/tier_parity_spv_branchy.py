"""V1 numeric parity on Vulkan for the new SPV-bound kernel spv_branchy.
fast (spv_opt_level=0) must match balanced/full bit-for-bit (or within noise)."""
import os, shutil
import numpy as np
from heavy_kernels import make_spv_branchy


def run_one(tier: str):
    home = os.path.expanduser("~")
    p = os.path.join(home, ".cache", "taichi", "ticache")
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    import taichi as ti
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False, compile_tier=tier,
            random_seed=42, default_fp=ti.f32, log_level=ti.WARN)
    run = make_spv_branchy(ti)
    run()
    ti.sync()
    # Re-fetch field from module globals via the factory is not exposed; use
    # taichi's impl: read through the kernel's closure.
    arr = run.__globals__.get("out")  # fallback not reliable
    # Simpler: re-declare to capture.
    ti.reset()
    return None


def run_capture(tier: str):
    home = os.path.expanduser("~")
    p = os.path.join(home, ".cache", "taichi", "ticache")
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    import taichi as ti
    ti.reset()
    ti.init(arch=ti.vulkan, offline_cache=False, compile_tier=tier,
            random_seed=42, default_fp=ti.f32, log_level=ti.WARN)

    N = 256
    out = ti.field(ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in out:
            buf = ti.Vector.zero(ti.f32, 32)
            for k in ti.static(range(32)):
                a = ti.f32(i * (k + 1)) * 0.001
                if ti.static(k % 4 == 0):
                    buf[k] = ti.sin(a) + ti.cos(a * 2.0)
                elif ti.static(k % 4 == 1):
                    buf[k] = ti.sqrt(ti.abs(a) + 1.0)
                elif ti.static(k % 4 == 2):
                    buf[k] = a / (1.0 + a * a)
                else:
                    buf[k] = a * a - a
            s = ti.f32(0.0)
            for k in ti.static(range(32)):
                if buf[k] > 0.0:
                    s += buf[k] * 1.25
                else:
                    s -= buf[k] * 0.5
                for j in ti.static(range(4)):
                    s += ti.sin(buf[k] * float(j + 1))
            out[i] = s

    run()
    ti.sync()
    arr = out.to_numpy()
    ti.reset()
    return arr


outs = {t: run_capture(t) for t in ("fast", "balanced", "full")}
ref = outs["balanced"]
print(f"balanced head: {ref[:4]}  tail: {ref[-2:]}")
ok = True
for tier in ("fast", "full"):
    diff = float(np.abs(outs[tier] - ref).max())
    print(f"{tier:9s}  max|Delta| vs balanced = {diff:.3e}")
    if diff > 1e-4:
        ok = False
print("OK" if ok else "MISMATCH")
