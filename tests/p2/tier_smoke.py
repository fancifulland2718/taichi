import taichi as ti

for tier in ("fast", "balanced", "full"):
    ti.reset()
    ti.init(arch=ti.cpu, compile_tier=tier)
    x = ti.field(ti.f64, shape=64)

    @ti.kernel
    def k():
        for i in x:
            x[i] = i * i + 0.5

    k()
    a = x.to_numpy()
    print(f"{tier:8s} head={a[:4].tolist()} tail={a[-2:].tolist()}", flush=True)
print("OK", flush=True)
