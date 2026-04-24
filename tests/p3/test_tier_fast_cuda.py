"""CUDA sanity for tier=fast: compile + run and ensure numerical parity
versus tier=balanced within 1e-5 relative tolerance (Taichi cross-backend bar).
"""
import taichi as ti


def run(tier):
    ti.init(arch=ti.cuda, offline_cache=False, compile_tier=tier, random_seed=0)
    x = ti.field(ti.f64, shape=64)
    @ti.kernel
    def k():
        for i in x:
            v = ti.cast(i, ti.f64) * 1.25 + 0.5
            x[i] = v * v - v * 0.3
    k()
    return [x[i] for i in range(8)]


if __name__ == "__main__":
    a = run("balanced")
    b = run("fast")
    delta = max(abs(x - y) for x, y in zip(a, b))
    print(f"CUDA balanced head={a[:4]}")
    print(f"CUDA fast     head={b[:4]}")
    print(f"CUDA max abs delta={delta:.3e}")
    assert delta < 1e-5, f"CUDA tier=fast drifted {delta:.3e}"
    print("OK")
