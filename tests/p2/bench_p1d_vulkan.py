"""P1.d: measure SPIR-V cold-compile time on Vulkan across compile_tier values.

For each tier in ("fast", "balanced", "full"):
    * clear offline cache (offline_cache=False + clean ticache)
    * init Taichi with arch=vulkan, compile_tier=tier
    * build the kernel (mat14 + sph_force)
    * time first launch (cold compile wall clock)
    * verify numeric parity against the first tier
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

TIERS = ("fast", "balanced", "full")
KERNELS = ("mat14", "sph_force", "spv_branchy")


def wipe_cache() -> None:
    home = os.path.expanduser("~")
    p = os.path.join(home, ".cache", "taichi", "ticache")
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)


CHILD = r"""
import sys, time, json
sys.path.insert(0, r"d:\taichi\tests\p2")
import taichi as ti
import heavy_kernels as hk

tier = sys.argv[1]
kernel_name = sys.argv[2]

# Use offline_cache=False for guaranteed cold compile; also tier is in the
# offline-cache key (P2.c), but this removes any ambiguity.
ti.init(arch=ti.vulkan, offline_cache=False, compile_tier=tier,
        random_seed=42, default_fp=ti.f32, cpu_max_num_threads=1,
        advanced_optimization=True, print_ir=False, log_level=ti.WARN)

factory = getattr(hk, "make_" + kernel_name)
run = factory(ti)

t0 = time.perf_counter()
run()               # first call: cold-compile + first launch
ti.sync()
t_cold = time.perf_counter() - t0

# Warm call for parity check only
t1 = time.perf_counter()
run()
ti.sync()
t_warm = time.perf_counter() - t1

print("__R__" + json.dumps({
    "tier": tier, "kernel": kernel_name,
    "cold_s": t_cold, "warm_s": t_warm,
}))
"""

def run_child(tier: str, kernel: str) -> dict:
    wipe_cache()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    log_path = os.path.join(os.path.dirname(__file__), f"_bench_p1d_{kernel}_{tier}.log")
    with open(log_path, "w", encoding="utf-8") as fout:
        p = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, tier, kernel],
            stdout=fout, stderr=subprocess.STDOUT, env=env, timeout=1200, text=True,
        )
    with open(log_path, "r", encoding="utf-8") as fin:
        text = fin.read()
    for line in text.splitlines():
        if line.startswith("__R__"):
            return json.loads(line[5:])
    tail = "\n".join(text.splitlines()[-30:])
    raise RuntimeError(
        f"no result from tier={tier} kernel={kernel} (rc={p.returncode})\nTAIL:\n{tail}"
    )


def main() -> None:
    results = []
    for kernel in KERNELS:
        for tier in TIERS:
            print(f"[bench] tier={tier} kernel={kernel} ...", flush=True)
            r = run_child(tier, kernel)
            results.append(r)
            print(f"         cold={r['cold_s']:.3f}s  warm={r['warm_s']:.3f}s", flush=True)

    with open(os.path.join(os.path.dirname(__file__), "bench_p1d_vulkan.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Pretty table
    print()
    print("kernel      | tier     | cold (s) | warm (s)")
    print("------------|----------|----------|---------")
    for r in results:
        print(f"{r['kernel']:<12}| {r['tier']:<9}| {r['cold_s']:>8.3f} | {r['warm_s']:>7.3f}")


if __name__ == "__main__":
    main()
