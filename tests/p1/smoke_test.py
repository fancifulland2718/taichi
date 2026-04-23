"""Quick smoke test for P1 changes."""
import taichi as ti

# Test P1.b: llvm_opt_level config
ti.init(arch=ti.cpu, log_level='error', llvm_opt_level=2)

x = ti.field(ti.f32, shape=1024)

@ti.kernel
def saxpy(a: float):
    for i in x:
        x[i] = a * x[i] + 1.0

# Trigger compile
saxpy(2.0)
ti.sync()
print('P1.b: llvm_opt_level=2 kernel compiled and ran OK')

ti.reset()

# Test P1.b: default llvm_opt_level=3 (backward compat)
ti.init(arch=ti.cpu, log_level='error')
from taichi.lang import impl
cfg = impl.default_cfg()
assert cfg.llvm_opt_level == 3, f"Expected default=3, got {cfg.llvm_opt_level}"

y = ti.field(ti.f32, shape=1024)

@ti.kernel
def fill():
    for i in y:
        y[i] = float(i)

fill()
ti.sync()
val = y[42]
print(f'P1.b: default config OK (llvm_opt_level={cfg.llvm_opt_level}, y[42]={val:.1f})')

print('ALL P1 smoke tests PASSED')
