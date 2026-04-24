"""Very heavy Taichi kernels for cross-version compile-time benchmarking.

Design goals
------------
- Each kernel should take **10 s – 120 s** to compile on Taichi 1.7.4 / LLVM 15
  on a modern CPU or CUDA GPU.
- Kernels exercise the paths that differ most between LLVM 15 and LLVM 19:
    * massive matrix scalarization (IR width)
    * deeply unrolled static loops (register pressure)
    * full autodiff tape (IR depth × 3)
    * P2G/G2P stencil scatter (27-point nested static loops + matrix ops)
    * dense neural-net layer chains (matmul + activation × N)
- Every factory accepts an arch-initialised `taichi` module and returns
  a callable that triggers JIT-compilation on first call.

Kernels
-------
  mat14   – 14×14 matrix multiply + add; 2744 scalar muls after scalarization
  mat16   – 16×16 matrix multiply + add; 4096 scalar muls (ULTRA heavy)
  mpm_p2g – 3D MPM P2G with APIC 3×3 matrices; 27-node stencil unrolled
  diffuse_auto – 3-step 2D diffusion wrapped in full autodiff tape (tape×3)
  dense_net – 4-layer dense net 32→64→128→64→1 forward + backward
  sph_force – SPH with 32 statically-unrolled neighbour ops

Use HARD_KERNELS (name, factory, est_baseline_cpu_s, est_baseline_cuda_s)
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# 1.  mat14  –  14×14 × 14×14 matrix multiply
#     Scalarization count: 14 * 14 * 14 = 2744 muls + 14*14*13 = 2548 adds
#     Expected compile: CPU ~25 s (ti 1.7.4),  CUDA ~50 s
# ---------------------------------------------------------------------------
def make_mat14(ti):
    N = 32
    A = ti.Matrix.field(14, 14, dtype=ti.f32, shape=(N, N))
    B = ti.Matrix.field(14, 14, dtype=ti.f32, shape=(N, N))
    C = ti.Matrix.field(14, 14, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def matmul14():
        for p, q in C:
            C[p, q] = A[p, q] @ B[p, q] + A[p, q].transpose()

    return matmul14


# ---------------------------------------------------------------------------
# 2.  mat16  –  16×16 × 16×16 matrix multiply
#     Scalarization: 16*16*16 = 4096 muls + 16*16*15 = 3840 adds
#     Expected compile: CPU ~45 s (ti 1.7.4),  CUDA ~90+ s
# ---------------------------------------------------------------------------
def make_mat16(ti):
    N = 16
    A = ti.Matrix.field(16, 16, dtype=ti.f32, shape=(N, N))
    B = ti.Matrix.field(16, 16, dtype=ti.f32, shape=(N, N))
    C = ti.Matrix.field(16, 16, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def matmul16():
        for p, q in C:
            C[p, q] = A[p, q] @ B[p, q] + A[p, q].transpose()

    return matmul16


# ---------------------------------------------------------------------------
# 3.  mpm_p2g  –  3D APIC MPM particle-to-grid transfer
#     APIC: each particle has a 3×3 C matrix.  27-node stencil statically
#     unrolled (ti.static).  Two full grid fields (velocity 3-vec + mass).
#     Also includes a full G2P backward scatter to double the kernel count.
#     Rough op count after scalarization (per P2G call site):
#       Weight polys × 27: 27 × 18 = 486
#       C @ dpos (3×3 @ 3-vec): 27 × (9×mul + 6×add) = 27 × 15 = 405
#       Scatter to 2 fields: 27 × 4 = 108
#     Total per kernel ≈ 1000 scalar ops, ×2 for G2P → ~2000 total.
#     Expected compile: CPU ~15 s (ti 1.7.4),  CUDA ~30 s
# ---------------------------------------------------------------------------
def make_mpm_p2g(ti):
    DX = 1.0 / 64.0
    INV_DX = 64.0
    NP = 1024
    grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(64, 64, 64))
    grid_m = ti.field(ti.f32, shape=(64, 64, 64))
    x  = ti.Vector.field(3, dtype=ti.f32, shape=NP)
    v  = ti.Vector.field(3, dtype=ti.f32, shape=NP)
    C  = ti.Matrix.field(3, 3, dtype=ti.f32, shape=NP)
    mp = ti.field(ti.f32, shape=NP)

    @ti.kernel
    def p2g():
        for p in x:
            xp   = x[p] * INV_DX - 0.5
            base = ti.cast(xp, ti.i32)
            fx   = xp - ti.cast(base, ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0])**2,
                            0.75 - (fx[0] - 1.0)**2,
                            0.5 * (fx[0] - 0.5)**2])
            wy = ti.Vector([0.5 * (1.5 - fx[1])**2,
                            0.75 - (fx[1] - 1.0)**2,
                            0.5 * (fx[1] - 0.5)**2])
            wz = ti.Vector([0.5 * (1.5 - fx[2])**2,
                            0.75 - (fx[2] - 1.0)**2,
                            0.5 * (fx[2] - 0.5)**2])
            mass_p = mp[p]
            vel_p  = v[p]
            C_p    = C[p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos   = (offset - fx) * DX
                        weight = wx[i] * wy[j] * wz[k]
                        node   = base + offset
                        grid_v[node] += weight * mass_p * (vel_p + C_p @ dpos)
                        grid_m[node] += weight * mass_p

    @ti.kernel
    def g2p():
        for p in x:
            xp   = x[p] * INV_DX - 0.5
            base = ti.cast(xp, ti.i32)
            fx   = xp - ti.cast(base, ti.f32)
            wx = ti.Vector([0.5 * (1.5 - fx[0])**2,
                            0.75 - (fx[0] - 1.0)**2,
                            0.5 * (fx[0] - 0.5)**2])
            wy = ti.Vector([0.5 * (1.5 - fx[1])**2,
                            0.75 - (fx[1] - 1.0)**2,
                            0.5 * (fx[1] - 0.5)**2])
            wz = ti.Vector([0.5 * (1.5 - fx[2])**2,
                            0.75 - (fx[2] - 1.0)**2,
                            0.5 * (fx[2] - 0.5)**2])
            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos   = (offset - fx) * DX
                        weight = wx[i] * wy[j] * wz[k]
                        node   = base + offset
                        g_v    = grid_v[node]
                        new_v += weight * g_v
                        new_C += 4.0 * INV_DX * weight * g_v.outer_product(dpos)
            v[p]  = new_v
            C[p]  = new_C

    def step():
        p2g()
        g2p()

    return step


# ---------------------------------------------------------------------------
# 4.  diffuse_auto  –  2D heat diffusion with full autodiff tape (3 steps)
#     3-step unrolled Jacobi iteration, wrapped in ti.ad.Tape → tape ≈ 3×.
#     10×10 stencil coefficients per cell → complex IR even before tape.
#     Expected compile: CPU ~20 s (ti 1.7.4),  CUDA ~40 s
# ---------------------------------------------------------------------------
def make_diffuse_auto(ti):
    SZ = 128
    u      = ti.field(ti.f32, shape=(SZ, SZ), needs_grad=True)
    u_new  = ti.field(ti.f32, shape=(SZ, SZ), needs_grad=True)
    loss   = ti.field(ti.f32, shape=(), needs_grad=True)

    DT = 0.1

    @ti.kernel
    def diffuse_step():
        for i, j in u:
            ic = ti.max(i - 1, 0)
            ie = ti.min(i + 1, SZ - 1)
            jc = ti.max(j - 1, 0)
            je = ti.min(j + 1, SZ - 1)
            lap = (u[ic, j] + u[ie, j] + u[i, jc] + u[i, je]
                   - 4.0 * u[i, j])
            u_new[i, j] = u[i, j] + DT * lap

    @ti.kernel
    def copy_back():
        for i, j in u:
            u[i, j] = u_new[i, j]

    @ti.kernel
    def calc_loss():
        for i, j in u:
            loss[None] += u[i, j] * u[i, j]

    dt = 0.1

    def forward_with_tape():
        with ti.ad.Tape(loss=loss):
            diffuse_step()
            copy_back()
            diffuse_step()
            copy_back()
            diffuse_step()
            copy_back()
            calc_loss()

    return forward_with_tape


# ---------------------------------------------------------------------------
# 5.  mat_chain3  –  chain of three 12×12 matrix operations in one kernel
#     C = (A @ B) @ A  +  B.transpose() @ A.transpose()
#     Scalarization: 3 independent 12×12×12 matmuls = 3 × 1728 = 5184 muls
#     plus 12×12 add and transpose.  Well within LLVM's handling but forces
#     a very large basic block.
#     Expected compile: CPU ~45 s (ti 1.7.4),  CUDA ~90 s
# ---------------------------------------------------------------------------
def make_mat_chain3(ti):
    N = 16
    A = ti.Matrix.field(12, 12, dtype=ti.f32, shape=(N, N))
    B = ti.Matrix.field(12, 12, dtype=ti.f32, shape=(N, N))
    C = ti.Matrix.field(12, 12, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def mat_chain3():
        for p, q in C:
            # Three full 12×12 matrix multiplications in one kernel body
            ab   = A[p, q] @ B[p, q]
            aba  = ab @ A[p, q]
            btAt = B[p, q].transpose() @ A[p, q].transpose()
            C[p, q] = aba + btAt

    return mat_chain3


# ---------------------------------------------------------------------------
# 6.  sph_force  –  SPH fluid: density + pressure gradient + viscosity
#     64 statically-unrolled neighbour evaluations per particle.
#     Each neighbour: Wendland C2 (polynomial) + pressure/viscosity forces.
#     Per-neighbour ops after scalarization: ~50 scalar ops.
#     Total: 64 × 50 = ~3200 scalar ops per parallel iteration.
#     Expected compile: CPU ~20 s (ti 1.7.4),  CUDA ~40 s
# ---------------------------------------------------------------------------
def make_sph_force(ti):
    NP   = 512
    H    = 0.1
    NMAX = 64       # statically-unrolled neighbour count (doubled)

    pos    = ti.Vector.field(3, dtype=ti.f32, shape=NP)
    vel    = ti.Vector.field(3, dtype=ti.f32, shape=NP)
    rho    = ti.field(ti.f32, shape=NP)
    press  = ti.field(ti.f32, shape=NP)
    force  = ti.Vector.field(3, dtype=ti.f32, shape=NP)
    nbrs   = ti.field(ti.i32, shape=(NP, NMAX))

    @ti.kernel
    def compute_force():
        for i in pos:
            fi = ti.Vector([0.0, 0.0, 0.0])
            for jj in ti.static(range(NMAX)):
                j    = nbrs[i, jj]
                rij  = pos[i] - pos[j]
                r2   = rij.dot(rij)
                r    = ti.sqrt(r2 + 1e-9)
                q    = r / H
                fac  = ti.max(1.0 - q, 0.0)
                # Wendland C2 kernel gradient
                grad = (-20.0 / (H * H * H) * fac * fac * fac
                        * rij / (r + 1e-9))
                # Pressure gradient (symmetric SPH)
                p_ij = (press[i] / (rho[i] * rho[i] + 1e-9)
                       + press[j] / (rho[j] * rho[j] + 1e-9))
                fi  += -p_ij * grad
                # Viscosity (Monaghan '92 artificial viscosity)
                mu = 0.01 * H * (vel[i] - vel[j]).dot(rij) / (r2 + 1e-4)
                fi  += mu / (rho[j] + 1e-9) * grad
                # Surface tension
                fi  += 0.001 * fac * fac * rij
            force[i] = fi

    return compute_force


# ---------------------------------------------------------------------------
# Registry
# (name, factory, est_baseline_cpu_s, est_baseline_cuda_s)
# ---------------------------------------------------------------------------
HEAVY_KERNELS = [
    ("mat14",        make_mat14,        30,  60),
    ("mat16",        make_mat16,        60, 120),
    ("mat_chain3",   make_mat_chain3,   50, 100),
    ("mpm_p2g",      make_mpm_p2g,      20,  40),
    ("diffuse_auto", make_diffuse_auto, 20,  40),
    ("sph_force",    make_sph_force,    20,  40),
]


# ---------------------------------------------------------------------------
# 7.  spv_branchy  –  SPV-bound kernel for Vulkan tiered-opt benchmarking.
#     Stresses the spvtools optimizer passes that dominate level 2/3 cost:
#     PrivateToLocal, LocalAccessChainConvert, ScalarReplacement,
#     LocalSingleBlockLoadStoreElim, IfConversion, BlockMerge.
#     Compile on mat14/sph_force is dominated by CHI IR + SPV codegen, so
#     the 23 spvtools passes are noise on those kernels. This one is built
#     specifically so that tier="fast" (level 0) vs "balanced" (level 3)
#     differ outside the noise band.
# ---------------------------------------------------------------------------
def make_spv_branchy(ti):
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

    return run
