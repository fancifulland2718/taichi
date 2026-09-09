# FidelityFX Parallel Sort source

`FFX_ParallelSort.h` is from GPUOpen-Effects/FidelityFX-ParallelSort,
commit `0c539948c8d196ae338d91efbc8ca495f1ea0d1d`, v1.1.1.
The original MIT notice is retained in the header.

Forge changes: the twelve speculative key/payload preloads are bounded by
`NumKeys`, permitting exact-capacity ndarrays without upstream tail padding.
`parallel_sort.hlsl` is the Forge binding wrapper: one descriptor set,
immutable push constants, direct dispatch only. The upstream algorithm is
otherwise unchanged. DXC is an explicit caller-provided cold JIT dependency,
not a shared runtime imported by the wheel. No shader compilation at replay.
