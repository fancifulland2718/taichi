// Forge bindings for FidelityFX Parallel Sort. See FFX_ParallelSort.h (MIT).
#define FFX_HLSL
#include "FFX_ParallelSort.h"

struct Parameters {
    FFX_ParallelSortCB cb;
    uint shift;
};
[[vk::push_constant]] Parameters parameters;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> dst;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> srcPayload;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> dstPayload;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> sums;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> reduced;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> scanSrc;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> scanDst;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> scanScratch;

// Forge-owned, coalesced prefix of the bounded bin-major histogram. A 16 KiB
// shared tile transposes contiguous global access to per-thread segments.
// Padding prevents shared-memory bank conflicts in both directions.
groupshared uint prefixTile[32][129];
groupshared uint prefixCarry;
[numthreads(128, 1, 1)]
void Prefix(uint localID : SV_GroupThreadID, uint bin : SV_GroupID) {
    if (bin != 0) return;
    uint values = parameters.cb.NumThreadGroups * 16;
    if (localID == 0) prefixCarry = 0;
    for (uint base = 0; base < values; base += 4096) {
        for (uint j = 0; j < 32; ++j) {
            uint i = j * 128 + localID;
            prefixTile[i % 32][i / 32] = base + i < values ? sums[base + i] : 0;
        }
        GroupMemoryBarrierWithGroupSync();
        uint localSum = 0;
        for (uint j = 0; j < 32; ++j) {
            uint value = prefixTile[j][localID];
            prefixTile[j][localID] = localSum;
            localSum += value;
        }
        uint offset = FFX_ParallelSort_BlockScanPrefix(localSum, localID) + prefixCarry;
        for (uint j = 0; j < 32; ++j) prefixTile[j][localID] += offset;
        GroupMemoryBarrierWithGroupSync();
        if (localID == 127) prefixCarry = offset + localSum;
        for (uint j = 0; j < 32; ++j) {
            uint i = j * 128 + localID;
            if (base + i < values) reduced[base + i] = prefixTile[i % 32][i / 32];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(128, 1, 1)]
void Count(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID) {
    FFX_ParallelSort_Count_uint(localID, groupID, parameters.cb, parameters.shift, src, sums);
}
[numthreads(128, 1, 1)]
void Reduce(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID) {
    FFX_ParallelSort_ReduceCount(localID, groupID, parameters.cb, sums, reduced);
}
[numthreads(128, 1, 1)]
void Scan(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID) {
    FFX_ParallelSort_ScanPrefix(parameters.cb.NumScanValues, localID, groupID,
        0, 512 * groupID, false, parameters.cb, scanSrc, scanDst, scanScratch);
}
[numthreads(128, 1, 1)]
void ScanAdd(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID) {
    uint bin = groupID / parameters.cb.NumReduceThreadgroupPerBin;
    uint base = (groupID % parameters.cb.NumReduceThreadgroupPerBin) * 512;
    FFX_ParallelSort_ScanPrefix(parameters.cb.NumThreadGroups, localID, groupID,
        bin * parameters.cb.NumThreadGroups, base, true, parameters.cb, scanSrc, scanDst, scanScratch);
}
[numthreads(128, 1, 1)]
void Scatter(uint localID : SV_GroupThreadID, uint groupID : SV_GroupID) {
    FFX_ParallelSort_Scatter_uint(localID, groupID, parameters.cb, parameters.shift, src, dst, sums
#ifdef kRS_ValueCopy
        , srcPayload, dstPayload
#endif
    );
}
