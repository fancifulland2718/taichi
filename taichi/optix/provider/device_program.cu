#include <optix.h>
#include <optix_device.h>

// Dense field subranges can be word-aligned without float4 alignment. Keep
// this wire layout at four-byte alignment; no packing allocation is required.
struct PackedFloat4 {
  float x, y, z, w;
  __device__ PackedFloat4() = default;
  __device__ PackedFloat4(float4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {
  }
};
struct PackedUint4 {
  unsigned int x, y, z, w;
  __device__ PackedUint4() = default;
  __device__ PackedUint4(uint4 v) : x(v.x), y(v.y), z(v.z), w(v.w) {
  }
};
struct RayRecord {
  PackedFloat4 origin_tmin;
  PackedFloat4 direction_tmax;
};

struct HitRecord {
  PackedFloat4 value;
};

struct LaunchParams {
  const RayRecord *rays;
  HitRecord *hits;
  OptixTraversableHandle traversable;
  PackedUint4 *hit_indices;
};

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __miss__forge_batch_ray() {
}

#if !TI_FORGE_OPTIX_TYPED
extern "C" __global__ void __raygen__forge_batch_ray() {
  const unsigned int index = optixGetLaunchIndex().x;
  const RayRecord ray = params.rays[index];
  unsigned int t_bits = __float_as_uint(-1.0f);
  unsigned int primitive = __float_as_uint(-1.0f);
  unsigned int instance = __float_as_uint(-1.0f);
  unsigned int hit = 0;
  optixTrace(
      params.traversable,
      make_float3(ray.origin_tmin.x, ray.origin_tmin.y, ray.origin_tmin.z),
      make_float3(ray.direction_tmax.x, ray.direction_tmax.y,
                  ray.direction_tmax.z),
      ray.origin_tmin.w, ray.direction_tmax.w, 0.0f, OptixVisibilityMask(0xff),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t_bits, primitive, instance, hit);
  params.hits[index].value =
      make_float4(__uint_as_float(t_bits), __uint_as_float(primitive),
                  __uint_as_float(instance), __uint_as_float(hit));
}

extern "C" __global__ void __closesthit__forge_batch_ray() {
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(__float_as_uint(float(optixGetPrimitiveIndex())));
  optixSetPayload_2(__float_as_uint(float(optixGetInstanceId())));
  optixSetPayload_3(__float_as_uint(1.0f));
}

#else
extern "C" __global__ void __raygen__forge_batch_ray_typed() {
  const unsigned int index = optixGetLaunchIndex().x;
  const RayRecord ray = params.rays[index];
  unsigned int t = __float_as_uint(-1.0f);
  unsigned int primitive = ~0u, instance = ~0u, custom = ~0u;
  unsigned int u = 0, v = 0, hit = 0;
  optixTrace(
      params.traversable,
      make_float3(ray.origin_tmin.x, ray.origin_tmin.y, ray.origin_tmin.z),
      make_float3(ray.direction_tmax.x, ray.direction_tmax.y,
                  ray.direction_tmax.z),
      ray.origin_tmin.w, ray.direction_tmax.w, 0.0f, OptixVisibilityMask(0xff),
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, t, primitive, instance, custom, u,
      v, hit);
  params.hits[index].value = make_float4(__uint_as_float(t), __uint_as_float(u),
                                         __uint_as_float(v), 0.0f);
  params.hit_indices[index] = make_uint4(primitive, instance, custom, hit);
}

extern "C" __global__ void __closesthit__forge_batch_ray_typed() {
  const float2 barycentrics = optixGetTriangleBarycentrics();
  optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
  optixSetPayload_1(optixGetPrimitiveIndex());
  optixSetPayload_2(optixGetInstanceIndex());
  optixSetPayload_3(optixGetInstanceId());
  optixSetPayload_4(__float_as_uint(barycentrics.x));
  optixSetPayload_5(__float_as_uint(barycentrics.y));
  optixSetPayload_6(1u);
}
#endif
