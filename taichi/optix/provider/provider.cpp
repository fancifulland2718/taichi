#include "taichi/optix/forge_optix_provider.h"

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <string>

#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "device_program_ptx.h"

#if OPTIX_ABI_VERSION != 93 && OPTIX_ABI_VERSION != 105 && \
    OPTIX_ABI_VERSION != 118
#error "Forge OptiX provider supports only SDK ABI 93, 105, and 118"
#endif

namespace {

thread_local std::string last_error;
thread_local std::string last_optix_log;
std::mutex optix_loader_mutex;
void *optix_library_handle{nullptr};
std::size_t optix_context_count{0};
std::string active_optix_runtime_library_path;

#define TI_FORGE_STRINGIFY_IMPL(value) #value
#define TI_FORGE_STRINGIFY(value) TI_FORGE_STRINGIFY_IMPL(value)

constexpr char kProviderName[] = "taichi-forge-optix";
constexpr char kBuildIdentity[] =
    "forge-optix-provider-abi1-optix-abi" TI_FORGE_STRINGIFY(
        OPTIX_ABI_VERSION) "-scene-refit2";
constexpr uint64_t kFeatures = TI_FORGE_OPTIX_FEATURE_TRIANGLE_GAS |
                               TI_FORGE_OPTIX_FEATURE_SINGLE_INSTANCE_IAS |
                               TI_FORGE_OPTIX_FEATURE_GAS_UPDATE |
                               TI_FORGE_OPTIX_FEATURE_BATCH_CLOSEST_HIT |
                               TI_FORGE_OPTIX_FEATURE_RUNTIME_ORDERED_STREAM |
                               TI_FORGE_OPTIX_FEATURE_EXACT_DEVICE_MEMORY;

void clear_error_state() {
  last_error.clear();
  last_optix_log.clear();
}

TiForgeOptixResult fail(TiForgeOptixResult result, std::string message) {
  last_error = std::move(message);
  return result;
}

TiForgeOptixResult cuda_check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return TI_FORGE_OPTIX_SUCCESS;
  }
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  return fail(TI_FORGE_OPTIX_ERROR_CUDA_CALL,
              std::string(operation) + " failed: " +
                  (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + " (" +
                  (description == nullptr ? "no description" : description) +
                  ")");
}

TiForgeOptixResult optix_check(OptixResult result, const char *operation) {
  if (result == OPTIX_SUCCESS) {
    last_optix_log.clear();
    return TI_FORGE_OPTIX_SUCCESS;
  }
  std::string message =
      std::string(operation) + " failed: " +
      (optixGetErrorName(result) == nullptr
           ? std::to_string(static_cast<int>(result))
           : optixGetErrorName(result));
  if (!last_optix_log.empty()) {
    message += "; validation log: " + last_optix_log;
  }
  last_optix_log.clear();
  return fail(TI_FORGE_OPTIX_ERROR_OPTIX_CALL, std::move(message));
}

struct DeviceBuffer {
  CUdeviceptr pointer{0};
  std::size_t bytes{0};

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    reset();
  }

  TiForgeOptixResult allocate(std::size_t requested) {
    reset();
    bytes = requested;
    if (bytes == 0) {
      return TI_FORGE_OPTIX_SUCCESS;
    }
    const auto result = cuda_check(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      pointer = 0;
      bytes = 0;
    }
    return result;
  }

  void reset() {
    if (pointer != 0) {
      cuMemFree(pointer);
    }
    pointer = 0;
    bytes = 0;
  }
};

struct EmptySbtData {
  uint32_t reserved{0};
};

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
  char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct Context {
  CUcontext cuda_context{nullptr};
  OptixDeviceContext optix_context{nullptr};
  OptixModule module{nullptr};
  OptixProgramGroup raygen{nullptr};
  OptixProgramGroup miss{nullptr};
  OptixProgramGroup hitgroup{nullptr};
  OptixPipeline pipeline{nullptr};
  DeviceBuffer raygen_record;
  DeviceBuffer miss_record;
  DeviceBuffer hitgroup_record;
  OptixShaderBindingTable sbt{};
  std::atomic<std::size_t> scene_count{0};
};

struct LaunchParams {
  CUdeviceptr rays;
  CUdeviceptr hits;
  OptixTraversableHandle traversable;
};

struct Scene {
  Context *context{nullptr};
  uint32_t vertex_count{0};
  uint32_t triangle_count{0};
  bool allow_update{false};
  DeviceBuffer gas;
  DeviceBuffer ias;
  DeviceBuffer scratch;
  DeviceBuffer ias_scratch;
  DeviceBuffer instance;
  DeviceBuffer launch_params;
  OptixTraversableHandle gas_handle{0};
  OptixTraversableHandle ias_handle{0};
};

void optix_log(unsigned int level,
               const char *tag,
               const char *message,
               void *) {
  if (level <= 3 && message != nullptr) {
    last_optix_log = std::string("OptiX[") + (tag == nullptr ? "" : tag) +
                     "]: " + message;
  }
}

OptixResult init_optix_library(const std::string &library_path, void **handle) {
  if (library_path.empty()) {
    return optixInitWithHandle(handle);
  }
  *handle = nullptr;
#if defined(_WIN32)
  const int wide_size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                             library_path.c_str(), -1, nullptr, 0);
  if (wide_size <= 0) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  std::wstring wide_path(static_cast<std::size_t>(wide_size), L'\0');
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                          library_path.c_str(), -1, wide_path.data(),
                          wide_size) <= 0) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  *handle = LoadLibraryW(wide_path.c_str());
  if (*handle == nullptr) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  void *symbol = reinterpret_cast<void *>(
      GetProcAddress(static_cast<HMODULE>(*handle), "optixQueryFunctionTable"));
#else
  *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return OPTIX_ERROR_LIBRARY_NOT_FOUND;
  }
  void *symbol = dlsym(*handle, "optixQueryFunctionTable");
#endif
  if (symbol == nullptr) {
    optixUninitWithHandle(*handle);
    *handle = nullptr;
    return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
  }
  auto *query = reinterpret_cast<OptixQueryFunctionTable_t *>(symbol);
  const auto result = query(OPTIX_ABI_VERSION, 0, nullptr, nullptr,
                            &OPTIX_FUNCTION_TABLE_SYMBOL,
                            sizeof(OPTIX_FUNCTION_TABLE_SYMBOL));
  if (result != OPTIX_SUCCESS) {
    optixUninitWithHandle(*handle);
    *handle = nullptr;
  }
  return result;
}

TiForgeOptixResult retain_optix_loader(const char *library_path) {
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  const std::string requested = library_path == nullptr ? "" : library_path;
  if (optix_context_count == 0) {
    const auto result = init_optix_library(requested, &optix_library_handle);
    if (result != OPTIX_SUCCESS) {
      optix_library_handle = nullptr;
      return fail(TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE,
                  std::string("OptiX runtime initialization failed: ") +
                      (optixGetErrorName(result) == nullptr
                           ? std::to_string(static_cast<int>(result))
                           : optixGetErrorName(result)));
    }
    active_optix_runtime_library_path = requested;
  } else if (requested != active_optix_runtime_library_path) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "all live contexts in one adapter must use the same OptiX runtime library");
  }
  ++optix_context_count;
  return TI_FORGE_OPTIX_SUCCESS;
}

void release_optix_loader() {
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  if (optix_context_count == 0) {
    return;
  }
  --optix_context_count;
  if (optix_context_count == 0 && optix_library_handle != nullptr) {
    optixUninitWithHandle(optix_library_handle);
    optix_library_handle = nullptr;
    active_optix_runtime_library_path.clear();
  }
}

TiForgeOptixResult probe_runtime(const char *library_path) {
  clear_error_state();
  std::lock_guard<std::mutex> lock(optix_loader_mutex);
  const std::string requested = library_path == nullptr ? "" : library_path;
  if (optix_context_count != 0) {
    if (requested == active_optix_runtime_library_path) {
      return TI_FORGE_OPTIX_SUCCESS;
    }
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "cannot probe another OptiX runtime while contexts are live");
  }
  void *handle = nullptr;
  const auto result = init_optix_library(requested, &handle);
  if (result != OPTIX_SUCCESS) {
    return fail(TI_FORGE_OPTIX_ERROR_OPTIX_UNAVAILABLE,
                std::string("OptiX runtime probe failed: ") +
                    (optixGetErrorName(result) == nullptr
                         ? std::to_string(static_cast<int>(result))
                         : optixGetErrorName(result)));
  }
  optixUninitWithHandle(handle);
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult copy_sbt_record(DeviceBuffer &destination,
                                   OptixProgramGroup program_group) {
  SbtRecord<EmptySbtData> record{};
  auto result = optix_check(optixSbtRecordPackHeader(program_group, &record),
                            "optixSbtRecordPackHeader");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = destination.allocate(sizeof(record));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return cuda_check(cuMemcpyHtoD(destination.pointer, &record, sizeof(record)),
                    "cuMemcpyHtoD(SBT)");
}

TiForgeOptixResult create_pipeline(Context *context) {
  OptixModuleCompileOptions module_options{};
  module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OptixPipelineCompileOptions pipeline_options{};
  pipeline_options.usesMotionBlur = false;
  pipeline_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_options.numPayloadValues = 4;
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
#if OPTIX_ABI_VERSION == 118
  pipeline_options.pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams);
#endif

  char log[8192]{};
  std::size_t log_size = sizeof(log);
  auto result = optix_check(
      optixModuleCreate(context->optix_context, &module_options,
                        &pipeline_options, ti_forge_optix_device_ptx,
                        std::strlen(ti_forge_optix_device_ptx), log, &log_size,
                        &context->module),
      "optixModuleCreate");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    if (log_size > 1) {
      last_error += std::string("; module log: ") + log;
    }
    return result;
  }

  OptixProgramGroupOptions group_options{};
  OptixProgramGroupDesc raygen_desc{};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = context->module;
  raygen_desc.raygen.entryFunctionName = "__raygen__forge_batch_ray";
  log_size = sizeof(log);
  result = optix_check(
      optixProgramGroupCreate(context->optix_context, &raygen_desc, 1,
                              &group_options, log, &log_size,
                              &context->raygen),
      "optixProgramGroupCreate(raygen)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  OptixProgramGroupDesc miss_desc{};
  miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_desc.miss.module = context->module;
  miss_desc.miss.entryFunctionName = "__miss__forge_batch_ray";
  log_size = sizeof(log);
  result = optix_check(
      optixProgramGroupCreate(context->optix_context, &miss_desc, 1,
                              &group_options, log, &log_size, &context->miss),
      "optixProgramGroupCreate(miss)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  OptixProgramGroupDesc hitgroup_desc{};
  hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_desc.hitgroup.moduleCH = context->module;
  hitgroup_desc.hitgroup.entryFunctionNameCH =
      "__closesthit__forge_batch_ray";
  log_size = sizeof(log);
  result = optix_check(
      optixProgramGroupCreate(context->optix_context, &hitgroup_desc, 1,
                              &group_options, log, &log_size,
                              &context->hitgroup),
      "optixProgramGroupCreate(hitgroup)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  const OptixProgramGroup groups[] = {context->raygen, context->miss,
                                      context->hitgroup};
  OptixPipelineLinkOptions link_options{};
  link_options.maxTraceDepth = 1;
  log_size = sizeof(log);
  result = optix_check(
      optixPipelineCreate(context->optix_context, &pipeline_options,
                          &link_options, groups, 3, log, &log_size,
                          &context->pipeline),
      "optixPipelineCreate");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = optix_check(
      optixPipelineSetStackSize(context->pipeline, 0, 0, 8192, 2),
      "optixPipelineSetStackSize");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }

  result = copy_sbt_record(context->raygen_record, context->raygen);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = copy_sbt_record(context->miss_record, context->miss);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = copy_sbt_record(context->hitgroup_record, context->hitgroup);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  context->sbt.raygenRecord = context->raygen_record.pointer;
  context->sbt.missRecordBase = context->miss_record.pointer;
  context->sbt.missRecordStrideInBytes = sizeof(SbtRecord<EmptySbtData>);
  context->sbt.missRecordCount = 1;
  context->sbt.hitgroupRecordBase = context->hitgroup_record.pointer;
  context->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<EmptySbtData>);
  context->sbt.hitgroupRecordCount = 1;
  return TI_FORGE_OPTIX_SUCCESS;
}

void destroy_pipeline(Context *context) {
  if (context->pipeline != nullptr) {
    optixPipelineDestroy(context->pipeline);
  }
  if (context->hitgroup != nullptr) {
    optixProgramGroupDestroy(context->hitgroup);
  }
  if (context->miss != nullptr) {
    optixProgramGroupDestroy(context->miss);
  }
  if (context->raygen != nullptr) {
    optixProgramGroupDestroy(context->raygen);
  }
  if (context->module != nullptr) {
    optixModuleDestroy(context->module);
  }
  context->pipeline = nullptr;
  context->hitgroup = nullptr;
  context->miss = nullptr;
  context->raygen = nullptr;
  context->module = nullptr;
}

OptixBuildInput triangle_build_input(
    const TiForgeOptixTriangleSceneDesc &desc,
    CUdeviceptr *vertex_buffer,
    unsigned int *geometry_flags) {
  *vertex_buffer = static_cast<CUdeviceptr>(desc.vertices);
  *geometry_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  input.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
  input.triangleArray.numVertices = desc.vertex_count;
  input.triangleArray.vertexBuffers = vertex_buffer;
  input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  input.triangleArray.indexStrideInBytes = sizeof(uint32_t) * 3;
  input.triangleArray.numIndexTriplets = desc.triangle_count;
  input.triangleArray.indexBuffer = static_cast<CUdeviceptr>(desc.indices);
  input.triangleArray.flags = geometry_flags;
  input.triangleArray.numSbtRecords = 1;
  return input;
}

OptixAccelBuildOptions build_options(bool allow_update,
                                     OptixBuildOperation operation) {
  OptixAccelBuildOptions options{};
  options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                       (allow_update ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0u);
  options.operation = operation;
  return options;
}

TiForgeOptixResult create_gas(Scene *scene,
                              const TiForgeOptixTriangleSceneDesc &desc) {
  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(desc, &vertex_buffer, &geometry_flags);
  auto options = build_options(scene->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  auto result = optix_check(
      optixAccelComputeMemoryUsage(scene->context->optix_context, &options,
                                   &input, 1, &sizes),
      "optixAccelComputeMemoryUsage(GAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->gas.allocate(sizes.outputSizeInBytes);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->scratch.allocate(std::max(sizes.tempSizeInBytes,
                                            sizes.tempUpdateSizeInBytes));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc.cuda_stream), &options,
                      &input, 1, scene->scratch.pointer, scene->scratch.bytes,
                      scene->gas.pointer, scene->gas.bytes, &scene->gas_handle,
                      nullptr, 0),
      "optixAccelBuild(GAS)");
}

TiForgeOptixResult create_ias(Scene *scene, CUstream stream) {
  OptixInstance host_instance{};
  const float identity[12] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::memcpy(host_instance.transform, identity, sizeof(identity));
  host_instance.instanceId = 0;
  host_instance.sbtOffset = 0;
  host_instance.visibilityMask = 0xff;
  host_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
  host_instance.traversableHandle = scene->gas_handle;

  auto result = scene->instance.allocate(sizeof(host_instance));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = cuda_check(cuMemcpyHtoDAsync(scene->instance.pointer, &host_instance,
                                       sizeof(host_instance), stream),
                      "cuMemcpyHtoDAsync(OptixInstance)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  OptixBuildInput input{};
  input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  input.instanceArray.instances = scene->instance.pointer;
  input.instanceArray.numInstances = 1;
  const auto options =
      build_options(scene->allow_update, OPTIX_BUILD_OPERATION_BUILD);
  OptixAccelBufferSizes sizes{};
  result =
      optix_check(optixAccelComputeMemoryUsage(scene->context->optix_context,
                                               &options, &input, 1, &sizes),
                  "optixAccelComputeMemoryUsage(IAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->ias_scratch.allocate(
      std::max(sizes.tempSizeInBytes, sizes.tempUpdateSizeInBytes));
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = scene->ias.allocate(sizes.outputSizeInBytes);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = optix_check(
      optixAccelBuild(scene->context->optix_context, stream, &options, &input,
                      1, scene->ias_scratch.pointer, scene->ias_scratch.bytes,
                      scene->ias.pointer, scene->ias.bytes, &scene->ias_handle,
                      nullptr, 0),
      "optixAccelBuild(IAS)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  result = cuda_check(cuStreamSynchronize(stream),
                      "cuStreamSynchronize(scene build)");
  if (result == TI_FORGE_OPTIX_SUCCESS && !scene->allow_update) {
    scene->ias_scratch.reset();
  }
  return result;
}

TiForgeOptixResult create_context(const TiForgeOptixContextDesc *desc,
                                  TiForgeOptixContext *out_context) {
  clear_error_state();
  if (desc == nullptr || out_context == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixContextDesc)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX context descriptor");
  }
  *out_context = nullptr;
  CUcontext cuda_context = reinterpret_cast<CUcontext>(desc->cuda_context);
  if (cuda_context == nullptr) {
    auto result = cuda_check(cuCtxGetCurrent(&cuda_context), "cuCtxGetCurrent");
    if (result != TI_FORGE_OPTIX_SUCCESS) {
      return result;
    }
  }
  if (cuda_context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_CUDA_CONTEXT,
                "Forge OptiX provider requires the active Taichi CUDA context");
  }
  auto result = retain_optix_loader(desc->runtime_library_path);
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  auto *context = new (std::nothrow) Context;
  if (context == nullptr) {
    release_optix_loader();
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX context");
  }
  context->cuda_context = cuda_context;
  OptixDeviceContextOptions options{};
  options.logCallbackFunction = optix_log;
  options.logCallbackLevel = desc->validation_mode ? 4 : 1;
  result = optix_check(
      optixDeviceContextCreate(cuda_context, &options, &context->optix_context),
      "optixDeviceContextCreate");
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_pipeline(context);
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    destroy_pipeline(context);
    if (context->optix_context != nullptr) {
      optixDeviceContextDestroy(context->optix_context);
    }
    delete context;
    release_optix_loader();
    return result;
  }
  *out_context = context;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_context(TiForgeOptixContext raw_context) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX context is null");
  }
  if (context->scene_count.load() != 0) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX context still owns live triangle scenes");
  }
  cuCtxSynchronize();
  destroy_pipeline(context);
  if (context->optix_context != nullptr) {
    optixDeviceContextDestroy(context->optix_context);
  }
  delete context;
  release_optix_loader();
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult create_triangle_scene(
    TiForgeOptixContext raw_context,
    const TiForgeOptixTriangleSceneDesc *desc,
    TiForgeOptixTriangleScene *out_scene) {
  clear_error_state();
  auto *context = static_cast<Context *>(raw_context);
  if (context == nullptr || desc == nullptr || out_scene == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTriangleSceneDesc) ||
      desc->vertex_count == 0 || desc->triangle_count == 0 ||
      desc->vertices == 0 || desc->indices == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX triangle scene descriptor");
  }
  *out_scene = nullptr;
  auto *scene = new (std::nothrow) Scene;
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_OUT_OF_MEMORY,
                "failed to allocate the Forge OptiX scene");
  }
  scene->context = context;
  scene->vertex_count = desc->vertex_count;
  scene->triangle_count = desc->triangle_count;
  scene->allow_update = desc->allow_update != 0;
  auto result = scene->launch_params.allocate(sizeof(LaunchParams));
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_gas(scene, *desc);
  }
  if (result == TI_FORGE_OPTIX_SUCCESS) {
    result = create_ias(scene, reinterpret_cast<CUstream>(desc->cuda_stream));
  }
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    delete scene;
    return result;
  }
  context->scene_count.fetch_add(1);
  *out_scene = scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult update_triangle_scene(
    TiForgeOptixTriangleScene raw_scene,
    const TiForgeOptixTriangleSceneDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || desc == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTriangleSceneDesc) ||
      desc->vertex_count != scene->vertex_count ||
      desc->triangle_count != scene->triangle_count || desc->vertices == 0 ||
      desc->indices == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX scene update must preserve geometry shape");
  }
  if (!scene->allow_update) {
    return fail(TI_FORGE_OPTIX_ERROR_LIFETIME,
                "OptiX scene was not created with update support");
  }
  CUdeviceptr vertex_buffer = 0;
  unsigned int geometry_flags = 0;
  auto input = triangle_build_input(*desc, &vertex_buffer, &geometry_flags);
  auto options = build_options(true, OPTIX_BUILD_OPERATION_UPDATE);
  auto result = optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc->cuda_stream), &options,
                      &input, 1, scene->scratch.pointer, scene->scratch.bytes,
                      scene->gas.pointer, scene->gas.bytes, &scene->gas_handle,
                      nullptr, 0),
      "optixAccelBuild(GAS update)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  // A changed GAS bound does not refresh its parent IAS. The topology and GAS
  // handle are fixed, so reuse the instance description and retained scratch.
  // Both updates are ordered on the caller's stream; there is no readback or
  // host synchronization between the producer, these updates and later rays.
  OptixBuildInput instance_input{};
  instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instance_input.instanceArray.instances = scene->instance.pointer;
  instance_input.instanceArray.numInstances = 1;
  return optix_check(
      optixAccelBuild(scene->context->optix_context,
                      reinterpret_cast<CUstream>(desc->cuda_stream), &options,
                      &instance_input, 1, scene->ias_scratch.pointer,
                      scene->ias_scratch.bytes, scene->ias.pointer,
                      scene->ias.bytes, &scene->ias_handle, nullptr, 0),
      "optixAccelBuild(IAS update)");
}

TiForgeOptixResult trace(TiForgeOptixTriangleScene raw_scene,
                         const TiForgeOptixTraceDesc *desc) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || desc == nullptr ||
      desc->struct_size < sizeof(TiForgeOptixTraceDesc) ||
      desc->ray_count == 0 || desc->rays == 0 || desc->hits == 0) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX trace descriptor");
  }
  const auto stream = reinterpret_cast<CUstream>(desc->cuda_stream);
  const LaunchParams params{static_cast<CUdeviceptr>(desc->rays),
                            static_cast<CUdeviceptr>(desc->hits),
                            scene->ias_handle};
  auto result = cuda_check(cuMemcpyHtoDAsync(scene->launch_params.pointer,
                                             &params, sizeof(params), stream),
                           "cuMemcpyHtoDAsync(launch params)");
  if (result != TI_FORGE_OPTIX_SUCCESS) {
    return result;
  }
  return optix_check(
      optixLaunch(scene->context->pipeline, stream, scene->launch_params.pointer,
                  sizeof(params), &scene->context->sbt, desc->ray_count, 1, 1),
      "optixLaunch");
}

TiForgeOptixResult get_scene_memory(TiForgeOptixTriangleScene raw_scene,
                                    TiForgeOptixSceneMemory *out_memory) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr || out_memory == nullptr ||
      out_memory->struct_size < sizeof(TiForgeOptixSceneMemory)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "invalid OptiX scene memory query");
  }
  out_memory->reserved = 0;
  out_memory->gas_bytes = scene->gas.bytes;
  out_memory->ias_bytes = scene->ias.bytes;
  out_memory->build_update_scratch_bytes =
      scene->scratch.bytes + scene->ias_scratch.bytes;
  out_memory->instance_bytes = scene->instance.bytes;
  out_memory->launch_params_bytes = scene->launch_params.bytes;
  out_memory->shared_pipeline_sbt_bytes =
      scene->context->raygen_record.bytes + scene->context->miss_record.bytes +
      scene->context->hitgroup_record.bytes;
  return TI_FORGE_OPTIX_SUCCESS;
}

TiForgeOptixResult destroy_triangle_scene(
    TiForgeOptixTriangleScene raw_scene) {
  clear_error_state();
  auto *scene = static_cast<Scene *>(raw_scene);
  if (scene == nullptr) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "OptiX triangle scene is null");
  }
  cuCtxSynchronize();
  scene->context->scene_count.fetch_sub(1);
  delete scene;
  return TI_FORGE_OPTIX_SUCCESS;
}

std::size_t get_last_error(char *destination, std::size_t destination_size) {
  const std::size_t required = last_error.size() + 1;
  if (destination != nullptr && destination_size != 0) {
    const std::size_t copied = std::min(last_error.size(), destination_size - 1);
    std::memcpy(destination, last_error.data(), copied);
    destination[copied] = '\0';
  }
  return required;
}

}  // namespace

extern "C" TI_FORGE_OPTIX_EXPORT TiForgeOptixResult
taichi_forge_optix_provider_query(uint32_t requested_abi_version,
                                  size_t api_size,
                                  TiForgeOptixProviderApi *out_api) {
  clear_error_state();
  if (requested_abi_version != TI_FORGE_OPTIX_PROVIDER_ABI_VERSION) {
    return fail(TI_FORGE_OPTIX_ERROR_ABI_MISMATCH,
                "unsupported Forge OptiX provider ABI");
  }
  if (out_api == nullptr || api_size < sizeof(TiForgeOptixProviderApi)) {
    return fail(TI_FORGE_OPTIX_ERROR_INVALID_ARGUMENT,
                "Forge OptiX provider API table is too small");
  }
  std::memset(out_api, 0, sizeof(*out_api));
  out_api->struct_size = sizeof(TiForgeOptixProviderApi);
  out_api->provider_abi_version = TI_FORGE_OPTIX_PROVIDER_ABI_VERSION;
  out_api->info.struct_size = sizeof(TiForgeOptixProviderInfo);
  out_api->info.provider_abi_version = TI_FORGE_OPTIX_PROVIDER_ABI_VERSION;
  out_api->info.optix_abi_version = OPTIX_ABI_VERSION;
  out_api->info.optix_version = OPTIX_VERSION;
  out_api->info.features = kFeatures;
  out_api->info.provider_name = kProviderName;
  out_api->info.build_identity = kBuildIdentity;
  out_api->probe_runtime = probe_runtime;
  out_api->create_context = create_context;
  out_api->destroy_context = destroy_context;
  out_api->create_triangle_scene = create_triangle_scene;
  out_api->update_triangle_scene = update_triangle_scene;
  out_api->trace = trace;
  out_api->get_scene_memory = get_scene_memory;
  out_api->destroy_triangle_scene = destroy_triangle_scene;
  out_api->get_last_error = get_last_error;
  return TI_FORGE_OPTIX_SUCCESS;
}
