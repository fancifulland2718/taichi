#pragma once

#include <functional>
#include <memory>
#include <variant>

#include "taichi/aot/graph_data.h"
#include "taichi/rhi/device.h"

namespace taichi::lang::gfx {

// Internal JIT materializer contract. Validation and recording happen once at
// binding publication; replay contains neither a provider call nor this object.
class TI_DLL_EXPORT ExternalGraphCommand {
 public:
  virtual ~ExternalGraphCommand() = default;
  virtual std::vector<aot::Arg> arguments() const = 0;
  virtual void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &args) const = 0;
  virtual void record(Device *device, CommandList *commands) const = 0;
};

struct GraphRecordingSource {
  std::variant<aot::CompiledGraph *, std::shared_ptr<ExternalGraphCommand>>
      value;
};

class GraphReplayRegistration;
class TI_DLL_EXPORT FixedGraphRecording {
 public:
  FixedGraphRecording(Program &program,
                      std::unique_ptr<GraphReplayRegistration> registration);
  ~FixedGraphRecording();
  void run();
  void close();
  std::uint64_t argument_bytes() const;

 private:
  Program *program_;
  mutable std::mutex mutex_;
  std::unique_ptr<GraphReplayRegistration> registration_;
};

// Conversion schema only. This object is never used as an executable or AOT
// artifact; native commands remain separate from serialized Graph dispatches.
TI_DLL_EXPORT aot::CompiledGraph graph_recording_argument_schema(
    const std::vector<GraphRecordingSource> &sources);

}  // namespace taichi::lang::gfx
