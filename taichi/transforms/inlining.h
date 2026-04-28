#pragma once

#include "taichi/ir/pass.h"

namespace taichi::lang {

class InliningPass : public Pass {
 public:
  static const PassID id;

  struct Args {
    // P9.A-3 (F3): per-callee statement-count cap.
    //   budget < 0  -> no cap (legacy behavior, inline every FuncCallStmt).
    //   budget == 0 -> disabled (visit no-op; preserves all FuncCallStmt).
    //   budget > 0  -> inline only if callee top-level Block stmt count
    //                  <= budget; else preserve FuncCallStmt.
    // Default -1 = legacy/no-cap so existing callers (inlining_test) keep
    // their semantics without source change.
    int budget{-1};
  };
};

}  // namespace taichi::lang
