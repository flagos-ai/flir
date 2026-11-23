#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h"

#include "mlir-ext/Dialect/CommonIR/IR/LayoutHintAttrs.h"

namespace mlir::triton::tile {
#define GEN_PASS_DEF_COMMONIRCLEARLAYOUTHINTS
#include "mlir-ext/Dialect/CommonIR/Transforms/Passes.h.inc"

namespace {

struct CommonIRClearLayoutHintsPass
    : impl::CommonIRClearLayoutHintsBase<CommonIRClearLayoutHintsPass> {
  void runOnOperation() override { clearLayoutHints(getOperation()); }
};

} // namespace
} // namespace mlir::triton::tile
