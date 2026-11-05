#ifndef WARP_IR_TRANSFORMS_PASSES_H
#define WARP_IR_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::tile {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createWarpIRInferLayoutPass();

#define GEN_PASS_DECL
#include "mlir-ext/Dialect/WarpIR/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir-ext/Dialect/WarpIR/Transforms/Passes.h.inc"

} // namespace mlir::triton::tile

#endif // TILE_IR_TRANSFORMS_PASSES_H
