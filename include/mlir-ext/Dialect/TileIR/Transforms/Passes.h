#ifndef TILE_IR_TRANSFORMS_PASSES_H
#define TILE_IR_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::triton::tile {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTileIRInferLayoutPass();

#define GEN_PASS_DECL
#include "mlir-ext/Dialect/TileIR/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir-ext/Dialect/TileIR/Transforms/Passes.h.inc"

} // namespace mlir::triton::tile

#endif // TILE_IR_TRANSFORMS_PASSES_H
