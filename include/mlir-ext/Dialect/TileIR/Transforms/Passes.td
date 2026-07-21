#ifndef TILE_IR_TRANSFORMS_PASSES
#define TILE_IR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def TileIRInferLayout : Pass<"tile-infer-layout", "mlir::ModuleOp"> {
  let summary = "Infer default TileIR layout attributes";
  let description = [{
    Fills missing TileIR layout attributes with a conservative ND layout.
    More target-specific layouts, such as GPU shared-memory swizzles or dot
    operand layouts, can be refined by later target-aware passes.
  }];
  let constructor = "mlir::triton::tile::createTileIRInferLayoutPass()";
}

#endif // TILE_IR_TRANSFORMS_PASSES
