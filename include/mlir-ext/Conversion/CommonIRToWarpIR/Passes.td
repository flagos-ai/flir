
include "mlir/Pass/PassBase.td"

def CommonIRToWarpIR : Pass<"commonir-to-warpir", "mlir::ModuleOp"> {
  let summary = "Convert CommonIR to WarpIR";
}
