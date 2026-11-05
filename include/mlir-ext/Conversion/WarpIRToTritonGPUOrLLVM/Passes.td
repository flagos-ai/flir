
include "mlir/Pass/PassBase.td"

def WarpIRToLLVM : Pass<"convert-warpir-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert WarpIR to LLVM";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect",
                           "mlir::triton::warp::WarpIRDialect"];
}


def WarpIRToTritonGPU : Pass<"convert-warpir-to-ttgir", "mlir::ModuleOp"> {
  let summary = "Convert WarpIR to TritonGPU";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect",
                           "mlir::triton::warp::WarpIRDialect"];
}
