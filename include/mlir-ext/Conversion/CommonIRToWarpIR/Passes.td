
include "mlir/Pass/PassBase.td"

def CommonIRToWarpIR : Pass<"convert-commonir-to-warpir", "mlir::ModuleOp"> {
  let summary = "Convert CommonIR to WarpIR";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::tile::CommonIRDialect"];
  let options = [
      Option<"target", "target",
            "std::string", /*default*/"\"\"",
            "the GPU target, e.g., cuda:80, hip:gfx942">,
      Option<"numWarps", "num-warps",
             "int32_t", /*default*/"4",
             "number of warps">,
      Option<"threadsPerWarp", "threads-per-warp",
             "int32_t", /*default*/"32",
             "number of threads per warp">,
      Option<"numCTAs", "num-ctas",
             "int32_t", /*default*/"1",
             "number of ctas in a cga">,
      Option<"enableSourceRemat", "enable-source-remat",
             "bool", /*default*/"false",
             "enable trivial source rematerialization">,
   ];
}
