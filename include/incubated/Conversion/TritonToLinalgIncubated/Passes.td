#ifndef TRITON_TO_LINALG_CONVERSION_PASSES
#define TRITON_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgIncubated : Pass<"triton-to-linalg-incubated", "mlir::ModuleOp"> {
    let summary = "Convert Triton to Linalg dialect (Incubated)";
    let constructor = "createTritonToLinalgIncubatedPass()";
    let options = [
        Option<"globalKernel", "global-kernel", 
            "bool", /*default*/"true",
            "generate a global kernel">,
        Option<"namedOps", "named-ops", 
            "bool", /*default*/"false",
            "use linalg named ops instead of linalg.generic">,
        Option<"enableNd2nzOnVector", "enable-nd2nz-on-vector", 
            "bool", /*default*/"false",
            "enable nd2nz on vector">,
        Option<"enableSelectAnalysis", "enable-select-analysis",
            "bool", /*default*/"true",
            "enable select analysis">,
        Option<"compileOn91095", "compile-on-910-95", 
            "bool", /*default*/"false",
            "compile on 910_95">
    ];
}

def MarkTensorKind : Pass<"mark-tensor-kind", "mlir::ModuleOp"> {
    let summary = "Mark tensor kind (INPUT/OUTPUT/INPUT_OUTPUT) on Triton function arguments";
    let constructor = "triton::createMarkTensorKindPass()";
}

def ReduceSumStrength : Pass<"reduce-sum-strength", "mlir::ModuleOp"> {
    let summary = "Reduce sum strength for profitable tl.sum-style add reductions";
    let description = [{
        This pass rewrites profitable add reductions by pre-adding two equal halves
        of the reduced dimension and reducing the half-sized tensor. It is owned by
        the FLIR ttir-to-ttadapter pipeline for backends where a full-width reduction
        is more expensive than an elementwise add plus a smaller reduction.
    }];
    let constructor = "triton::Incubated::createReduceSumStrengthPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect"];

    let options = [
        Option<"enable", "enable", "bool", /*default*/"true",
               "Enable the rewrite">,
        Option<"splitFactor", "split-factor", "int32_t", /*default*/"2",
               "Reduction-axis split factor; currently only 2 is supported">
    ];
}

#endif // TRITON_TO_LINALG_CONVERSION_PASSES
