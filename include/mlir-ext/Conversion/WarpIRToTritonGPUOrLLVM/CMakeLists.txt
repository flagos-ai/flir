set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name WarpIRToTritonGPUOrLLVM)
add_public_tablegen_target(WarpIRToTritonGPUOrLLVMConversionPassIncGen)
