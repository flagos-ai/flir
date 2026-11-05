add_triton_library(WarpIRToTritonGPUOrLLVM
  WarpIRToTritonGPUPasses.cpp
  WarpIRToLLVMPasses.cpp
  LoadStoreOpConversion.cpp
  
  DEPENDS
  WarpIRToTritonGPUOrLLVMConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
)