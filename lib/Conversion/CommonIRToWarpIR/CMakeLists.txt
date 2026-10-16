add_triton_library(CommonIRToWarpIR
  CommonIRToWarpIRPass.cpp

  DEPENDS
  CommonIRToWarpIRConversionPassIncGen

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
