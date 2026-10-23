add_triton_library(CommonIRToWarpIR
  CommonIRToWarpIRPass.cpp
  InferWarpDistribution.cpp
  ConvertCommonIROpPatterns.cpp
  ConvertTritonOpPatterns.cpp
  WarpTypeConverter.cpp

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
