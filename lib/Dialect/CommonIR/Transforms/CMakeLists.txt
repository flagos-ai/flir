add_triton_library(CommonIRTransforms
  InferLayout.cpp
  LaneVectorize.cpp

  DEPENDS
  CommonIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CommonIRIR
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRTensorDialect
  TritonIR
)
