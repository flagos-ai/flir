add_triton_library(LaneVectorize
  LaneVectorize.cpp

  DEPENDS
  LaneVectorizePassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRTensorDialect
  TritonIR
)
