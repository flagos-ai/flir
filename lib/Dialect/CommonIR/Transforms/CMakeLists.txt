add_triton_library(CommonIRTransforms
  InferLayout.cpp
  ClearLayoutHints.cpp
  LaneVectorize.cpp
  LayoutHints.cpp

  DEPENDS
  CommonIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRSCFDialect
  CommonIRIR
  MLIRTransformUtils
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRTensorDialect
  TritonIR
)

target_include_directories(CommonIRTransforms PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../include>
)
