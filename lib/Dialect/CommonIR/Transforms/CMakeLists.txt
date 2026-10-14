add_subdirectory(LaneVectorize)

add_triton_library(CommonIRTransforms
  InferLayout.cpp

  DEPENDS
  CommonIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CommonIRIR
)
