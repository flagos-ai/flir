add_triton_library(CommonIRToTritonGPU
  SetLayout.cpp

  LINK_LIBS PUBLIC
  MLIRTransforms
  CommonIRIR
  TritonGPUIR
)
