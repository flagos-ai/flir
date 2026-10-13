add_triton_library(CommonIRAxisInfo
  AxisInfoExt.cpp

  DEPENDS
  CommonIRTableGen

  LINK_LIBS PUBLIC
  CommonIRIR
  TritonAnalysis
)
