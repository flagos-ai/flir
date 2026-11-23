add_triton_library(CommonIRIR
  CommonIRAttrs.cpp
  CommonIRDialect.cpp
  CommonIROps.cpp
  LayoutHintAttrs.cpp

  DEPENDS
  CommonIRTableGen
  CommonIRAttrDefsIncGen
  CommonIRTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)
