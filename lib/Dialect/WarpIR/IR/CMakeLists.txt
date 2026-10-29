add_triton_library(WarpIRIR
  WarpIRAttrs.cpp
  WarpIRDialect.cpp
  WarpIROps.cpp

  DEPENDS
  WarpIRTableGen
  WarpIRAttrDefsIncGen
  WarpIRTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)