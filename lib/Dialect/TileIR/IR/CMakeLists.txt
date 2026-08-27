add_triton_library(TileIRIR
  TileIRAttrs.cpp
  TileIRDialect.cpp
  TileIROps.cpp

  DEPENDS
  TileIRTableGen
  TileIRAttrDefsIncGen
  TileIRTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
)