add_triton_library(TileIRTransforms
  InferLayout.cpp

  DEPENDS
  TileIRTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TileIRIR
)
