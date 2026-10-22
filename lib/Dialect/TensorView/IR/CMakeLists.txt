add_triton_library(TensorViewIR
  TensorViewDialect.cpp
  TensorViewTypes.cpp
  TensorViewAttrs.cpp
  TensorViewOps.cpp
  TensorViewBuilder.cpp

  DEPENDS
  TensorViewTableGen
  TensorViewTypesIncGen
  TensorViewAttrsIncGen
  TensorViewInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  TritonIR
)
