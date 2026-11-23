add_triton_library(TritonToLinalgGeneric
  GenericOpConverter.cpp
  GenericUtils.cpp
  GenericDebug.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
)
