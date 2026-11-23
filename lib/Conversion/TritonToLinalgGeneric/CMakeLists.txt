add_triton_library(TritonToLinalgGeneric
  GenericOpConverter.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
)
