add_triton_library(TritonToLinalgGeneric
  GenericOpConverter.cpp
  GenericUtils.cpp
  GenericDebug.cpp
  FunctionConverter.cpp
  ArgMinMaxConverter.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
)
