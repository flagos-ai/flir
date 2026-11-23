add_triton_library(TritonToLinalgGeneric
  GenericOpConverter.cpp
  GenericUtils.cpp
  GenericDebug.cpp
  FunctionConverter.cpp
  ArgMinMaxConverter.cpp
  UseAnalysis.cpp
  HoistBroadcast.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
)
