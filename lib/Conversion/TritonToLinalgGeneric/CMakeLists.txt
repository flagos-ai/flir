add_triton_library(TritonToLinalgGeneric
  GenericOpConverter.cpp
  GenericUtils.cpp
  GenericDebug.cpp
  FunctionConverter.cpp
  ArgMinMaxConverter.cpp
  UseAnalysis.cpp
  HoistBroadcast.cpp
  DevicePrintOffsetRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
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
