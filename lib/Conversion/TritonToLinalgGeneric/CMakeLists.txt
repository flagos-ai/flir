set(TRITON_TO_LINALG_GENERIC_SOURCES
  GenericUtils.cpp
  GenericDebug.cpp
  FunctionConverter.cpp
  ArgMinMaxConverter.cpp
  UseAnalysis.cpp
  HoistBroadcast.cpp
  DevicePrintOffsetRewrite.cpp
  DebugUtils.cpp)

# A backend that keeps its own elementwise pattern implementations in the same
# translation unit as hardware-specific patterns can skip GenericOpConverter to
# avoid duplicate TTOpConverters definitions at link time.
if(NOT TRITON_TO_LINALG_GENERIC_SKIP_OPS)
  list(APPEND TRITON_TO_LINALG_GENERIC_SOURCES GenericOpConverter.cpp)
endif()

add_triton_library(TritonToLinalgGeneric
  ${TRITON_TO_LINALG_GENERIC_SOURCES}

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
)
