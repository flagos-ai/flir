add_triton_library(TritonToUnstructured
  TritonToUnstructuredPass.cpp
  AtomicOpsConverter.cpp

  DEPENDS
  TritonStructuredTableGen
  TritonToUnstructuredConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
  TritonTransforms
  TritonSharedAnalysisStructured
  TritonStructuredIR
  TritonSharedUtils
  MLIRMemRefDialect
  MLIRLinalgDialect
  MLIRBufferizationDialect
  MLIRLinalgTransforms
)
