if(FLAGTREE_TLE_STRUCT)
  set(TleToLinalgLib TleToLinalg)
endif()

add_triton_library(TritonToLinalgIncubated
        TritonToLinalgPass.cpp
        LoadStoreConverter.cpp
        FunctionConverter.cpp
        ArgMinMaxConverter.cpp
        TritonOpConverter.cpp
        HoistBroadcast.cpp
        BlockPtrAnalysis.cpp
        MaskAnalysis.cpp
        UseAnalysis.cpp
        ImplicitPermute.cpp
        DescriptorConverter.cpp
        MarkTensorKindPass.cpp

  DEPENDS
  TritonToLinalgIncubatedConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
  ${TleToLinalgLib}
)
