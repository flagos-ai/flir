if(LLVM_MAJOR_VERSION_22_OR_HIGHER)
  set(TPtrDialectTd TPtrDialect.td)
else()
  set(TPtrDialectTd TPtrDialectForLLVM21.td)
endif()

set(LLVM_TARGET_DEFINITIONS ${TPtrDialectTd})
mlir_tablegen(TPtrDialect.h.inc -gen-dialect-decls -dialect=tptr)
mlir_tablegen(TPtrDialect.cpp.inc -gen-dialect-defs -dialect=tptr)
mlir_tablegen(TPtrOps.h.inc -gen-op-decls)
mlir_tablegen(TPtrOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS ${TPtrDialectTd})
mlir_tablegen(TPtrTypes.h.inc -gen-typedef-decls -typedefs-dialect=tptr)
mlir_tablegen(TPtrTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tptr)

add_public_tablegen_target(TPtrTableGen)
