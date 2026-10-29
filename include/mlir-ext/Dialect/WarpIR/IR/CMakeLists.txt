set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Ops tablegen
set(LLVM_TARGET_DEFINITIONS WarpIROps.td)
mlir_tablegen(WarpIRDialect.h.inc -gen-dialect-decls -dialect=warp)
mlir_tablegen(WarpIRDialect.cpp.inc -gen-dialect-defs -dialect=warp)
mlir_tablegen(WarpIROps.h.inc -gen-op-decls)
mlir_tablegen(WarpIROps.cpp.inc -gen-op-defs)
add_mlir_doc(WarpIRDialect WarpIRDialect dialects/ -gen-dialect-doc)
add_mlir_doc(WarpIROps WarpIROps dialects/ -gen-op-doc)
add_public_tablegen_target(WarpIRTableGen)

# AttrDefs tablegen
set(LLVM_TARGET_DEFINITIONS WarpIRAttrDefs.td)
mlir_tablegen(WarpIROpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(WarpIROpsAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(WarpIREnums.h.inc -gen-enum-decls)
mlir_tablegen(WarpIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(WarpIRAttrDefsIncGen)

# Types tablegen
set(LLVM_TARGET_DEFINITIONS WarpIRTypes.td)
mlir_tablegen(WarpIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(WarpIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(WarpIRTypesIncGen)