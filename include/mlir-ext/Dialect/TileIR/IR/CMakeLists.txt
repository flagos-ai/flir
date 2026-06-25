set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Ops tablegen
set(LLVM_TARGET_DEFINITIONS TileIROps.td)
mlir_tablegen(TileIRDialect.h.inc -gen-dialect-decls -dialect=tile)
mlir_tablegen(TileIRDialect.cpp.inc -gen-dialect-defs -dialect=tile)
mlir_tablegen(TileIROps.h.inc -gen-op-decls)
mlir_tablegen(TileIROps.cpp.inc -gen-op-defs)
add_mlir_doc(TileIRDialect TileIRDialect dialects/ -gen-dialect-doc)
add_mlir_doc(TileIROps TileIROps dialects/ -gen-op-doc)
add_public_tablegen_target(TileIRTableGen)

# AttrDefs tablegen
set(LLVM_TARGET_DEFINITIONS TileIRAttrDefs.td)
mlir_tablegen(TileIROpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TileIROpsAttrDefs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(TileIREnums.h.inc -gen-enum-decls)
mlir_tablegen(TileIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TileIRAttrDefsIncGen)

# Types tablegen
set(LLVM_TARGET_DEFINITIONS TileIRTypes.td)
mlir_tablegen(TileIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TileIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(TileIRTypesIncGen)