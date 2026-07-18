# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
