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

set(LLVM_TARGET_DEFINITIONS TritonTilingExtOps.td)
mlir_tablegen(TritonTilingExtOpsDialect.h.inc -gen-dialect-decls -dialect=ttx)
mlir_tablegen(TritonTilingExtOpsDialect.cpp.inc -gen-dialect-defs -dialect=ttx)
mlir_tablegen(TritonTilingExtOps.h.inc -gen-op-decls)
mlir_tablegen(TritonTilingExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TritonTilingExtOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TritonTilingExtInterfaces.td)
mlir_tablegen(TritonTilingExtInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TritonTilingExtInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(TritonTilingExtInterfacesIncGen)
