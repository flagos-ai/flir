// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonArithToLinalg : Pass<"triton-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert Triton arithmetic operations into linalg";
  let options = [
      Option<"pidsToFuncArgs", "pids-to-func-args", "bool", /*default*/"true",
             "Convert tt.get_program_id and tt.get_num_programs to reference to function arguments">,
      Option<"ttToFuncFunc", "tt-to-func-func", "bool", /*default*/"true",
             "Convert tt.func to func.func">,
      Option<"addptrToLinalg", "addptr-to-linalg", "bool", /*default*/"true",
             "Convert tt.addptr on tensors to linalg">,
      Option<"assertToCf", "assert-to-cf", "bool", /*default*/"true",
             "Convert tt.assert to cf.assert">,
      Option<"tensorPtrToLinalg", "tensor-ptr-to-linalg", "bool", /*default*/"false",
             "Convert triton ops on tensor of pointers to linalg.generic">,
  ];
}

#endif
