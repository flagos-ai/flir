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

#ifndef TRITON_TO_STRUCTURED_PASSES
#define TRITON_TO_STRUCTURED_PASSES

include "mlir/Pass/PassBase.td"

def TritonToStructuredIncubated : Pass<"triton-to-structured-incubated", "mlir::ModuleOp"> {
    let summary = "remove reminder/divider and reproduce addptr/mask expression ";
    let constructor = "triton::createTritonToStructuredIncubatedPass()";
    let options = [
        Option<"enableMaskFallbackConversion", "enable-mask-fallback-conversion", 
            "bool", /*default*/"false",
            "If enabled, select will perform a fallback conversion when mask matching fails.">,
        Option<"optimizeDynamicOffset", "optimize-dynamic-offset", 
            "bool", /*default*/"false",
            "Enable dynamic offset feature">
    ];
}

#endif // TRITON_TO_STRUCTURED_PASSES
