//===-- WarpIRDialect.td - WarpIR dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the WarpIR dialect definition file. WarpIR is a hardware-oriented IR
// that explicitly encodes memory hierarchy, engine binding, data layout, and
// synchronization — enabling deterministic 1:1 lowering to HIVM (Ascend NPU).
//
//===----------------------------------------------------------------------===//

#ifndef WARP_IR_DIALECT
#define WARP_IR_DIALECT

include "mlir/IR/OpBase.td"


def WarpIR_Dialect : Dialect {
 let name = "warp";
  let cppNamespace = "::mlir::triton::warp";
  let summary = "Warp-level intermediate representation";
  let description = [{
    WarpIR is an intermediate representation between tile-level operations
    and TritonGPU IR.

    WarpIR explicitly represents:
      - warp-cooperative execution
      - lane distribution
      - register fragments
      - warp-level memory operations
      - warp collectives
      - warp-level matrix multiply
      - synchronization

    WarpIR is intended to lower deterministically to TritonGPU IR.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
    "mlir::triton::tile::CommonIRDialect",
  ];

  let usePropertiesForAttributes = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // WARP_IR_DIALECT