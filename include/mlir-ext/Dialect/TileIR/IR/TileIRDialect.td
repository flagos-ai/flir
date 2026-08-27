//===-- TileIRDialect.td - TileIR dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the TileIR dialect definition file. TileIR is a hardware-oriented IR
// that explicitly encodes memory hierarchy, engine binding, data layout, and
// synchronization — enabling deterministic 1:1 lowering to HIVM (Ascend NPU).
//
//===----------------------------------------------------------------------===//

#ifndef TILE_IR_DIALECT
#define TILE_IR_DIALECT

include "mlir/IR/OpBase.td"

def TileIR_Dialect : Dialect {
  let name = "tile";
  let cppNamespace = "::mlir::triton::tile";
  let summary = "The TileIR dialect for explicit hardware control on NPU.";

  let description = [{
    TileIR is a hardware-oriented intermediate representation that encodes:
    - Memory hierarchy (GM/L1/L0A/L0B/L0C/UB)
    - Engine binding (Cube, Vector, MTE2, MTE3)
    - Data layout (ND, NZ fractal)
    - Buffer policy (single/double buffer)
    - Cross-engine synchronization (set_flag/wait_flag/pipe_barrier)
    - Async compute (cube_launch)

    TileIR ops map 1:1 to HIVM instructions, eliminating the need for
    compiler-driven scheduling inference.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
  ];

  let usePropertiesForAttributes = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // TILE_IR_DIALECT