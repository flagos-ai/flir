//===-- TileIRAttrDefs.td - TileIR attribute definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines TileIR attributes: memory space, data layout, buffer policy,
// pipe types, event IDs, and engine attributes. These are used as operands /
// attributes on TileIR ops to encode explicit hardware control.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_IR_ATTRDEFS
#define TILE_IR_ATTRDEFS

include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//

class TileIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TileIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Memory Space Enum
//===----------------------------------------------------------------------===//

def TileIR_MemorySpace : I64EnumAttr<
  "MemorySpace", "TileIR memory space", [
    I64EnumAttrCase<"GM",  0, "gm">,
    I64EnumAttrCase<"L1",  1, "l1">,
    I64EnumAttrCase<"L0A", 2, "l0a">,
    I64EnumAttrCase<"L0B", 3, "l0b">,
    I64EnumAttrCase<"L0C", 4, "l0c">,
    I64EnumAttrCase<"UB",  5, "ub">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// Data Layout Enum
//===----------------------------------------------------------------------===//

def TileIR_Layout : I64EnumAttr<
  "Layout", "TileIR data layout format", [
    I64EnumAttrCase<"ND",  0, "nd">,
    I64EnumAttrCase<"NZ",  1, "nz">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// Buffer Policy Enum
//===----------------------------------------------------------------------===//

def TileIR_Policy : I64EnumAttr<
  "Policy", "TileIR buffer policy", [
    I64EnumAttrCase<"SingleBuffer", 0, "single_buffer">,
    I64EnumAttrCase<"DoubleBuffer", 1, "double_buffer">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// Engine Enum
//===----------------------------------------------------------------------===//

def TileIR_Engine : I64EnumAttr<
  "Engine", "TileIR engine type", [
    I64EnumAttrCase<"Cube",    0, "cube">,
    I64EnumAttrCase<"Vector",  1, "vector">,
    I64EnumAttrCase<"MTE2",    2, "mte2">,
    I64EnumAttrCase<"MTE3",    3, "mte3">,
    I64EnumAttrCase<"MTE1",    4, "mte1">,
    I64EnumAttrCase<"FixPipe", 5, "fixpipe">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// Pipe Enum (for set_flag / wait_flag / pipe_barrier)
//===----------------------------------------------------------------------===//

def TileIR_Pipe : I64EnumAttr<
  "Pipe", "TileIR pipe type for cross-engine synchronization", [
    I64EnumAttrCase<"PIPE_M",    0, "PIPE_M">,
    I64EnumAttrCase<"PIPE_V",    1, "PIPE_V">,
    I64EnumAttrCase<"PIPE_MTE1", 2, "PIPE_MTE1">,
    I64EnumAttrCase<"PIPE_MTE2", 3, "PIPE_MTE2">,
    I64EnumAttrCase<"PIPE_MTE3", 4, "PIPE_MTE3">,
    I64EnumAttrCase<"PIPE_FIX",  5, "PIPE_FIX">,
    I64EnumAttrCase<"PIPE_S",    6, "PIPE_S">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// Event ID Enum
//===----------------------------------------------------------------------===//

def TileIR_EventID : I64EnumAttr<
  "EventID", "TileIR hardware event IDs for cross-engine synchronization", [
    I64EnumAttrCase<"EVENT_ID0", 0, "EVENT_ID0">,
    I64EnumAttrCase<"EVENT_ID1", 1, "EVENT_ID1">,
    I64EnumAttrCase<"EVENT_ID2", 2, "EVENT_ID2">,
    I64EnumAttrCase<"EVENT_ID3", 3, "EVENT_ID3">,
    I64EnumAttrCase<"EVENT_ID4", 4, "EVENT_ID4">,
    I64EnumAttrCase<"EVENT_ID5", 5, "EVENT_ID5">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

//===----------------------------------------------------------------------===//
// NZ Fractal Layout Attribute (for Cube operands)
//===----------------------------------------------------------------------===//

def TileIR_NZLayoutAttr : TileIR_Attr<"NZLayout", "nz"> {
  let summary = "NZ fractal layout attribute for Cube operands";
  let description = [{
    Specifies the NZ (Z-order) fractal layout used for Cube MMA operands.
    The fractal_h and fractal_w parameters define the fractal tile size
    (typically 16x16 for Ascend 910B).
  }];
  let parameters = (ins
    "int64_t":$fractal_h,
    "int64_t":$fractal_w
  );
  let assemblyFormat = "`<` $fractal_h `x` $fractal_w `>`";
}

//===----------------------------------------------------------------------===//
// Alloc lifetime attribute
//===----------------------------------------------------------------------===//

def TileIR_Lifetime : I64EnumAttr<
  "Lifetime", "TileIR buffer lifetime policy", [
    I64EnumAttrCase<"Default",      0, "default">,
    I64EnumAttrCase<"LoopCarried",  1, "loop_carried">,
  ]> {
  let cppNamespace = "::mlir::triton::tile";
}

#endif // TILE_IR_ATTRDEFS