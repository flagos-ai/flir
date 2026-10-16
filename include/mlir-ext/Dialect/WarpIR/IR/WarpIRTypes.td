//===-- WarpIRTypes.td - WarpIR type definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines WarpIR types: !tile.buf (mutable buffer) and
// !tile.tensor (immutable tensor view), both parameterized by shape,
// element type, and memory space.
//
//===----------------------------------------------------------------------===//

#ifndef WARP_IR_TYPES
#define WARP_IR_TYPES

include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.td"
include "mlir-ext/Dialect/WarpIR/IR/WarpIRAttrDefs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Type base class
//===----------------------------------------------------------------------===//

class WarpIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WarpIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


//===----------------------------------------------------------------------===//
// !warp.reg
//===----------------------------------------------------------------------===//

def Warp_RegType : WarpIR_Type<"RegType", "reg"> {
  let summary = "A warp-distributed register fragment";

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    TypeParameter<"mlir::Type", "pointee type">:$pointeeType,
    AttrParameter<"mlir::Attribute", "layout">:$layout
  );

  let assemblyFormat = [{
    `<` `[` $shape `]` `,` $pointeeType `,` $layout `>`
  }];
}

//===----------------------------------------------------------------------===//
// !warp.ptr
//===----------------------------------------------------------------------===//

def Warp_PtrType : WarpIR_Type<"PtrType", "ptr"> {
  let summary = "A pointer used by warp-level memory operations";

  let parameters = (ins
    TypeParameter<"mlir::Type", "pointee type">:$pointeeType,
    AttrParameter<"mlir::Attribute", "memory space">:$memorySpace
  );

  let assemblyFormat = [{
    `<` $pointeeType `,` $memorySpace `>`
  }];
}

#endif // WARP_IR_TYPES