//===-- TileIRTypes.td - TileIR type definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines TileIR types: !tile.buf (mutable buffer) and
// !tile.tensor (immutable tensor view), both parameterized by shape,
// element type, and memory space.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_IR_TYPES
#define TILE_IR_TYPES

include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.td"
include "mlir-ext/Dialect/TileIR/IR/TileIRAttrDefs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Type base class
//===----------------------------------------------------------------------===//

class TileIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TileIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// !tile.buf<[shape], dtype, space> — mutable buffer type
//===----------------------------------------------------------------------===//

def TileIR_BufType : TileIR_Type<"Buf", "buf"> {
  let summary = "TileIR mutable buffer type";
  let description = [{
    A !tile.buf represents a mutable buffer allocated in a specific memory space
    (GM, L1, L0A, L0B, L0C, or UB). It carries shape, element type, and
    memory space — enabling the IR to track physical buffer locations.

    Example:
    ```mlir
    !tile.buf<[128,128], f16, #tile.L1>
    !tile.buf<[2,128,128], f16, #tile.L0A>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    EnumParameter<TileIR_MemorySpace>:$memorySpace
  );

  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `,` $memorySpace `>`";
}

//===----------------------------------------------------------------------===//
// !tile.tensor<[shape], dtype, space> — immutable tensor view type
//===----------------------------------------------------------------------===//

def TileIR_TensorType : TileIR_Type<"Tensor", "tensor"> {
  let summary = "TileIR immutable tensor view type";
  let description = [{
    A !tile.tensor represents an immutable view of data in a specific memory
    space, typically used as the result of tile.to_tensor or tile.load.
    Unlike !tile.buf, it is read-only and is used as input to element-wise
    and reduce operations.

    Example:
    ```mlir
    !tile.tensor<[128,128], f32, #tile.UB>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    EnumParameter<TileIR_MemorySpace>:$memorySpace
  );

  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `,` $memorySpace `>`";
}

#endif // TILE_IR_TYPES