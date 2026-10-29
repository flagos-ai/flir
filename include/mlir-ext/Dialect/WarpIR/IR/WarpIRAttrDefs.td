//===-- WarpIRAttrDefs.td - WarpIR attribute definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines WarpIR attributes: memory space, data layout, buffer policy,
// pipe types, event IDs, and engine attributes. These are used as operands /
// attributes on WarpIR ops to encode explicit hardware control.
//
//===----------------------------------------------------------------------===//

#ifndef WARP_IR_ATTRDEFS
#define WARP_IR_ATTRDEFS

include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Attribute base class
//===----------------------------------------------------------------------===//



class WarpIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WarpIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def Warp_MmaOperand : I32EnumAttr<"MmaOperand", "MMA operand role", [
  I32EnumAttrCase<"A", 0, "a">,
  I32EnumAttrCase<"B", 1, "b">,
  I32EnumAttrCase<"C", 2, "c">   
]> { let cppNamespace = "::mlir::triton::warp"; let genSpecializedAttr = 0; }

def Warp_MmaKind : I32EnumAttr<"MmaKind", "warp-level MMA instruction family", [
  I32EnumAttrCase<"NvMmaSync",    0, "nv_mma_sync">,    // Ampere/Ada mma.sync
  I32EnumAttrCase<"NvWgmmaSlice", 1, "nv_wgmma_slice">, // Hopper wgmma 
  I32EnumAttrCase<"AmdMfma",      2, "amd_mfma">,       // wave64
  I32EnumAttrCase<"AmdWmma",      3, "amd_wmma">
]> { let cppNamespace = "::mlir::triton::warp"; let genSpecializedAttr = 0; }


//===----------------------------------------------------------------------===//
// Warp distribution
//===----------------------------------------------------------------------===//
def Warp_DistributionAttr : WarpIR_Attr<"Distribution", "distributed"> {
  let summary = "Lane distribution of a warp-local tensor";
  let description = [{
    shape[d] == threadsPerDim[d] * elementsPerThread[d] * repeat[d]
    elementsPerThread[order[0]] contiguous elements are adjacent in memory
    (vectorizable).
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "32">:$warpSize,
    ArrayRefParameter<"int64_t">:$threadsPerDim,
    ArrayRefParameter<"int64_t">:$elementsPerThread,
    ArrayRefParameter<"int64_t">:$repeat,
    ArrayRefParameter<"int64_t">:$order
  );

  let assemblyFormat = [{
    `<` `{`
      `warp_size` `=` $warpSize `,`
      `threads_per_dim` `=` $threadsPerDim `,`
      `elements_per_thread` `=` $elementsPerThread `,`
      `repeat` `=` $repeat `,`
      `order` `=` $order
    `}` `>`
  }];

  let genVerifyDecl = 1;
}


def Warp_MmaAttr : WarpIR_Attr<"Mma", "mma"> {
  let summary = "Warp-local register fragment of an MMA operand";
  let parameters = (ins
    EnumParameter<Warp_MmaKind>:$kind,
    ArrayRefParameter<"int64_t">:$instrShape,   // [m, n, k]
    EnumParameter<Warp_MmaOperand>:$operand,
    "int64_t":$elemBitWidth,   
    ArrayRefParameter<"int64_t">:$repeat,     
    ArrayRefParameter<"int64_t">:$repeatOrder,
    DefaultValuedParameter<"int64_t", "32">:$warpSize
  );

  let assemblyFormat = [{
    `<` `kind` `=` $kind `,` `instr` `=` `[` $instrShape `]` `,`
        `operand` `=` $operand `,` `bits` `=` $elemBitWidth `,`
        `repeat` `=` `[` $repeat `]` `,` `repeatOrder` `=` `[` $repeatOrder `]`
        (`,` `warpSize` `=` $warpSize^)? `>`
  }];

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    SmallVector<int64_t> getTileShape() const;       // 单条指令覆盖的逻辑 tile
    int64_t getElemsPerLaneInInstr() const;
    int64_t getElemsPerReg() const { return 32 / getElemBitWidth(); }
    // WarpLayoutInterface
    int64_t getRegsPerLane(ArrayRef<int64_t> shape) const;
    // LinearLayout toLinearLayout(ArrayRef<int64_t> shape) const;
  }];
}
#endif // WARP_IR_ATTRDEFS