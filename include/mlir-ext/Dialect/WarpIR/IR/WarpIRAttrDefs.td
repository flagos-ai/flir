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

def LinearLayoutParams : AttrOrTypeParameter<"::mlir::triton::gpu::LinearLayout",
                                            "linear layout"> {
  let cppAccessorType = "const ::mlir::triton::gpu::LinearLayout &";
}

def CTAEncodingAttr : WarpIR_Attr<"CTAEncoding", "cta_encoding"> {
  let parameters = (ins AnyType:$linearLayout);

  let description = [{
Describes how blocks (CTAs) in a cooperative thread array (CGA) map onto logical
tensor dimensions. The `LinearLayout` maps from `block` into `dim0`, `dim1`...
  }];

  let extraClassDeclaration = [{
    static CTAEncodingAttr getDefault(MLIRContext *context, int rank);
    // Legacy, we should kill this! Note that it is not true in general that
    // fromSplitParams(enc.getCTAsPerCGA(), enc.getCTASplitNum(), enc.getCTAOrder()) == enc!!
    static CTAEncodingAttr fromSplitParams(MLIRContext *context,
                                           ArrayRef<unsigned> CTAsPerCGA,
                                           ArrayRef<unsigned> CTASplitNum,
                                           ArrayRef<unsigned> CTAOrder);

    // unsigned getRank() const { return getLinearLayout().getNumOutDims(); }
    SmallVector<unsigned> getCTAsPerCGA() const;
    SmallVector<unsigned> getCTASplitNum() const;
    SmallVector<unsigned> getCTAOrder() const;
  }];

  let assemblyFormat = [{
  `<` $linearLayout `>`
  }];
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
  }];

  let parameters = (ins
    DefaultValuedParameter<"int64_t", "32">:$warpSize,
    ArrayRefParameter<"unsigned int">:$threadsPerWarp,
    ArrayRefParameter<"unsigned int">:$elementsPerThread,
    ArrayRefParameter<"unsigned int">:$order
  );

  let assemblyFormat = [{
    `<` `{`
      `warp_size` `=` $warpSize `,`
      `threadsPerWarp` `=` $threadsPerWarp `,`
      `elements_per_thread` `=` $elementsPerThread `,`
      `order` `=` $order
    `}` `>`
  }];

  let genVerifyDecl = 1;
}

#endif // WARP_IR_ATTRDEFS