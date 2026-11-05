//===-- WarpIROps.td - WarpIR operation definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all WarpIR operations. WarpIR ops explicitly encode memory
// hierarchy, engine binding, data layout, synchronization, and async compute —
// enabling deterministic 1:1 lowering to HIVM (Ascend NPU).
//
//===----------------------------------------------------------------------===//

#ifndef WARP_IR_OPS
#define WARP_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"



include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.td"
include "mlir-ext/Dialect/WarpIR/IR/WarpIRAttrDefs.td"
include "mlir-ext/Dialect/WarpIR/IR/WarpIRTypes.td"

//===----------------------------------------------------------------------===//
// Op Base
//===----------------------------------------------------------------------===//

class WarpIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<WarpIR_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Memory
//===----------------------------------------------------------------------===//


def WarpIR_LoadOp :
    WarpIR_Op<"load", [MemoryEffects<[MemRead]>,AttrSizedOperandSegments]> {

  let summary = "Cooperative warp-level load";

  let description = [{
    Loads a tensor cooperatively across all lanes of a warp.

    Unlike tt.load, the operation explicitly describes the distribution
    of the loaded elements among warp lanes.
  }];

  let arguments = (ins
    AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other,

    DefaultValuedAttr<DenseI32ArrayAttr, "::llvm::ArrayRef<int32_t>{}">:$boundaryCheck,
    OptionalAttr<TT_PaddingOptionAttr>:$padding,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile,
    // begin flagtree tle
    DefaultValuedAttr<StrAttr, "\"\"">:$flagtree_hints
    // end flagtree tle
  );

  let results = (outs
    TT_Type:$result
  );

}

def Warp_ExpandDimsOp : WarpIR_Op<"expand_dims", [
    Pure,
    SameOperandsAndResultElementType
]> {
  let summary = "Insert a size-1 dimension at the given axis";
  let description = [{
    Expands the shape of `src` by inserting a dimension of size 1 at `axis`.
    The result rank is `rank(src) + 1`, and `result.shape[axis] == 1`.

    Example:
    ```mlir
    %1 = warp.expand_dims %0 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$src,
    I32Attr:$axis
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

  let hasVerifier = 1;
}

def TT_BroadcastOp : WarpIR_Op<"broadcast", [Pure,
                                         SameOperandsAndResultElementType,
                                         SameOperandsAndResultEncoding]> {
    let summary = "broadcast a tensor";

    let description = [{
      For a given tensor, broadcast changes one or more dimensions with size 1
      to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
      change the size of a non-1 dimension.
    }];

    let arguments = (ins AnyRankedTensor:$src);

    let results = (outs AnyRankedTensor:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// Store
//===----------------------------------------------------------------------===//

def WarpIR_StoreOp :
    WarpIR_Op<"store", [MemoryEffects<[MemWrite]>]> {

  let summary = "Cooperative warp-level store";

  let arguments = (ins
    Warp_RegType:$src,
    Warp_PtrType:$dst,
    OptionalAttr<Warp_DistributionAttr>:$layout
  );

  let assemblyFormat = [{
    $src `->` $dst attr-dict
    `:` type($src) `,` type($dst)
  }];
}

def Warp_RegToTensorOp : WarpIR_Op<"reg_to_tensor"> {
  let summary = "Convert a WarpIR register value to a tensor value";

  let arguments = (ins Warp_RegType:$input);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

#endif // WARP_IR_OPS