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
    WarpIR_Op<"load", [MemoryEffects<[MemRead]>]> {

  let summary = "Cooperative warp-level load";

  let description = [{
    Loads a tensor cooperatively across all lanes of a warp.

    Unlike tt.load, the operation explicitly describes the distribution
    of the loaded elements among warp lanes.

    Example:

      %a = warp.load %ptr {
        layout = #warp.distributed<
          warp_size = 32,
          threads_per_dim = [8,4],
          elements_per_thread = [2,2],
          order = [1,0]>
      } : !warp.ptr<f16,#warp.global>
          -> !warp.reg<[16,16],f16,#warp.distributed<...>>
  }];

  let arguments = (ins
    Warp_PtrType:$src
  );

  let results = (outs
    Warp_RegType:$result
  );

  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
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

#endif // WARP_IR_OPS