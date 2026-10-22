//===-- TensorViewOps.td - TensorView operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_OPS
#define TENSORVIEW_OPS

include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.td"
include "triton-shared/Dialect/TensorView/IR/TensorViewTypes.td"
include "triton-shared/Dialect/TensorView/IR/TensorViewAttrs.td"
include "triton-shared/Dialect/TensorView/IR/TensorViewInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class TensorView_Op<string mnemonic, list<Trait> traits = []>
    : Op<TensorView_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// make_tensor_view : materialize a base tensor_view from a pointer.
//===----------------------------------------------------------------------===//
def TV_MakeTensorViewOp : TensorView_Op<"make_tensor_view",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "Materialize a base tensor_view (no encoding) from a pointer.";
  let description = [{
    `$source` is the `tt.ptr` the view is based on. `createTensorViewBase`
    passes it through unchanged, so the base pointer keeps the type the rest of
    the Triton pipeline already knows how to lower.

    Note that the Ascend lowering currently requires the source to be a kernel
    argument whose every use is a view construction: that is the form
    `commonir-tv-ptr-legalize` converts to a GM memref. A pointer that also
    feeds ordinary pointer arithmetic is left as a `tt.ptr`, and lowering it
    fails later with a live `memref` / `tt.ptr` materialization.
  }];
  let arguments = (ins
    TT_Ptr:$source,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `,` `sizes` `=` `[` $sizes `]` `,` `strides` `=` `[` $strides `]`
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// make_partition_view / make_strided_view / make_gather_scatter_view :
// attach a view encoding to a base tensor_view.
//===----------------------------------------------------------------------===//
class TV_MakeViewOp<string mnemonic>
    : TensorView_Op<mnemonic, [Pure, TensorViewEncodingOpInterface,
                               DeclareOpInterfaceMethods<ViewOpInterface>]> {
  let arguments = (ins TV_TensorViewType:$source);
  let results = (outs TV_TensorViewType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` qualified(type($source)) `->` qualified(type($result))";
}

def TV_MakePartitionViewOp     : TV_MakeViewOp<"make_partition_view"> {
  let summary = "Attach a #tv.partition_view encoding to a base view.";
}
def TV_MakeStridedViewOp       : TV_MakeViewOp<"make_strided_view"> {
  let summary = "Attach a #tv.strided_view encoding to a base view.";
}
def TV_MakeGatherScatterViewOp : TV_MakeViewOp<"make_gather_scatter_view"> {
  let summary = "Attach a #tv.gather_scatter_view encoding to a base view.";
}

#endif // TENSORVIEW_OPS
