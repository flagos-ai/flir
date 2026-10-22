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
    `$source` is always a genuine `tv.ptr`: every construction site
    (`createTensorViewBase`) retypes or bridges a plain
    `tt.ptr` value to `tv.ptr` itself, via `convertToTensorViewPtr`, before ever
    building this op -- so there is nothing left for a lowering to reconcile
    later.
  }];
  let arguments = (ins
    TV_PtrType:$source,
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

//===----------------------------------------------------------------------===//
// view_load / view_store : load/store a tile of an encoded view as a tensor.
//===----------------------------------------------------------------------===//
def TV_ViewLoadOp : TensorView_Op<"view_load",
    [DeclareOpInterfaceMethods<ViewOpInterface>, MemoryEffects<[MemRead]>]> {
  let summary = "Load the tile at the given index-space position into a tensor.";
  let description = [{
    Elements outside the base view's logical shape are replaced with the
    padding value carried by the view encoding.
  }];
  let arguments = (ins
    TV_TensorViewType:$view,
    Variadic<TV_ViewIndex>:$indices);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` attr-dict `:` qualified(type($view)) `,`
    type($indices) `->` type($result)
  }];
}

def TV_ViewStoreOp : TensorView_Op<"view_store",
    [DeclareOpInterfaceMethods<ViewOpInterface>, MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor into the tile at the given index-space position.";
  let description = [{
    Elements outside the base view's logical shape are not written.
  }];
  let arguments = (ins
    TV_TensorViewType:$view,
    AnyRankedTensor:$value,
    Variadic<TV_ViewIndex>:$indices);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $view `[` $indices `]` `,` $value
    attr-dict `:` qualified(type($view)) `,` type($value) `,` type($indices)
  }];
}

//===----------------------------------------------------------------------===//
// ptr_load / ptr_store : discrete gather/scatter over a tensor of pointers.
//===----------------------------------------------------------------------===//
def TV_PtrLoadOp : TensorView_Op<"ptr_load",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Zipped gather from an N-D tensor of pointers.";
  let description = [{
    For every lane `k`, load from
    `pointers[indices[0][k], ..., indices[R - 1][k]]`, where `R` is the rank
    of `pointers`.
  }];
  let arguments = (ins
    AnyRankedTensor:$pointers,
    Variadic<TV_IndexTensor>:$indices);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $pointers `,` `indices` `=` `[` $indices `]`
    attr-dict `:` type($pointers) `,` type($indices) `->` type($result)
  }];
}

def TV_PtrStoreOp : TensorView_Op<"ptr_store",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Zipped scatter to an N-D tensor of pointers.";
  let description = [{
    For every lane `k`, store `value[k]` to
    `pointers[indices[0][k], ..., indices[R - 1][k]]`, where `R` is the rank
    of `pointers`.
  }];
  let arguments = (ins
    AnyRankedTensor:$pointers,
    AnyRankedTensor:$value,
    Variadic<TV_IndexTensor>:$indices);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $pointers `,` $value `,` `indices` `=` `[` $indices `]`
    attr-dict `:` type($pointers) `,` type($value) `,` type($indices)
  }];
}

#endif // TENSORVIEW_OPS
