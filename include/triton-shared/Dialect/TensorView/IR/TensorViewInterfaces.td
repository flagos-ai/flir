//===-- TensorViewInterfaces.td - TensorView op interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TENSORVIEW_INTERFACES
#define TENSORVIEW_INTERFACES

include "mlir/IR/OpBase.td"

def TensorViewEncodingOpInterface
    : OpInterface<"TensorViewEncodingOpInterface"> {
  let description = [{
    Common interface for operations that attach an encoding to a base
    `tv.tensor_view`.
  }];
  let cppNamespace = "::mlir::triton::tv";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the base tensor view.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getSource">
  ];
}

def ViewOpInterface : OpInterface<"ViewOpInterface"> {
  let description = [{
    Common interface for ops that produce or consume an encoded `tv.tensor_view`
    (i.e. a view carrying a partition/strided/gather-scatter encoding).  The
    tile shape and index-space rank are derived from that encoding.
  }];
  let cppNamespace = "::mlir::triton::tv";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Return the encoded `tensor_view` this op produces (for the
                 `make_*_view` ops) or consumes (for `view_load`/`view_store`).}],
      /*retTy=*/"::mlir::triton::tv::TensorViewType",
      /*methodName=*/"getEncodedView">,
    InterfaceMethod<
      /*desc=*/"Return the view tile shape (== loaded/stored tensor shape).",
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getViewTileShape",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::triton::tv::getEncodingTileShape(
            $_op.getEncodedView().getEncoding());
      }]>,
    InterfaceMethod<
      /*desc=*/"Return the number of index-space (tile-grid) dimensions.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getViewIndexSpaceRank",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::mlir::triton::tv::getEncodingIndexSpaceRank(
            $_op.getEncodedView().getEncoding());
      }]>,
  ];
}

#endif // TENSORVIEW_INTERFACES
