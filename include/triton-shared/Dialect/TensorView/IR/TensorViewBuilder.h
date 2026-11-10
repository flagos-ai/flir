//===- TensorViewBuilder.h - Direct TensorView construction -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_DIALECT_TENSOR_VIEW_BUILDER_H
#define TRITON_SHARED_DIALECT_TENSOR_VIEW_BUILDER_H

#include "triton-shared/Dialect/TensorView/IR/TensorViewAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace triton {
namespace tv {

// Pointee type of a `!tt.ptr<T>`, or null otherwise.
Type getPtrPointeeType(Type type);

// Build an unencoded view from a scalar pointer, shape, and element strides.
FailureOr<Value> createTensorViewBase(OpBuilder &b, Location loc, Value basePtr,
                                      ValueRange shape, ValueRange strides);

// Attach an explicit access encoding to a base view.
Value createPartitionView(OpBuilder &b, Location loc, Value baseView,
                          ArrayRef<int64_t> tile, PaddingValue paddingValue);
Value createStridedView(OpBuilder &b, Location loc, Value baseView,
                        ArrayRef<int64_t> tile,
                        ArrayRef<int64_t> traversalStrides,
                        PaddingValue paddingValue);
Value createGatherScatterView(OpBuilder &b, Location loc, Value baseView,
                              ArrayRef<int64_t> tile,
                              ArrayRef<int64_t> sparseDims,
                              PaddingValue paddingValue);

} // namespace tv
} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_DIALECT_TENSOR_VIEW_BUILDER_H
