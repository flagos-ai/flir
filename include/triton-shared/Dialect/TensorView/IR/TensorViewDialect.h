//===- TensorViewDialect.h - MLIR TensorView dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TensorView (`tv`) dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TENSORVIEW_DIALECT_H
#define TRITON_DIALECT_TENSORVIEW_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Dialect/TensorView/IR/TensorViewAttrs.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewTypes.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewInterfaces.h"

#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TensorView/IR/TensorViewOps.h.inc"

#endif // TRITON_DIALECT_TENSORVIEW_DIALECT_H
