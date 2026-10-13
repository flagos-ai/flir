//===- CommonIRDialect.cpp - CommonIR Dialect registration --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CommonIR dialect, registering all operations,
// attributes, and types.
//
//===----------------------------------------------------------------------===//

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::triton::tile;

namespace {
struct CommonIRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Buffer operations retain their order and SSA identity when the caller
  // inlines them. Call legality (including noinline) belongs to its dialect.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }
};
} // namespace

void CommonIRDialect::initialize() {
  addInterfaces<CommonIRInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROpsAttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRTypes.cpp.inc"
      >();
}

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.cpp.inc"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIREnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROpsAttrDefs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROps.cpp.inc"
