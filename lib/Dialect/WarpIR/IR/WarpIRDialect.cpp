//===- WarpIRDialect.cpp - WarpIR Dialect registration --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the WarpIR dialect, registering all operations,
// attributes, and types.
//
//===----------------------------------------------------------------------===//

#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::triton::warp;

namespace {
struct WarpOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<mlir::triton::warp::DistributionAttr>(attr)) {
      os << "warp_dist";          // 自动加后缀编号:#warp_dist, #warp_dist1, ...
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace
void WarpIRDialect::initialize() {
  addInterfaces<WarpOpAsmInterface>();
  addOperations<
#define GET_OP_LIST
#include "mlir-ext/Dialect/WarpIR/IR/WarpIROps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir-ext/Dialect/WarpIR/IR/WarpIROpsAttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRTypes.cpp.inc"
      >();
}

#include "mlir-ext/Dialect/WarpIR/IR/WarpIRDialect.cpp.inc"
#include "mlir-ext/Dialect/WarpIR/IR/WarpIREnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir-ext/Dialect/WarpIR/IR/WarpIROpsAttrDefs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-ext/Dialect/WarpIR/IR/WarpIRTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-ext/Dialect/WarpIR/IR/WarpIROps.cpp.inc"