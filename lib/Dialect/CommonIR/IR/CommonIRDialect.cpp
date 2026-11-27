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

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::triton::tile;

void CommonIRDialect::initialize() {
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
// The operands are a single ordered variadic group so that generic two-operand
// copies defined before predication remain valid.
static ParseResult parseCopyPredication(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    SmallVectorImpl<Type> &types) {
  if (failed(parser.parseOptionalKeyword("mask")))
    return success();
  if (parser.parseOperand(values.emplace_back()) ||
      parser.parseColonType(types.emplace_back()))
    return failure();
  if (succeeded(parser.parseOptionalKeyword("other")))
    return failure(parser.parseOperand(values.emplace_back()) ||
                   parser.parseColonType(types.emplace_back()));
  return success();
}

static void printCopyPredication(OpAsmPrinter &printer, Operation *,
                                 OperandRange values, TypeRange types) {
  if (!values.empty())
    printer << "mask " << values[0] << " : " << types[0];
  if (values.size() > 1)
    printer << " other " << values[1] << " : " << types[1];
}

#define GET_OP_CLASSES
#include "mlir-ext/Dialect/CommonIR/IR/CommonIROps.cpp.inc"