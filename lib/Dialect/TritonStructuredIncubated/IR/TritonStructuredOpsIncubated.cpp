// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>
#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "incubated/Dialect/TritonStructuredIncubated/IR/TritonStructuredDialectIncubated.h"
using namespace mlir;
using namespace mlir::tts::Incubated;

namespace mlir {
namespace tts {
namespace Incubated {

LogicalResult GetStructuredStateOp::verify() {
  auto expectedOffsetAndStrideTypes =
      getOffsetAndStrideTypes(getContext(), getInput().getType());

  if (!expectedOffsetAndStrideTypes.has_value()) {
    return failure();
  }

  auto [expectedOffsetTypes, expectedStrideTypes] =
      *expectedOffsetAndStrideTypes;

  return success(expectedOffsetTypes.size() == getOffsets().size() &&
                 llvm::equal(expectedOffsetTypes, getOffsets().getTypes()) &&
                 expectedStrideTypes.size() == getStrides().size() &&
                 llvm::equal(expectedStrideTypes, getStrides().getTypes()));
}

void GetStructuredStateOp::build(OpBuilder &b, OperationState &state,
                                 Value val) {
  auto type = val.getType();

  // Builder cannot fail, so we default to empty offset and stride types.
  // The invalid op will be rejected by the verifier later.
  auto [offsetTypes, strideTypes] =
      getOffsetAndStrideTypes(b.getContext(), type)
          .value_or(std::make_pair(SmallVector<Type>{}, SmallVector<Type>{}));

  build(b, state, val.getType(), offsetTypes, strideTypes, val);
}

std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
GetStructuredStateOp::getOffsetAndStrideTypes(MLIRContext *context, Type type) {
  auto sizes = getOffsetAndStrideSegmentSizes(type);
  if (!sizes.has_value()) {
    return std::nullopt;
  }
  return std::make_pair(
      SmallVector<Type>(sizes->first, IndexType::get(context)),
      SmallVector<Type>(sizes->second, IndexType::get(context)));
}

std::optional<std::pair<int32_t, int32_t>>
GetStructuredStateOp::getOffsetAndStrideSegmentSizes(Type type) {
  int32_t offsetSegmentSize = 0;
  int32_t strideSegmentSize = 0;

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    if (tensorType.getElementType().isIntOrIndex()) {
      // Tensors of offsets
      // Important note:
      // We only care about tensor of index / int (in addition to pointer type)
      // because only values of int and index type can potentially be part of a
      // pointer arithmetic sequence.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      // Unstructured pointers (tensor<!tt.ptr<type>>)
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    }
  }
  // Block pointers (!tt.ptr<tensor<type>> or !tt.ptr<type>)
  else if (auto ptrType = llvm::dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else {
      // The only relevant state that can be updated in loops for scalar
      // pointers are offset. No need to include stride here.
      offsetSegmentSize = 1;
    }
  } else {
    return std::nullopt;
  }

  return std::make_pair(offsetSegmentSize, strideSegmentSize);
}

} // namespace Incubated
} // namespace tts
} // namespace mlir
