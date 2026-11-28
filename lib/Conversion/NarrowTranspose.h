#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::Incubated {

inline constexpr llvm::StringLiteral kOriginalTransposeOrder =
    "tt.original_transpose_order";
inline constexpr llvm::StringLiteral kOriginalTransposeOrdersRecorded =
    "tt.original_transpose_orders_recorded";

} // namespace mlir::triton::Incubated
