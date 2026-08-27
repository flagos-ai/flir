/*
 * Copyright 2026- Xcoresigma Technology Co., Ltd
 */

#ifndef TRITON_ADAPTER_CONVERSION_REDUCESUMSTRENGTHPASS_H
#define TRITON_ADAPTER_CONVERSION_REDUCESUMSTRENGTHPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace triton {
namespace Incubated {

std::unique_ptr<OperationPass<ModuleOp>> createReduceSumStrengthPass();

std::unique_ptr<OperationPass<ModuleOp>>
createReduceSumStrengthPass(bool enable, int32_t splitFactor);

} // namespace Incubated
} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_CONVERSION_REDUCESUMSTRENGTHPASS_H
