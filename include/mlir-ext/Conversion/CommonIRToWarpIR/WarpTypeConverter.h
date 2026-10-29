
//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the WarpIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_WARP_TRANSFORMS_WARPCONVERSION_H_
#define TRITON_DIALECT_WARP_TRANSFORMS_WARPCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {
namespace warp {

class WarpTypeConverter : public TypeConverter {
public:
  WarpTypeConverter(MLIRContext *context, int numWarps, int threadsPerWarp,
                    int numCTAs);
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

} // namespace warp
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_WARP_TRANSFORMS_WARPCONVERSION_H_