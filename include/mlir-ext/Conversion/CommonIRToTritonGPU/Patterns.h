#ifndef MLIR_EXT_CONVERSION_COMMONIR_TO_TRITON_GPU_PATTERNS_H
#define MLIR_EXT_CONVERSION_COMMONIR_TO_TRITON_GPU_PATTERNS_H

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

namespace triton::tile {

// Lower tile.set_layout to its input or ttg.convert_layout, and require the
// conversion to eliminate all tile.set_layout ops. Layout propagation is
// performed separately by the caller before applying the conversion patterns.
void populateSetLayoutToTritonGPUPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        ConversionTarget &target);

} // namespace triton::tile
} // namespace mlir

#endif
