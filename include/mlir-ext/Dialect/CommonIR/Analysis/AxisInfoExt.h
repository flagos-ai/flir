#ifndef COMMONIR_ANALYSIS_AXISINFOEXT_H
#define COMMONIR_ANALYSIS_AXISINFOEXT_H

namespace mlir::triton {
class AxisInfoVisitorList;

namespace tile {

// Extend an existing Triton AxisInfo analysis with logical tensor tile rules.
void addCommonIRAxisInfoVisitors(AxisInfoVisitorList &visitors);

} // namespace tile
} // namespace mlir::triton

#endif
