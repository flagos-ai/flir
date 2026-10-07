//===----------------------------------------------------------------------===//
// LaneVectorize.cpp - Lane-parallel operation vectorization (SLP for Triton)
//
// ===========================================================================
// 1. WHAT THIS PASS DOES
// ===========================================================================
//
// Triton kernels frequently contain several INDEPENDENT copies of the same
// computation over the same tensor shape. They come from hand-written
// multi-lane/multi-head code, from `tl.static_range` loops that the frontend
// unrolled, or from frontend-inlined `@triton.jit` helpers. Each copy is a
// separate `tensor<Mxf32>` (or similar) chain, so the backend emits N separate
// narrow tensor computations that cannot use the full width of the hardware.
//
// This pass finds those independent copies (the "lanes"), packs them into a
// single tensor with a synthesized LEADING lane dimension, replays the
// computation once in packed form, and unpacks the results at the boundary.
// It is a form of Superword Level Parallelism (SLP) vectorization.
//
//   BEFORE (2 lanes, tensor<4xf32>)     AFTER (one tensor<2x4xf32>)
//   ----------------------------       --------------------------
//   iter_args(%lane0 = %a0,
//             %lane1 = %a1)              iter_args(%p = pack(%a0, %a1))
//     %r0 = arith.addf %lane0, %y0       %rp = arith.addf %p, %yb
//     %r1 = arith.addf %lane1, %y1  ==>  ...
//     scf.yield %r0, %r1                 scf.yield %rp
//   ...                                ...; unpack(%p) -> lane0, lane1
//
// Per lane i the old IR computes `lane_i = f(x_i)` for i = 0..N-1; the new IR
// computes `packed = f_packed(x_packed)`, where `f_packed` is `f` with the lane
// axis inserted at dimension 0 and every operand either packed (per-lane) or
// broadcast (lane-invariant).
//
// Matching is purely STRUCTURAL -- there is no "softmax", "norm" or "sinkhorn"
// pattern anywhere in this file. An op is lifted exactly when:
//   (a) each of its per-lane instances is the same op over the same operand
//       positions (its "lane images"), and
//   (b) those lane-varying operands are already packed (or, for a leaf,
//       supplied by cone discovery), with every other operand lane-invariant,
// i.e. exactly when the packed form is trivially mathematically equivalent.
//
// An associative combine ACROSS the lanes (e.g. an add tree
// `r = r0 + r1 + ... + rN-1` over the per-lane results) is recognized and
// turned into a `tt.reduce` over the synthesized lane axis -- see
// Packer::liftCrossLane().
//
// ===========================================================================
// 2. LOOP MODE
// ===========================================================================
//
//   rewriteLaneVectorizeLoop: the lanes are the same-typed tensor `iter_args`
//   of one `scf.for`. The loop is rebuilt with a single packed iter_arg; the
//   body is lifted once; results are unpacked and distributed back to the
//   original iter-arg indices. This handles the "normalization fixpoint"
//   shape where the lane-parallel update is loop-carried. The Packer below
//   owns the lifted value mapping and the resolve() classifier.
//
// ===========================================================================
// 3. TERMINOLOGY
// ===========================================================================
//
//   lane            one of the N independent values, e.g. `%lane0` above.
//   lane group      the N values {v_0, ..., v_{N-1}} that are lane images of
//                   each other. Stored as SmallVector<Value>.
//   reference/ref   the lane-0 value; `lanesOf[ref]` is the whole group and
//                   always satisfies lanesOf[ref][0] == ref.
//   cone            the operand-closure of a seed lane group: every op that
//                   produces a group member is part of the cone, and its
//                   lane-varying operands are recursively grouped. Ops outside
//                   the cone are lane-invariant leaves (shared).
//   packed value    a single tensor with the lane axis at dimension 0.
//   shared          lane-invariant: one value used by all lanes; it is
//                   broadcast along the lane axis at each use.
//   cross-lane      an op whose operands are different lanes of the SAME
//                   computation, reduced over the lane axis.
//
// ===========================================================================
// 4. WHERE IT RUNS
// ===========================================================================
//
// Registered as `triton-lane-vectorize` (declared in
// include/triton-shared/Transforms/LaneVectorize/Passes.td) and enabled only in
// the Ascend TTIR pipeline. It runs AFTER `add_inliner` (so per-lane
// `@triton.jit` helpers are already flattened into the cone) and BEFORE `cse`
// / `loop-unroll`, operating on TTIR (`scf`, `tt.reduce`, `arith`, `tensor`).
// The pass is always enabled in the Ascend pipeline.
//
// ===========================================================================
// 5. HARD BOUNDARIES (deliberately conservative)
// ===========================================================================
//
//   * Only side-effect-free "math" ops are lifted. Loads/stores/copies/
//     barriers are boundaries, left for other passes (memory vectorization is
//     out of scope). See isLiftableOp / isMemoryEffectFree checks.
//
//   * Only "compute" types are packed: scalar/vector/tensor of integer, float,
//     index or complex. Pointer/buffer/memref/token types are never packed, so
//     a cone can never materialize a pointer-typed tensor. See isComputeType.
//
//   * `arith.select` (tl.where) is NOT packable even though it is elementwise.
//     It is used to select between the arms of a conditional ping-pong, and
//     packing those arms makes the (Ascend) emitter produce non-zero-offset
//     lane subviews -> dynamic-stride memrefs that the downstream stride-align,
//     PlanMemory and hivmc stages cannot handle. Every other op allowed here is
//     offset-free, so its lane views keep statically unit striding.
//
//   * `tt.reshape allow_reorder` is rejected: element reordering could move
//     data across lanes.
//
// ===========================================================================
// 6. DEBUGGING
// ===========================================================================
//
// Set LANE_VECTORIZE_DEBUG=1 (or true/on/yes) to print the lifted lane groups
// and the packed op for every lift (see dumpLaneGroup); output goes to stderr,
// independent of LLVM_DEBUG.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "triton-shared/Transforms/LaneVectorize/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <vector>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLANEVECTORIZE
#include "triton-shared/Transforms/LaneVectorize/Passes.h.inc"

namespace {

// Diagnostics go to stderr regardless of build type when LANE_VECTORIZE_DEBUG
// is set to a truthy value. This is intentionally independent of LLVM_DEBUG.
bool debugEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("LANE_VECTORIZE_DEBUG");
    if (!value)
      return false;
    llvm::StringRef str(value);
    return str == "1" || str.equals_insensitive("true") ||
           str.equals_insensitive("on") || str.equals_insensitive("yes");
  }();
  return enabled;
}

// Hard bounds that keep a pathological block from blowing up compile time: the
// largest lane group we will synthesize, and the number of successful rewrites
// per block.
// Counters collected during one run and folded into the pass's
// Pass::Statistic members at the end. Keeping them local keeps the free rewrite
// helpers independent of the pass object.
struct LaneVectorizeStats {
  unsigned loopsPacked = 0;
  unsigned lanesPacked = 0;
};

// Prints a value as its defining op (which carries the result type) or, for a
// block argument/constant, the value and its type.
void dumpValue(Value v) {
  if (Operation *def = v.getDefiningOp())
    llvm::errs() << *def;
  else
    llvm::errs() << v << " : " << v.getType();
}

// Prints a compact before/after view of a lane group and its packed result:
//
//   [lane-vectorize] lift: 2 lanes -> tensor<2x4xf32>
//   [lane-vectorize]   | lane0: %r0 = arith.addf %l0, %y : tensor<4xf32>
//   [lane-vectorize]   | lane1: %r1 = arith.addf %l1, %y : tensor<4xf32>
//   [lane-vectorize]   v
//   [lane-vectorize]   %rp = arith.addf %lp, %yb : tensor<2x4xf32>
//
// Only emits output when LANE_VECTORIZE_DEBUG is set.
void dumpLaneGroup(StringRef label, ArrayRef<Value> lanes, Value packed) {
  if (!debugEnabled())
    return;
  llvm::errs() << "[lane-vectorize] " << label << ": " << lanes.size()
               << " lanes -> " << packed.getType() << "\n";
  for (auto [i, lane] : llvm::enumerate(lanes)) {
    llvm::errs() << "  | lane" << i << ": ";
    dumpValue(lane);
    llvm::errs() << "\n";
  }
  llvm::errs() << "  v\n  ";
  dumpValue(packed);
  llvm::errs() << "\n";
}

// ---------------------------------------------------------------------------
// Packing / unpacking primitives
// ---------------------------------------------------------------------------
//
// These helpers turn a lane group into a packed tensor and back. They are the
// ONLY places that know the physical layout of the lane axis; everything else
// works on abstract packings.
//
// packLanes({v0, v1, ..., vN-1}) -> tensor<N x ...> has two paths:
//
//   1. COALESCED (packContiguousSlices): if the lanes are contiguous
//      tensor.extract_slice results of one source, e.g.
//
//        %v0 = extract_slice %src[0][4][1] : tensor<8xf32> -> tensor<4xf32>
//        %v1 = extract_slice %src[4][4][1] : tensor<8xf32> -> tensor<4xf32>
//
//      then emit ONE wide slice + a reshape instead of a concat:
//
//        %w  = extract_slice %src[0][8][1] : tensor<8xf32> -> tensor<8xf32>
//          %p = tensor.reshape %w -> tensor<2x4xf32>
//
//      This is cheaper and leaves the result a strided view of the original
//      buffer (no data movement) in the contiguous case.
//
//   2. GENERIC: reshape each lane to insert a size-1 leading dim, then
//      tensor.concat on dim 0:
//
//          %p = tensor.concat dim(0) (reshape %v0 -> 1x4), (reshape %v1 -> 1x4)
//               : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
//
// unpackLanes is the exact inverse: for each lane, extract_slice the size-1
// leading slice and reshape the singleton dim away.
//
// NOTE: packLanes is used by loop mode on the loop INIT args. The scaffolding
// it emits (reshape/concat/extract_slice) must never be re-packed.

Value buildShapeConst(OpBuilder &builder, Location loc,
                      ArrayRef<int64_t> shape) {
  return builder.create<arith::ConstantOp>(loc,
                                           builder.getI64TensorAttr(shape));
}

// If the lanes are a contiguous run of tensor.extract_slice of one source
// (e.g. src[8], src[12], src[16], src[20]), pack them with a single wider
// slice + reshape instead of a concat. Returns null when it does not apply.
Value packContiguousSlices(OpBuilder &builder, Location loc,
                           ArrayRef<Value> lanes) {
  auto toStatic =
      [](ArrayRef<OpFoldResult> ofrs) -> std::optional<SmallVector<int64_t>> {
    SmallVector<int64_t> out;
    for (OpFoldResult ofr : ofrs) {
      std::optional<int64_t> value = getConstantIntValue(ofr);
      if (!value)
        return std::nullopt;
      out.push_back(*value);
    }
    return out;
  };

  auto first = lanes.front().getDefiningOp<tensor::ExtractSliceOp>();
  if (!first)
    return Value();
  auto firstView = cast<OffsetSizeAndStrideOpInterface>(first.getOperation());
  Value source = first.getSource();
  auto sizes = toStatic(firstView.getMixedSizes());
  auto strides = toStatic(firstView.getMixedStrides());
  auto base = toStatic(firstView.getMixedOffsets());
  if (!sizes || !strides || !base || sizes->empty() || (*strides)[0] != 1)
    return Value();
  int64_t laneExtent = (*sizes)[0];
  if (laneExtent <= 0)
    return Value();

  // Every lane must be the same slice shape/strides, offset only in dim 0 by
  // exactly one lane extent each step, and identical in all other dims.
  for (auto [i, lane] : llvm::enumerate(lanes.drop_front())) {
    auto slice = lane.getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice || slice.getSource() != source)
      return Value();
    auto view = cast<OffsetSizeAndStrideOpInterface>(slice.getOperation());
    auto s = toStatic(view.getMixedSizes());
    auto st = toStatic(view.getMixedStrides());
    auto off = toStatic(view.getMixedOffsets());
    if (!s || !st || !off || *s != *sizes || *st != *strides)
      return Value();
    if ((*off)[0] != (*base)[0] + (int64_t)(i + 1) * laneExtent)
      return Value();
    for (unsigned d = 1; d < off->size(); ++d)
      if ((*off)[d] != (*base)[d])
        return Value();
  }

  auto laneTy = cast<RankedTensorType>(lanes.front().getType());
  int64_t n = (int64_t)lanes.size();

  SmallVector<OpFoldResult> newOffsets;
  for (int64_t o : *base)
    newOffsets.push_back(builder.getIndexAttr(o));
  SmallVector<OpFoldResult> newSizes;
  newSizes.push_back(builder.getIndexAttr(n * laneExtent));
  for (unsigned d = 1; d < sizes->size(); ++d)
    newSizes.push_back(builder.getIndexAttr((*sizes)[d]));
  SmallVector<OpFoldResult> newStrides;
  for (int64_t s : *strides)
    newStrides.push_back(builder.getIndexAttr(s));

  SmallVector<int64_t> wideShape;
  wideShape.push_back(n * laneExtent);
  wideShape.append(laneTy.getShape().begin() + 1, laneTy.getShape().end());
  auto wideTy = RankedTensorType::get(wideShape, laneTy.getElementType(),
                                      laneTy.getEncoding());
  Value wide = builder.create<tensor::ExtractSliceOp>(
      loc, wideTy, source, newOffsets, newSizes, newStrides);

  SmallVector<int64_t> packedShape;
  packedShape.push_back(n);
  packedShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto packedTy = RankedTensorType::get(packedShape, laneTy.getElementType(),
                                        laneTy.getEncoding());
  return builder.create<tensor::ReshapeOp>(
      loc, packedTy, wide, buildShapeConst(builder, loc, packedShape));
}

// Packs `lanes` (all of the same ranked tensor type) into one tensor with a new
// leading dimension of size lanes.size().
Value packLanes(OpBuilder &builder, Location loc, ArrayRef<Value> lanes) {
  // A lane group always holds at least two values; guard anyway so a future
  // caller cannot emit an invalid single-operand tensor.concat.
  if (lanes.size() < 2)
    return Value();
  if (Value coalesced = packContiguousSlices(builder, loc, lanes))
    return coalesced;
  auto laneTy = cast<RankedTensorType>(lanes.front().getType());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto singletonLaneTy = RankedTensorType::get(
      singletonLaneShape, laneTy.getElementType(), laneTy.getEncoding());

  auto reshapeLane = [&](Value lane) -> Value {
    return builder.create<tensor::ReshapeOp>(
        loc, singletonLaneTy, lane,
        buildShapeConst(builder, loc, singletonLaneShape));
  };

  SmallVector<Value> concatOperands;
  concatOperands.reserve(lanes.size());
  for (Value lane : lanes)
    concatOperands.push_back(reshapeLane(lane));

  SmallVector<int64_t> packedShape;
  packedShape.push_back(lanes.size());
  packedShape.append(laneTy.getShape().begin(), laneTy.getShape().end());
  auto packedTy = RankedTensorType::get(packedShape, laneTy.getElementType(),
                                        laneTy.getEncoding());
  return builder.create<tensor::ConcatOp>(loc, packedTy, 0, concatOperands);
}

// Inverse of packLanes: slices the leading lane dimension back into `laneCount`
// individual lane tensors.
SmallVector<Value> unpackLanes(OpBuilder &builder, Location loc, Value packed,
                               unsigned laneCount) {
  SmallVector<Value> lanes;
  lanes.reserve(laneCount);
  auto packedTy = dyn_cast<RankedTensorType>(packed.getType());
  if (!packedTy || packedTy.getRank() < 1 ||
      packedTy.getShape().front() != static_cast<int64_t>(laneCount))
    return lanes;

  SmallVector<int64_t> laneShape(packedTy.getShape().drop_front().begin(),
                                 packedTy.getShape().drop_front().end());
  auto laneTy = RankedTensorType::get(laneShape, packedTy.getElementType(),
                                      packedTy.getEncoding());

  SmallVector<int64_t> singletonLaneShape;
  singletonLaneShape.push_back(1);
  singletonLaneShape.append(laneShape.begin(), laneShape.end());
  auto singletonLaneTy = RankedTensorType::get(
      singletonLaneShape, packedTy.getElementType(), packedTy.getEncoding());

  SmallVector<OpFoldResult> sizes;
  sizes.reserve(packedTy.getRank());
  sizes.push_back(builder.getIndexAttr(1));
  for (int64_t dim : laneShape)
    sizes.push_back(builder.getIndexAttr(dim));
  SmallVector<OpFoldResult> strides(packedTy.getRank(),
                                    builder.getIndexAttr(1));

  for (unsigned laneIdx = 0; laneIdx < laneCount; ++laneIdx) {
    SmallVector<OpFoldResult> offsets(packedTy.getRank(),
                                      builder.getIndexAttr(0));
    offsets[0] = builder.getIndexAttr(laneIdx);
    Value slice = builder.create<tensor::ExtractSliceOp>(
        loc, singletonLaneTy, packed, offsets, sizes, strides);
    Value lane = builder.create<tensor::ReshapeOp>(
        loc, laneTy, slice, buildShapeConst(builder, loc, laneShape));
    lanes.push_back(lane);
  }
  if (debugEnabled()) {
    llvm::errs() << "[lane-vectorize] unpack: ";
    dumpValue(packed);
    llvm::errs() << "\n";
  }
  return lanes;
}

// Erases the defining op of `v` and any operand it uniquely kept alive, once
// each becomes trivially dead. Iterative to avoid deep recursion on long
// operand chains.
void eraseDeadTree(Value v) {
  SmallVector<Value> worklist{v};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val().getDefiningOp();
    if (!op || !isOpTriviallyDead(op))
      continue;
    SmallVector<Value> operands(op->getOperands());
    op->erase();
    llvm::append_range(worklist, operands);
  }
}

// ---------------------------------------------------------------------------
// Packability predicates
// ---------------------------------------------------------------------------
//
// The packer is an ALLOWLIST: an op is lifted only if isLiftableOp() says so.
// The predicates layer, from cheapest to most expensive:
//
//   isComputeType(type)        scalar/vector/tensor of int/float/index/complex?
//   hasOnlyComputeTypes(op)    every operand & result is a compute type?
//   isPackableElementwise(op)  compute-only + not select + elementwise/shape?
//   isAssociativeCombine(op)   a+, a*, max, min, and, or, xor (cross-lane)?
//   isLiftableOp(op)           packable elementwise, OR single-combiner reduce
//
// The shape ops (splat/expand_dims/trans/reshape) are not Elementwise traits
// but ARE packed: their packed form is a cheap retype / axis-shift, handled
// specially in Packer::liftPerLaneOp().

// A "compute" value is a scalar, or a vector/tensor whose element type is a
// number. Pointers, buffers, memrefs and tokens are addressing/memory, not
// math, and must never be materialized in packed form.
bool isComputeType(Type type) {
  if (isa<BaseMemRefType>(type))
    return false;
  Type elem = type;
  if (auto shaped = dyn_cast<ShapedType>(type))
    elem = shaped.getElementType();
  return isa<IntegerType, FloatType, IndexType, ComplexType>(elem);
}

// Packable ops may only produce and consume compute values.
bool hasOnlyComputeTypes(Operation *op) {
  return llvm::all_of(op->getOperands(),
                      [](Value v) { return isComputeType(v.getType()); }) &&
         llvm::all_of(op->getResultTypes(),
                      [](Type t) { return isComputeType(t); });
}

bool isPackableElementwise(Operation *op) {
  // Reject addressing/buffer ops (tt.addptr, tt.ptr_to_int, buffer
  // materializations) so a packed cone can never contain pointer/buffer types.
  if (!hasOnlyComputeTypes(op))
    return false;

  // See the file header: packing tl.where breaks the Ascend backend's
  // stride-align / PlanMemory / hivmc stages (dynamic-stride lane views).
  if (isa<arith::SelectOp>(op))
    return false;

  // An elementwise op applies pointwise and broadcasts operands to the result
  // shape, so adding a leading lane dimension preserves it.
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  return isa<arith::BitcastOp, triton::SplatOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::TransOp, triton::ReshapeOp>(op);
}

// Associative/commutative ops that can be realized as a tt.reduce over the
// synthesized lane axis.
bool isAssociativeCombine(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
             arith::MinNumFOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
             arith::MinUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op);
}

// Ops the packer can lift for a lane group: elementwise/shape ops and tt.reduce
// with an associative single combiner.
bool isLiftableOp(Operation *op) {
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    Operation *combiner = reduce.getSingleCombiner();
    return combiner && isAssociativeCombine(combiner);
  }
  return isPackableElementwise(op) && op->getNumRegions() == 0;
}

// Collects the leaves of an associative tree rooted at `v` (recursing through
// same-name ops only).
void collectAssociativeLeaves(Value v, StringRef opName,
                              SmallVectorImpl<Value> &leaves) {
  if (Operation *def = v.getDefiningOp()) {
    if (def->getName().getStringRef() == opName) {
      for (Value operand : def->getOperands())
        collectAssociativeLeaves(operand, opName, leaves);
      return;
    }
  }
  leaves.push_back(v);
}

// Collects every op of the associative tree rooted at `v`.
void collectAssociativeTreeOps(Value v, StringRef opName,
                               SmallVectorImpl<Operation *> &ops) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != opName)
    return;
  ops.push_back(def);
  for (Value operand : def->getOperands())
    collectAssociativeTreeOps(operand, opName, ops);
}

// ---------------------------------------------------------------------------
// Packed value bookkeeping
// ---------------------------------------------------------------------------
//
// Packing { value, shared } is the result of packing one reference value:
//
//   shared == false: `value` is the packed tensor (lane axis 0) of type
//                    packedTypeOf(ref, n); it holds every lane's copy.
//   shared == true : `value` is lane-invariant. It has the shape of a single
//                    lane (or is a scalar) and must be broadcast at each use;
//                    materialize() inserts expand_dims/splat+broadcast.
//
// packedTypeOf inserts a leading lane dim of size laneCount:
//     f32             -> tensor<N x f32>       (scalar lane)
//     tensor<4xf32>   -> tensor<N x 4xf32>
//     tensor<2x4xf32> -> tensor<N x 2x4xf32>
//
// packedOperandType keeps the operand's element type but takes the result's
// packed shape; needed for e.g. an i1 predicate operand whose result is f32.
//
// widenPacked + materialize reconcile values at different ranks/shapes:
// append size-1 dims (widenPacked) then broadcast (materialize) mirrors
// elementwise broadcast semantics without changing any values.

// The packed form of a value.
struct Packing {
  Value value;
  // A shared value is lane-invariant: it has the shape of a single lane (or is
  // a scalar) and must be broadcast along the lane axis at each use.
  bool shared = false;
};

// The packed type of a per-lane value: an extra leading dim of size laneCount.
RankedTensorType packedTypeOf(Value v, int64_t laneCount) {
  SmallVector<int64_t> shape;
  shape.push_back(laneCount);
  Type elemTy;
  Attribute enc;
  if (auto rt = dyn_cast<RankedTensorType>(v.getType())) {
    llvm::append_range(shape, rt.getShape());
    elemTy = rt.getElementType();
    enc = rt.getEncoding();
  } else {
    elemTy = v.getType();
  }
  return RankedTensorType::get(shape, elemTy, enc);
}

// The packed type an operand of an elementwise op must take: the op broadcasts
// its operands to the result shape, so use the result's packed shape but keep
// the operand's element type (e.g. an i1 select condition becomes
// tensor<[N]x...xi1>).
RankedTensorType packedOperandType(Value operand,
                                   RankedTensorType resultPackedTy) {
  Type elemTy = isa<RankedTensorType>(operand.getType())
                    ? cast<RankedTensorType>(operand.getType()).getElementType()
                    : operand.getType();
  if (elemTy == resultPackedTy.getElementType())
    return resultPackedTy;
  return RankedTensorType::get(resultPackedTy.getShape(), elemTy,
                               resultPackedTy.getEncoding());
}

// Grows an already packed value to `dstTy` by appending size-1 dims and
// broadcasting. Used when a scalar-lane packed value meets a tensor-lane one.
Value widenPacked(OpBuilder &builder, Location loc, Value v,
                  RankedTensorType dstTy) {
  auto srcTy = dyn_cast<RankedTensorType>(v.getType());
  if (!srcTy)
    return Value();
  if (srcTy == dstTy)
    return v;

  Value cur = v;
  auto curTy = srcTy;
  while (curTy.getRank() < dstTy.getRank()) {
    SmallVector<int64_t> shape(curTy.getShape().begin(),
                               curTy.getShape().end());
    shape.push_back(1);
    auto nextTy = RankedTensorType::get(shape, curTy.getElementType(),
                                        curTy.getEncoding());
    cur =
        builder.create<triton::ExpandDimsOp>(loc, nextTy, cur, curTy.getRank());
    curTy = nextTy;
  }
  if (curTy == dstTy)
    return cur;
  return builder.create<triton::BroadcastOp>(loc, dstTy, cur);
}

// Brings a packed/shared value to the packed shape `dstTy`.
Value materialize(OpBuilder &builder, Location loc, Packing pv,
                  RankedTensorType dstTy) {
  if (!pv.shared)
    return widenPacked(builder, loc, pv.value, dstTy);

  if (auto srcTy = dyn_cast<RankedTensorType>(pv.value.getType())) {
    if (srcTy == dstTy)
      return pv.value;
    // Lane-invariant tensor: add the lane axis and broadcast.
    SmallVector<int64_t> shape;
    shape.push_back(1);
    llvm::append_range(shape, srcTy.getShape());
    auto expandedTy = RankedTensorType::get(shape, srcTy.getElementType(),
                                            srcTy.getEncoding());
    Value expanded =
        builder.create<triton::ExpandDimsOp>(loc, expandedTy, pv.value, 0);
    return builder.create<triton::BroadcastOp>(loc, dstTy, expanded);
  }

  // Lane-invariant scalar: broadcast to the packed shape.
  return builder.create<triton::SplatOp>(loc, dstTy, pv.value);
}

// Creates a `tt.reduce` over `src` along `axis` with an empty combine region
// (two scalar block arguments) and leaves the builder at the start of that
// region, ready for the caller to emit the combiner.
triton::ReduceOp createEmptyReduce(OpBuilder &builder, Location loc, Value src,
                                   int64_t axis) {
  auto reduce = builder.create<triton::ReduceOp>(loc, src, axis);
  Block *block = builder.createBlock(&reduce.getCombineOp());
  auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
  builder.setInsertionPointToStart(block);
  return reduce;
}

// Builds a tt.reduce over `src` whose combine region is cloned from `proto`
// (a per-lane reduce).
Value buildReduceFromRegion(OpBuilder &builder, Location loc, Value src,
                            int64_t axis, triton::ReduceOp proto) {
  triton::ReduceOp reduce = createEmptyReduce(builder, loc, src, axis);
  Block *block = &reduce.getCombineOp().front();
  Operation *combiner = proto.getSingleCombiner();
  Block &protoBlock = proto.getCombineOp().front();
  IRMapping mapping;
  mapping.map(protoBlock.getArgument(0), block->getArgument(0));
  mapping.map(protoBlock.getArgument(1), block->getArgument(1));
  Operation *c = builder.clone(*combiner, mapping);
  builder.create<triton::ReduceReturnOp>(loc, c->getResult(0));
  // Callers keep emitting after the reduce, so restore that insertion point.
  builder.setInsertionPointAfter(reduce);
  return reduce->getResult(0);
}

// Builds a tt.reduce over `src` whose combine region computes a single instance
// of `kind` (a cross-lane associative combine).
Value buildReduceFromKind(OpBuilder &builder, Location loc, Value src,
                          int64_t axis, Operation *kind) {
  triton::ReduceOp reduce = createEmptyReduce(builder, loc, src, axis);
  Block *block = &reduce.getCombineOp().front();
  auto elemTy = cast<RankedTensorType>(src.getType()).getElementType();
  OperationState state(loc, kind->getName());
  state.addOperands({block->getArgument(0), block->getArgument(1)});
  state.addTypes(elemTy);
  state.addAttributes(kind->getAttrs());
  Operation *c = builder.create(state);
  builder.create<triton::ReduceReturnOp>(loc, c->getResult(0));
  builder.setInsertionPointAfter(reduce);
  return reduce->getResult(0);
}

// ---------------------------------------------------------------------------
// Structural packer
// ---------------------------------------------------------------------------
//
// Lifts per-lane operations into packed form. State:
//   * lanesOf[ref] — the n per-lane values corresponding to the lane-0
//                    reference `ref` (lanesOf[ref][0] == ref). Filled
//                    incrementally as ops are lifted.
//   * packed[ref]  — the packed form of `ref`, or the value itself with
//                    shared==true when `ref` is lane-invariant.
//   * laneVarying  — every value known to vary per lane, for a fast resolve().
//
// resolve(v) is the single classification entry point for operands of a lifted
// op: it returns the packing to use for `v`, or nullopt when `v` is
// lane-varying but not yet lifted.
//
// ---------------------------------------------------------------------------
// The lifting loop, end to end
// ---------------------------------------------------------------------------
//
//   1. Seed
//        the n same-typed tensor iter args form the first group; packed is
//        seeded from the new loop's packed iter arg.
//   2. Visit ops in program order. For each op, liftOp() classifies it:
//
//        already packed?  ------------- yes --> skip (it is a seed or a leaf)
//              | no
//        effectful?  ------------------ yes --> abort
//              | no
//        all operands shared?  -------- yes --> liftSharedOp  (clone, shared)
//              | no
//        all operands resolved? ------- yes --> liftPerLaneOp
//              |                              (reduce -> liftLaneReduce)
//              | no
//        associative cross-lane tree? - yes --> liftCrossLane (reduce axis 0)
//              | no
//              +----------------------------------> drop the op
//
//   3. Escape
//        validate the yield and build the new yield; unpack the new loop result
//        into the original iter-arg slots.
//
// `packedRefs` is the lift-order list of lane-varying refs (needed by
// liftCrossLane); `sharedRefs` is the list of shared refs it must rewrite;
// `liftedOps` is everything the packer replaced.
// ---------------------------------------------------------------------------

struct Packer {
  // -- configuration --
  scf::ForOp forOp;           // the loop being rewritten
  Operation *scope = nullptr; // values defined outside it are lane-invariant
  OpBuilder &builder;
  Location loc;
  unsigned n;

  // -- state --
  DenseMap<Value, SmallVector<Value>> lanesOf; // ref -> {lane0..laneN-1}
  DenseMap<Value, Packing> packed;             // ref -> packed/shared form
  DenseSet<Value> laneVarying;                 // every lane-varying value
  DenseMap<Value, Value> argRemap;             // old loop args -> new loop args
  SmallVector<unsigned> laneIndices;           // iter-arg indices of the lanes
  SmallVector<Value> packedRefs;          // lane-varying refs in lift order
  SmallVector<Value> sharedRefs;          // cross-lane reduce results (shared)
  SmallPtrSet<Operation *, 32> liftedOps; // original ops replaced

  Packer(scf::ForOp forOp, OpBuilder &builder, unsigned n)
      : forOp(forOp), scope(forOp.getOperation()), builder(builder),
        loc(forOp.getLoc()), n(n) {}

  // Seeds the lane group from the loop's iter args, and remaps the induction
  // variable and every non-lane iter arg to the new loop's args.
  //
  // `packedCurrent` is the new loop's region iter arg 0 (NOT the pre-loop
  // packed init), because the pre-loop value does not dominate the body. The
  // non-lane iter args and the induction variable are recorded in `argRemap`
  // so resolve() hands out the NEW loop's corresponding args; without this, a
  // lifted op that consumed the old iv or a scalar accumulator would reference
  // the erased loop.
  void seed(Value packedCurrent, ArrayRef<unsigned> indices,
            scf::ForOp newFor) {
    laneIndices.assign(indices.begin(), indices.end());
    auto iterArgs = forOp.getRegionIterArgs();

    SmallVector<Value> lanes;
    lanes.reserve(n);
    for (unsigned idx : laneIndices)
      lanes.push_back(iterArgs[idx]);
    Value ref = lanes[0];
    lanesOf[ref] = lanes;
    laneVarying.insert(lanes.begin(), lanes.end());
    packed[ref] = {packedCurrent, /*shared=*/false};
    packedRefs.push_back(ref);

    argRemap[forOp.getInductionVar()] = newFor.getInductionVar();
    auto newIterArgs = newFor.getRegionIterArgs();
    unsigned nextNew = 1; // newIterArgs[0] is the packed lane group
    for (unsigned j = 0; j < iterArgs.size(); ++j) {
      if (llvm::is_contained(laneIndices, j))
        continue;
      argRemap[iterArgs[j]] = newIterArgs[nextNew++];
    }
  }

  // Classifies `v` as an operand of a lifted op:
  //   - packed/seeded:      returns its Packing,
  //   - lane-invariant:     returns {remappedValue, shared=true},
  //   - lane-varying, unresolved: nullopt.
  //
  // Decision order (first match wins):
  //
  //   packed already?                                  -> its Packing
  //   in laneVarying (and not packed)?                 -> nullopt (too early)
  //   BlockArgument?                                   -> shared (remapped)
  //   defined outside `scope` / no defining op?         -> shared
  //   otherwise                                        -> nullopt
  //
  // Returning nullopt is what makes liftOp() defer: the op is not lifted yet,
  // so we fall through to the cross-lane test or drop it. Returning shared for
  // an out-of-scope value is what lets a packed op consume a loop-carried
  // scalar or a value computed before the cone without rematerializing it.
  std::optional<Packing> resolve(Value v) {
    if (auto it = packed.find(v); it != packed.end())
      return it->second;
    if (laneVarying.contains(v))
      return std::nullopt;

    // Block arguments are lane-invariant. Loop body args (induction var and
    // non-lane iter args) are remapped to the new loop so the packed ops they
    // feed dominate their uses.
    if (isa<BlockArgument>(v)) {
      if (auto it = argRemap.find(v); it != argRemap.end())
        return Packing{it->second, /*shared=*/true};
      return Packing{v, /*shared=*/true};
    }

    // Constants and values defined outside the packed scope are lane-invariant.
    Operation *def = v.getDefiningOp();
    if (!def || !scope->isAncestor(def))
      return Packing{v, /*shared=*/true};

    return std::nullopt;
  }

  bool isShared(Value v) {
    auto p = resolve(v);
    return p.has_value() && p->shared;
  }

  bool allOperandsShared(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return isShared(a); });
  }

  bool allOperandsResolved(Operation *op) {
    return llvm::all_of(op->getOperands(),
                        [&](Value a) { return resolve(a).has_value(); });
  }

  // Finds the lane-`lane` counterpart of `op`. Prefers the structural mapping
  // recorded by cone discovery: findLaneOp cannot tell apart structurally
  // identical siblings (e.g. two tt.splat of the same scalar).
  Operation *findSiblingOp(Operation *op, unsigned lane) {
    if (auto it = lanesOf.find(op->getResult(0)); it != lanesOf.end()) {
      Value img = it->second[lane];
      if (img)
        return img.getDefiningOp();
    }
    return findLaneOp(op, lane);
  }

  // Structural search for the lane-`lane` counterpart of `op`, matching by
  // operand list (with lane-varying operands mapped to their lane images).
  //
  // Used when the op has no recorded entry in lanesOf. It looks at the users
  // of the first expected operand and requires an exact operand-list match plus
  // OperationEquivalence over attributes/regions. It can be ambiguous for
  // structurally identical siblings, which is why findSiblingOp() prefers the
  // recorded mapping when one exists.
  Operation *findLaneOp(Operation *op, unsigned lane) {
    SmallVector<Value> expected;
    for (Value a : op->getOperands()) {
      if (isShared(a)) {
        expected.push_back(a);
        continue;
      }
      auto it = lanesOf.find(a);
      if (it == lanesOf.end())
        return nullptr;
      expected.push_back(it->second[lane]);
    }
    if (expected.empty())
      return nullptr;

    for (Operation *u : expected.front().getUsers()) {
      if (u->getName() != op->getName() ||
          u->getNumOperands() != op->getNumOperands() ||
          u->getNumResults() != op->getNumResults())
        continue;
      if (!llvm::equal(u->getOperands(), expected))
        continue;
      if (!OperationEquivalence::isEquivalentTo(
              op, u, OperationEquivalence::ignoreValueEquivalence, nullptr,
              OperationEquivalence::Flags::IgnoreLocations))
        continue;
      return u;
    }
    return nullptr;
  }

  // Clones `op` with `operands` substituted and retypes the results.
  //
  // The clone first inherits the original result types (the mapping only
  // affects operands), then each result is retyped to `resultTypes` -- the
  // packed type. This is safe because every liftable op is elementwise/shape
  // based: adding a leading lane dim changes only the type, never the opcode
  // or attributes (the trans/expand_dims special cases fix up their axis attrs
  // separately).
  Value emitClonedOp(Operation *op, ValueRange operands,
                     TypeRange resultTypes) {
    IRMapping mapping;
    for (auto [from, to] : llvm::zip(op->getOperands(), operands))
      mapping.map(from, to);
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [result, type] : llvm::zip(newOp->getResults(), resultTypes))
      result.setType(type);
    return newOp->getResult(0);
  }

  // Lane-invariant op inside the loop: clone it (with packed operands) and
  // treat the result as shared.
  //
  // This fires when EVERY operand resolves to shared, e.g. `den = addf(r0, r1)`
  // where both r0 and r1 are lane-invariant reductions, or a chain of
  // constants. Since the packed operands for shared values are the (remapped)
  // shared values themselves, the clone computes the same lane-invariant
  // result; it is recorded as shared so later per-lane ops broadcast it instead
  // of packing it.
  LogicalResult liftSharedOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    SmallVector<Value> operands;
    for (Value a : op->getOperands()) {
      auto pv = resolve(a);
      if (!pv)
        return failure();
      operands.push_back(pv->value);
    }
    Value result = emitClonedOp(op, operands, op->getResult(0).getType());
    packed[op->getResult(0)] = {result, /*shared=*/true};
    return success();
  }

  // Records the lifted op `op` (reference) with its per-lane sibling ops.
  //
  // After this call:
  //   lanesOf[ref]      = the full lane group {lane0, ..., laneN-1}
  //   packed[ref]       = the packed value (shared == false)
  //   laneVarying       += lanes 1..N-1
  //   packedRefs        += ref (lift order matters for liftCrossLane)
  //   liftedOps         += the reference and all siblings (all originals the
  //                        rewrite replaced)
  // Dumping the group is gated behind LANE_VECTORIZE_DEBUG.
  void recordLifted(Operation *op, Value packedValue,
                    ArrayRef<Operation *> laneOps) {
    SmallVector<Value> lanes;
    lanes.push_back(op->getResult(0));
    for (Operation *oi : laneOps)
      lanes.push_back(oi->getResult(0));
    dumpLaneGroup("lift", lanes, packedValue);

    lanesOf[op->getResult(0)] = lanes;
    laneVarying.insert(lanes.begin() + 1, lanes.end());
    packed[op->getResult(0)] = {packedValue, /*shared=*/false};
    packedRefs.push_back(op->getResult(0));
    liftedOps.insert(op);
    for (Operation *oi : laneOps)
      liftedOps.insert(oi);
  }

  // A tt.reduce applied independently to every lane -> one tt.reduce over the
  // same axis + 1 (the new lane axis is 0).
  //
  //   reduce(lane0, axis=0) : f32      reduce(packed, axis=1) : tensor<Nxf32>
  //   reduce(lane1, axis=0) : f32  ==>     (the leading axis is the lane axis,
  //   ...                                  so every original axis shifts by 1)
  //
  // The combine region is cloned from the reference reduce, so the user's
  // combiner (addf, maxf, ...) is preserved exactly.
  LogicalResult liftLaneReduce(triton::ReduceOp reduce) {
    if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1)
      return failure();
    Operation *combiner = reduce.getSingleCombiner();
    if (!combiner || !isAssociativeCombine(combiner))
      return failure();
    auto pv = resolve(reduce.getOperand(0));
    if (!pv || pv->shared)
      return failure();

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findSiblingOp(reduce, i);
      if (!oi)
        return failure();
      auto other = dyn_cast<triton::ReduceOp>(oi);
      if (!other || !other.getSingleCombiner() ||
          other.getSingleCombiner()->getName() != combiner->getName())
        return failure();
      laneOps.push_back(oi);
    }

    Value packedValue = buildReduceFromRegion(builder, loc, pv->value,
                                              reduce.getAxis() + 1, reduce);
    recordLifted(reduce, packedValue, laneOps);
    return success();
  }

  // A pointwise/shape op applied independently to every lane -> one packed op.
  //
  //   addf(lane0, y) : tensor<4xf32>     addf(packed, yb) : tensor<Nx4xf32>
  //   addf(lane1, y) : tensor<4xf32>  ==> (yb is y broadcast to Nx4)
  //
  // Operands are first materialized to the packed result shape: per-lane
  // operands are widened/broadcast, shared operands are splat or
  // broadcast after adding the lane axis. Four shape ops need special care:
  //   tt.splat       scalar-lane -> widen the packed source
  //   tt.expand_dims axis += 1 (lane axis inserted at 0)
  //   tt.trans       order is prefixed with 0 and every axis += 1
  //   tt.reshape     cloned as-is (allow_reorder was rejected up front)
  LogicalResult liftPerLaneOp(Operation *op) {
    if (op->getNumResults() != 1)
      return failure();
    if (isa<triton::ReduceOp>(op))
      return liftLaneReduce(cast<triton::ReduceOp>(op));
    if (op->getNumRegions() != 0 || !isPackableElementwise(op))
      return failure();

    SmallVector<Packing> pvs;
    for (Value a : op->getOperands()) {
      auto pv = resolve(a);
      if (!pv)
        return failure();
      pvs.push_back(*pv);
    }

    SmallVector<Operation *> laneOps;
    for (unsigned i = 1; i < n; ++i) {
      Operation *oi = findSiblingOp(op, i);
      if (!oi)
        return failure();
      laneOps.push_back(oi);
    }

    auto resultTy = packedTypeOf(op->getResult(0), n);
    Value packedValue;
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      // Scalar-lane -> tensor-lane: widen the packed source.
      packedValue = widenPacked(builder, loc, pvs[0].value, resultTy);
    } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      packedValue = builder.create<triton::ExpandDimsOp>(
          loc, resultTy, pvs[0].value, expand.getAxis() + 1);
    } else if (auto trans = dyn_cast<triton::TransOp>(op)) {
      // The lane axis is inserted at 0, so every transposed axis shifts by one.
      IRMapping mapping;
      mapping.map(trans.getSrc(), pvs[0].value);
      auto newTrans = cast<triton::TransOp>(builder.clone(*op, mapping));
      newTrans.getResult().setType(resultTy);
      SmallVector<int32_t> order{0};
      for (int32_t axis : trans.getOrder())
        order.push_back(axis + 1);
      newTrans.setOrder(order);
      packedValue = newTrans.getResult();
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      // Element reordering could move data across lanes.
      if (reshape.getAllowReorder())
        return failure();
      packedValue = emitClonedOp(op, {pvs[0].value}, resultTy);
    } else {
      SmallVector<Value> operands;
      for (auto [a, pv] : llvm::zip(op->getOperands(), pvs)) {
        Value v = materialize(builder, loc, pv, packedOperandType(a, resultTy));
        if (!v)
          return failure();
        operands.push_back(v);
      }
      packedValue = emitClonedOp(op, operands, resultTy);
    }
    if (!packedValue)
      return failure();

    recordLifted(op, packedValue, laneOps);
    return success();
  }

  // An associative combine over the lanes (e.g. an add tree) -> a tt.reduce
  // over lane axis 0, plus any lane-invariant leaves folded back in.
  //
  //   %sum = addf(addf(r0, r1), eps)   where r0,r1 are lane-varying and eps is
  //   shared (a tensor that is identical for all lanes)
  //
  //              | reduce the lane-varying leaves over lane axis 0
  //              v
  //   %red = tt.reduce(packed, axis = 0) : tensor<2x4xf32> -> tensor<4xf32>
  //   %sum = addf(%red, eps) : tensor<4xf32>  ;; shared leaves folded back
  //
  // Only trees whose leaves contain EXACTLY the expected lane images (each
  // exactly once) and otherwise only shared values are accepted; anything else
  // is left alone. The reconstructed fold uses the reference op as the template
  // and the combine is applied left-to-right, so it stays associative-safe.
  LogicalResult liftCrossLane(Operation *op) {
    if (op->getNumResults() != 1 || !isAssociativeCombine(op))
      return failure();

    SmallVector<Value> leaves;
    collectAssociativeLeaves(op->getResult(0), op->getName().getStringRef(),
                             leaves);

    for (Value r : packedRefs) {
      auto pvIt = packed.find(r);
      if (pvIt == packed.end() || pvIt->second.shared)
        continue;
      if (!isa<RankedTensorType>(r.getType()))
        continue;

      auto lanesIt = lanesOf.find(r);
      if (lanesIt == lanesOf.end() || lanesIt->second.size() != n)
        continue;
      const SmallVector<Value> &expected = lanesIt->second;

      // Every expected lane image must appear EXACTLY once among the tree's
      // leaves. A repeated lane (e.g. `(v0 + v1) + v0`) must not be collapsed
      // to a single reduction over v0 and v1: that would drop the extra term
      // and change the result.
      llvm::SmallDenseMap<Value, unsigned> counts;
      for (Value l : leaves)
        counts[l]++;
      bool matched = true;
      for (Value e : expected) {
        auto it = counts.find(e);
        if (it == counts.end() || it->second != 1) {
          matched = false;
          break;
        }
      }
      if (!matched)
        continue;

      // All remaining leaves must be lane-invariant.
      SmallPtrSet<void *, 8> expectedSet;
      for (Value e : expected)
        expectedSet.insert(e.getAsOpaquePointer());
      bool restShared = true;
      for (Value l : leaves) {
        if (!expectedSet.contains(l.getAsOpaquePointer()) && !isShared(l)) {
          restShared = false;
          break;
        }
      }
      if (!restShared)
        continue;

      // Reduce over the lane axis, then fold the lane-invariant leaves back in.
      Value reduced =
          buildReduceFromKind(builder, loc, pvIt->second.value, 0, op);
      if (debugEnabled())
        llvm::errs() << "[lane-vectorize] cross-lane reduce: " << *op << "\n"
                     << "  v\n  " << *reduced.getDefiningOp() << "\n";
      SmallVector<Operation *> treeOps;
      collectAssociativeTreeOps(op->getResult(0), op->getName().getStringRef(),
                                treeOps);
      for (Operation *treeOp : treeOps)
        liftedOps.insert(treeOp);
      for (Value l : leaves) {
        if (expectedSet.contains(l.getAsOpaquePointer()))
          continue;
        auto lp = resolve(l);
        if (!lp)
          return failure();
        Value operand = lp->value;
        if (operand.getType() != reduced.getType()) {
          auto dstTy = dyn_cast<RankedTensorType>(reduced.getType());
          if (!dstTy)
            return failure();
          operand = materialize(builder, loc, *lp, dstTy);
          if (!operand)
            return failure();
        }
        IRMapping foldMapping;
        foldMapping.map(op->getOperand(0), reduced);
        foldMapping.map(op->getOperand(1), operand);
        reduced = builder.clone(*op, foldMapping)->getResult(0);
      }
      packed[op->getResult(0)] = {reduced, /*shared=*/true};
      sharedRefs.push_back(op->getResult(0));
      return success();
    }
    return failure();
  }

  // Classifies and lifts one op. See the "lifting loop" diagram above for the
  // dispatch order. A successful return means the op either was replaced or
  // deliberately dropped; failure rolls the loop rewrite back.
  LogicalResult liftOp(Operation *op) {
    // Values already seeded as leaves need no lifting.
    if (op->getNumResults() == 1 && packed.contains(op->getResult(0)))
      return success();

    // Effectful ops are hard boundaries: abort the whole rewrite.
    if (!isMemoryEffectFree(op)) {
      if (debugEnabled())
        llvm::errs() << "[lane-vectorize] reject: effectful op " << *op << "\n";
      return failure();
    }

    if (allOperandsShared(op))
      return liftSharedOp(op);
    if (allOperandsResolved(op))
      return liftPerLaneOp(op);
    if (succeeded(liftCrossLane(op)))
      return success();

    // An unmatched pure op is simply dropped from the packed loop.
    if (debugEnabled())
      llvm::errs() << "[lane-vectorize] drop: unmatched pure op " << *op
                   << "\n";
    return success();
  }

  // Loop mode: lifts the whole body and validates the yield.
  FailureOr<Value> run() {
    for (Operation &op : forOp.getBody()->without_terminator())
      if (failed(liftOp(&op)))
        return failure();

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      return failure();
    Value y0 = yield.getOperand(laneIndices[0]);
    auto it = packed.find(y0);
    if (it == packed.end() || it->second.shared)
      return failure();
    auto lanesIt = lanesOf.find(y0);
    if (lanesIt == lanesOf.end())
      return failure();
    for (unsigned i = 1; i < n; ++i)
      if (lanesIt->second[i] != yield.getOperand(laneIndices[i]))
        return failure();
    return it->second.value;
  }
};

// ---------------------------------------------------------------------------
// Loop mode
// ---------------------------------------------------------------------------
//
// Packs the same-typed tensor `iter_args` of an `scf.for` and replays the body
// once. Worked example (2 lanes, body collapsed to one op for brevity):
//
//   BEFORE
//   ------
//   %0:2 = scf.for %iv = %lb to %ub step %step
//            iter_args(%lane0 = %arg0, %lane1 = %arg1)
//            -> (tensor<4xf32>, tensor<4xf32>) {
//     %r0 = arith.addf %lane0, %y0 : tensor<4xf32>
//     %r1 = arith.addf %lane1, %y1 : tensor<4xf32>
//     scf.yield %r0, %r1 : tensor<4xf32>, tensor<4xf32>
//   }
//
//   AFTER
//   -----
//   %pinit = tensor.concat dim(0)
//              (tensor.reshape %arg0 -> 1x4), (tensor.reshape %arg1 -> 1x4)
//              : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
//   %0 = scf.for %iv = %lb to %ub step %step
//          iter_args(%p = %pinit) -> (tensor<2x4xf32>) {
//     %yb  = tt.broadcast ... : tensor<2x4xf32>       ;; %y0/lane-invariant
//     %rp  = arith.addf %p, %yb : tensor<2x4xf32>
//     scf.yield %rp : tensor<2x4xf32>
//   }
//   %r0 = tensor.reshape (extract_slice %0[0][1x4][1,1]) -> tensor<4xf32>
//   %r1 = tensor.reshape (extract_slice %0[1][1x4][1,1]) -> tensor<4xf32>
//
// The whole rewrite is ALL-OR-NOTHING: a new loop is built first, and if any
// step (body lift, yield validation, unpack) fails, the new loop and the
// packed init are erased and the original loop is left untouched.
// ---------------------------------------------------------------------------

LogicalResult rewriteLaneVectorizeLoop(scf::ForOp forOp,
                                       LaneVectorizeStats &stats) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  auto initArgs = forOp.getInitArgs();
  unsigned m = initArgs.size();
  if (m < 2)
    return failure();

  // Pick the largest group of same-typed tensor iter args to pack; the rest
  // are carried through unchanged. Types are compared exactly (shape, element
  // type, encoding), so only genuinely interchangeable lanes are grouped. At
  // least 2 lanes are required, otherwise there is nothing to vectorize.
  unsigned bestSize = 0;
  Type bestTy;
  for (unsigned i = 0; i < m; ++i) {
    if (!isa<RankedTensorType>(initArgs[i].getType()))
      continue;
    unsigned size = 0;
    for (unsigned j = 0; j < m; ++j)
      if (initArgs[j].getType() == initArgs[i].getType())
        ++size;
    if (size > bestSize) {
      bestSize = size;
      bestTy = initArgs[i].getType();
    }
  }
  if (bestSize < 2)
    return failure();

  SmallVector<unsigned> laneIndices;
  for (unsigned j = 0; j < m; ++j)
    if (initArgs[j].getType() == bestTy)
      laneIndices.push_back(j);
  unsigned n = laneIndices.size();

  SmallVector<Value> initLanes;
  for (unsigned idx : laneIndices)
    initLanes.push_back(initArgs[idx]);
  Value packedInit = packLanes(builder, loc, initLanes);
  if (!packedInit)
    return failure();
  dumpLaneGroup("loop init", initLanes, packedInit);

  SmallVector<unsigned> otherIndices;
  SmallVector<Value> newInitArgs{packedInit};
  for (unsigned j = 0; j < m; ++j) {
    if (llvm::is_contained(laneIndices, j))
      continue;
    otherIndices.push_back(j);
    newInitArgs.push_back(initArgs[j]);
  }

  auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                           forOp.getUpperBound(),
                                           forOp.getStep(), newInitArgs);
  newFor->setAttrs(forOp->getAttrs());

  // New loop signature:
  //   iter_args[0]          = the packed lane group (tensor<N x ...>)
  //   iter_args[1..k]       = the original non-lane iter args, in order
  // The body is then rewritten against these new region iter args.

  // The loop rewrite is all-or-nothing: any failure rolls back to the original
  // loop.
  auto fail = [&]() -> LogicalResult {
    newFor.erase();
    eraseDeadTree(packedInit);
    return failure();
  };

  Packer packer(forOp, builder, n);
  // The body must start from the loop-carried packed value, not the pre-loop
  // packed init.
  packer.seed(newFor.getRegionIterArg(0), laneIndices, newFor);
  builder.setInsertionPointToStart(newFor.getBody());
  FailureOr<Value> packedYield = packer.run();
  if (failed(packedYield))
    return fail();

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  builder.setInsertionPointToEnd(newFor.getBody());
  SmallVector<Value> newYieldOperands{*packedYield};
  // For every non-lane iter arg that the body yielded, resolve() re-applies the
  // loop-arg remap: either the packed body produced a new value for it, or it
  // was carried unchanged and resolves to the new loop's matching iter arg.
  for (unsigned j : otherIndices) {
    auto pv = packer.resolve(oldYield.getOperand(j));
    if (!pv)
      return fail();
    newYieldOperands.push_back(pv->value);
  }
  builder.create<scf::YieldOp>(loc, newYieldOperands);

  builder.setInsertionPointAfter(newFor);
  SmallVector<Value> unpacked =
      unpackLanes(builder, loc, newFor.getResult(0), n);
  if (unpacked.size() != n)
    return fail();

  SmallVector<Value> newResults(m);
  for (unsigned k = 0; k < n; ++k)
    newResults[laneIndices[k]] = unpacked[k];
  for (unsigned k = 0; k < otherIndices.size(); ++k)
    newResults[otherIndices[k]] = newFor.getResult(k + 1);

  for (unsigned j = 0; j < m; ++j)
    forOp.getResult(j).replaceAllUsesWith(newResults[j]);
  ++stats.loopsPacked;
  stats.lanesPacked += n;
  if (debugEnabled())
    llvm::errs() << "[lane-vectorize] packed loop: lanes=" << n << "\n";
  forOp.erase();
  return success();
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------
//
// Loop mode runs on every scf.for in walk order. Each successful rewrite
// rebuilds the loop with a packed iter_arg and replays the body once.
struct LaneVectorizePass
    : public impl::TritonLaneVectorizeBase<LaneVectorizePass> {
  using TritonLaneVectorizeBase::TritonLaneVectorizeBase;

  // Pass::Statistic wraps a std::atomic and is not copyable, which deletes the
  // implicit copy constructor used by clonePass(). Re-construct the base
  // instead of copying it (the statistics re-register against the clone)
  // rather than copy the base's statistics pointers.
  LaneVectorizePass() = default;
  LaneVectorizePass(const LaneVectorizePass &) : TritonLaneVectorizeBase() {}

  Pass::Statistic numLoopsPacked{this, "num-loops-packed",
                                 "Number of scf.for loops packed"};
  Pass::Statistic numLanesPacked{this, "num-lanes-packed",
                                 "Total number of lanes packed"};

  void runOnOperation() override {
    LaneVectorizeStats stats;
    getOperation().walk([&](scf::ForOp forOp) {
      if (succeeded(rewriteLaneVectorizeLoop(forOp, stats)))
        if (debugEnabled())
          llvm::errs() << "[lane-vectorize] packed loop\n";
    });
    numLoopsPacked += stats.loopsPacked;
    numLanesPacked += stats.lanesPacked;
  }
};

} // namespace

} // namespace mlir::triton
