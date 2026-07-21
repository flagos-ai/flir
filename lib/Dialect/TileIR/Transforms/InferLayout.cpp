#include "mlir-ext/Dialect/TileIR/Transforms/Passes.h"

#include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::tile {
#define GEN_PASS_DEF_TILEIRINFERLAYOUT
#include "mlir-ext/Dialect/TileIR/Transforms/Passes.h.inc"
} // namespace mlir::triton::tile

using namespace mlir;
namespace tile = mlir::triton::tile;

namespace {

struct TileIRInferLayoutPass
    : public tile::impl::TileIRInferLayoutBase<TileIRInferLayoutPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto nd = tile::LayoutAttr::get(&getContext(), tile::Layout::ND);

    module.walk([&](tile::AllocOp op) {
      if (!op->hasAttr("layout"))
        op->setAttr("layout", nd);
    });

    module.walk([&](tile::CopyOp op) {
      if (!op->hasAttr("src_layout"))
        op->setAttr("src_layout", nd);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::tile::createTileIRInferLayoutPass() {
  return std::make_unique<TileIRInferLayoutPass>();
}
