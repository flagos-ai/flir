#include "mlir-ext/Dialect/CommonIR/Transforms/BufferLayoutPropagation.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::tile {

Type BufferLayoutPlan::getType(Value value) const {
  Type type = types.lookup(value);
  return type ? type : value.getType();
}

void BufferLayoutPlan::apply() const {
  for (auto [value, type] : types)
    value.setType(type);
}

Value getBufferLayoutRoot(Value value, const BufferLayoutAdapter &adapter) {
  llvm::DenseSet<Value> visited;
  while (value && visited.insert(value).second) {
    if (adapter.isAllocation(value))
      return value;
    value = adapter.getViewSource(value);
  }
  return {};
}

FailureOr<BufferLayoutPlan>
planBufferLayouts(ArrayRef<BufferLayoutRequirement> requirements,
                  const BufferLayoutAdapter &adapter) {
  llvm::MapVector<Value, Attribute> roots;
  for (auto [value, encoding] : requirements) {
    Value root = getBufferLayoutRoot(value, adapter);
    if (!root)
      return failure();
    while (value != root) {
      encoding = adapter.toSourceEncoding(value, encoding);
      if (!encoding)
        return failure();
      value = adapter.getViewSource(value);
    }
    auto [it, inserted] = roots.try_emplace(root, encoding);
    if (!inserted && it->second != encoding)
      return failure();
  }

  BufferLayoutPlan plan;
  SmallVector<std::pair<Value, Attribute>> worklist(roots.begin(), roots.end());
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto [value, encoding] = worklist[i];
    Type type = adapter.withEncoding(value, encoding);
    if (!type)
      return failure();
    auto [it, inserted] = plan.types.try_emplace(value, type);
    if (!inserted) {
      if (it->second != type)
        return failure();
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getNumResults() == 1 &&
          adapter.getViewSource(user->getResult(0)) == value) {
        Value view = user->getResult(0);
        Attribute projected = adapter.toViewEncoding(view, encoding);
        if (!projected)
          return failure();
        worklist.emplace_back(view, projected);
      } else {
        plan.accesses.push_back(&use);
      }
    }
  }
  return plan;
}

} // namespace mlir::triton::tile
