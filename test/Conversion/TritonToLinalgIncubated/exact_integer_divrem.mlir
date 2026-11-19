// RUN: triton-shared-opt %s --triton-to-linalg-incubated='named-ops=true' --split-input-file | FileCheck %s
// RUN: sed 's/Ascend910B4/Ascend910_9382/g' %s | triton-shared-opt --triton-to-linalg-incubated='named-ops=true' --split-input-file | FileCheck %s
// RUN: triton-shared-opt %s --triton-to-linalg-incubated='named-ops=true compile-on-910-95=true' --split-input-file | FileCheck %s --check-prefix=LEGACY
// REQUIRES: flir-incubated

// CHECK-LABEL: func.func @signed_constant
// CHECK-NOT: arith.divsi
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.sitofp
// CHECK: arith.shrsi
// CHECK: arith.andi
// CHECK-NOT: arith.divsi
// CHECK-NOT: arith.remsi
// CHECK: return
// LEGACY-LABEL: func.func @signed_constant
// LEGACY: arith.divsi
// LEGACY: arith.remsi
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @signed_constant(%x: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %d = arith.constant dense<15> : tensor<8xi32>
    %q = arith.divsi %x, %d : tensor<8xi32>
    %r = arith.remsi %x, %d : tensor<8xi32>
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----
// CHECK-LABEL: func.func @unsigned_constant
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.uitofp
// CHECK: arith.shrsi
// CHECK-NOT: arith.divui
// CHECK-NOT: arith.remui
// CHECK: return
// LEGACY-LABEL: func.func @unsigned_constant
// LEGACY: arith.divui
// LEGACY: arith.remui
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unsigned_constant(%x: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %d = arith.constant dense<-2147483647> : tensor<8xi32>
    %q = arith.divui %x, %d : tensor<8xi32>
    %r = arith.remui %x, %d : tensor<8xi32>
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----
// CHECK-LABEL: func.func @signed_dynamic
// CHECK: arith.extsi {{.*}} : tensor<8xi32> to tensor<8xi64>
// CHECK: arith.divsi {{.*}} : tensor<8xi64>
// CHECK: arith.trunci {{.*}} : tensor<8xi64> to tensor<8xi32>
// CHECK: arith.remsi {{.*}} : tensor<8xi32>
// LEGACY-LABEL: func.func @signed_dynamic
// LEGACY: arith.divsi {{.*}} : tensor<8xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @signed_dynamic(%x: tensor<8xi32>, %d: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %q = arith.divsi %x, %d : tensor<8xi32>
    %r = arith.remsi %x, %d : tensor<8xi32>
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----
// CHECK-LABEL: func.func @unsigned_dynamic
// CHECK: arith.andi
// CHECK: arith.shrsi
// CHECK-NOT: arith.extui
// CHECK: arith.divsi {{.*}} : tensor<8xi64>
// CHECK: arith.trunci {{.*}} : tensor<8xi64> to tensor<8xi32>
// CHECK: arith.muli {{.*}} : tensor<8xi32>
// CHECK: arith.subi {{.*}} : tensor<8xi32>
// LEGACY-LABEL: func.func @unsigned_dynamic
// LEGACY: arith.divui {{.*}} : tensor<8xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unsigned_dynamic(%x: tensor<8xi32>, %d: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %q = arith.divui %x, %d : tensor<8xi32>
    %r = arith.remui %x, %d : tensor<8xi32>
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----
// Undefined division by zero remains unchanged; no new trap behavior.
// CHECK-LABEL: func.func @zero_divisor
// CHECK: arith.divsi
// LEGACY-LABEL: func.func @zero_divisor
// LEGACY: arith.divsi
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @zero_divisor(%x: tensor<8xi32>) -> tensor<8xi32> {
    %d = arith.constant dense<0> : tensor<8xi32>
    %q = arith.divsi %x, %d : tensor<8xi32>
    return %q : tensor<8xi32>
  }
}
