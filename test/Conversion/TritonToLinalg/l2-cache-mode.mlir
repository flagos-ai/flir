// RUN: triton-shared-opt --triton-to-linalg %s --split-input-file | FileCheck %s
// RUN: not triton-shared-opt --triton-to-linalg %s --split-input-file 2>&1 | FileCheck %s --check-prefix=ERROR

module {
  tt.func public @tensor_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    annotation.mark %src_ptr {l2_cache_mode = 4 : i32} : tensor<32x!tt.ptr<f16>>
    %value = tt.load %src_ptr : tensor<32x!tt.ptr<f16>>

    %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    annotation.mark %dst_ptr {l2_cache_mode = 0 : i32} : tensor<32x!tt.ptr<f16>>
    tt.store %dst_ptr, %value : tensor<32x!tt.ptr<f16>>
    tt.return
  }

  // CHECK-LABEL: func.func @tensor_load_store
  // CHECK: memref.copy {{.*}} {l2_cache_mode = 4 : i32}
  // CHECK: bufferization.materialize_in_destination {{.*}} {l2_cache_mode = 0 : i32}

  tt.func public @cache_modifier_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %value = tt.load %src_ptr cacheModifier = l2_disable : tensor<32x!tt.ptr<f16>>
    %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %dst_ptr, %value cacheModifier = l2_disable : tensor<32x!tt.ptr<f16>>
    tt.return
  }

  // CHECK-LABEL: func.func @cache_modifier_load_store
  // CHECK: memref.copy {{.*}} {l2_cache_mode = 4 : i32}
  // CHECK: bufferization.materialize_in_destination {{.*}} {l2_cache_mode = 4 : i32}

  tt.func public @canonicalized_scalar_load(%src: !tt.ptr<f32>) {
    annotation.mark %src {l2_cache_mode = 0 : i64} : !tt.ptr<f32>
    %value = tt.load %src : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: func.func @canonicalized_scalar_load
  // CHECK: annotation.mark {{.*}} {l2_cache_mode = 0 : i64}
  // CHECK: memref.load {{.*}} {l2_cache_mode = 0 : i64}

  tt.func public @masked_load_store(%src: !tt.ptr<f16>, %dst: !tt.ptr<f16>, %limit: i32) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %limit_splat = tt.splat %limit : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<32xi32>
    %zero = arith.constant 0.0 : f16
    %src_splat = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %src_ptr = tt.addptr %src_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    annotation.mark %src_ptr {l2_cache_mode = 3 : i32} : tensor<32x!tt.ptr<f16>>
    %value = tt.load %src_ptr, %mask, %zero : tensor<32x!tt.ptr<f16>>
    %dst_splat = tt.splat %dst : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %dst_ptr = tt.addptr %dst_splat, %range : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    annotation.mark %dst_ptr {l2_cache_mode = 2 : i32} : tensor<32x!tt.ptr<f16>>
    tt.store %dst_ptr, %value, %mask : tensor<32x!tt.ptr<f16>>
    tt.return
  }

  // CHECK-LABEL: func.func @masked_load_store
  // CHECK: memref.subview {{.*}} {l2_cache_mode = 3 : i32}
  // CHECK: memref.copy {{.*}} {l2_cache_mode = 3 : i32}
  // CHECK: memref.subview {{.*}} {l2_cache_mode = 2 : i32}
  // CHECK: bufferization.materialize_in_destination {{.*}} {l2_cache_mode = 2 : i32}
}

// -----

module {
  tt.func public @conflicting_modes(%src: !tt.ptr<f16>) {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %ptrs = tt.splat %src : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    annotation.mark %ptrs {l2_cache_mode = 0 : i32} : tensor<32x!tt.ptr<f16>>
    annotation.mark %ptrs {l2_cache_mode = 1 : i32} : tensor<32x!tt.ptr<f16>>
    %value = tt.load %ptrs : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}

// ERROR: conflicting l2_cache_mode annotations
