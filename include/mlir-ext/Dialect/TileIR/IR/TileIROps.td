//===-- TileIROps.td - TileIR operation definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all TileIR operations. TileIR ops explicitly encode memory
// hierarchy, engine binding, data layout, synchronization, and async compute —
// enabling deterministic 1:1 lowering to HIVM (Ascend NPU).
//
//===----------------------------------------------------------------------===//

#ifndef TILE_IR_OPS
#define TILE_IR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

include "mlir-ext/Dialect/TileIR/IR/TileIRDialect.td"
include "mlir-ext/Dialect/TileIR/IR/TileIRAttrDefs.td"
include "mlir-ext/Dialect/TileIR/IR/TileIRTypes.td"

//===----------------------------------------------------------------------===//
// Op Base
//===----------------------------------------------------------------------===//

class TileIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<TileIR_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// tile.alloc — Allocate a buffer in a specific memory space
// ---------------------------------------------------------------------------
def TileIR_AllocOp : TileIR_Op<"alloc", [Pure]> {
  let summary = "Allocate a buffer in a specific memory space";
  let description = [{
    Allocates a buffer with given shape, dtype, and memory space attributes.
    Returns a !tile.buf value that can be used by copy/subview/cube_launch ops.

    Example:
    ```mlir
    %buf = tile.alloc {space=#tile.UB, shape=[128,128], dtype=f32,
                       policy=#tile.single_buffer, layout=#tile.nd,
                       lifetime="loop_carried"}
             : !tile.buf<[128,128], f32, #tile.UB>
    ```
  }];

  let arguments = (ins
    TileIR_MemorySpace:$space,
    OptionalAttr<I64ArrayAttr>:$shape,
    OptionalAttr<TypeAttr>:$dtype,
    OptionalAttr<TileIR_Policy>:$policy,
    OptionalAttr<TileIR_Layout>:$layout,
    OptionalAttr<TileIR_Lifetime>:$lifetime,
    OptionalAttr<StrAttr>:$comment
  );

  let results = (outs TileIR_BufType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

// ---------------------------------------------------------------------------
// tile.subview — Extract a sub-view (slice) from a buffer
// ---------------------------------------------------------------------------
def TileIR_SubViewOp : TileIR_Op<"subview", [Pure]> {
  let summary = "Extract a sub-view (slice) from a buffer";
  let description = [{
    Creates a zero-copy view into a portion of a buffer. Used for slot
    selection from multi-buffered buffers (e.g., ping-pong slot extraction).

    Example:
    ```mlir
    %slot = tile.subview %buf[%off, %c0, %c0] [1, 128, 128] [1, 1, 1]
              : !tile.buf<[2,128,128],f16,#tile.L0A> -> !tile.buf<[128,128],f16,#tile.L0A>
    ```
  }];

  let arguments = (ins
    TileIR_BufType:$source,
    Variadic<Index>:$offsets,
    I64ArrayAttr:$sizes,
    I64ArrayAttr:$strides
  );

  let results = (outs TileIR_BufType:$result);

  let assemblyFormat = [{
    $source `[` $offsets `]` `[` $sizes `]` `[` $strides `]`
    attr-dict `:` type($source) `->` type($result)
  }];
}

// ---------------------------------------------------------------------------
// tile.gm_offset — Compute global memory address with strides
// ---------------------------------------------------------------------------
def TileIR_GmOffsetOp : TileIR_Op<"gm_offset", [Pure, SameVariadicOperandSize]> {
  let summary = "Compute global memory offset with multi-dimensional strides";
  let description = [{
    Computes a GM pointer from a base pointer and multi-dimensional indices
    with explicit strides. Replaces raw pointer arithmetic. Each index is
    paired with a stride, so `indices` and `strides` must have the same arity.

    Example:
    ```mlir
    %off = tile.gm_offset %Q [%bz, %by, %row, %c0] strides [%sQb, %sQh, %sQs, %sQd]
             : !tt.ptr<f16> -> !tt.ptr<f16>
    ```
  }];

  let arguments = (ins
    TT_Ptr:$base,
    Variadic<Index>:$indices,
    Variadic<Index>:$strides
  );

  let results = (outs TT_Ptr:$result);

  let assemblyFormat = [{
    $base `[` $indices `]` `strides` `[` $strides `]` attr-dict
    `:` type($base) `->` type($result)
  }];
}

// ---------------------------------------------------------------------------
// tile.copy — Copy data between memory regions with layout conversion
// ---------------------------------------------------------------------------
def TileIR_CopyOp : TileIR_Op<"copy", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Copy data between memory regions with optional layout conversion";
  let description = [{
    Copies data from source to destination, optionally performing layout
    conversion (e.g., ND→NZ for Cube operands) and transpose.

    Example:
    ```mlir
    tile.copy %src → %dst {engine=#tile.mte2, src_layout=#tile.nd,
               dst_layout=#tile.nz<16,16>, transpose=true}
    ```
  }];

  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    OptionalAttr<TileIR_Engine>:$engine,
    OptionalAttr<TileIR_Layout>:$src_layout,
    OptionalAttr<TileIR_NZLayoutAttr>:$dst_nz_layout,
    OptionalAttr<UnitAttr>:$transpose,
    OptionalAttr<StrAttr>:$comment
  );

  let assemblyFormat = [{
    $src `->` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

// ---------------------------------------------------------------------------
// tile.load — Load data from GM to local memory
// ---------------------------------------------------------------------------
def TileIR_LoadOp : TileIR_Op<"load", [Pure, MemoryEffects<[MemRead]>]> {
  let summary = "Load data from global memory to local memory";
  let description = [{
    Loads a tensor from a GM pointer into the specified destination memory
    space (typically UB for Vector compute).

    Example:
    ```mlir
    %t = tile.load %ptr {engine=#tile.mte2, dst_space=#tile.UB}
          : !tt.ptr<f32> -> !tile.tensor<[128,128],f32,#tile.UB>
    ```
  }];

  let arguments = (ins
    TT_Ptr:$src,
    OptionalAttr<TileIR_Engine>:$engine,
    OptionalAttr<TileIR_MemorySpace>:$dst_space
  );

  let results = (outs TileIR_TensorType:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// ---------------------------------------------------------------------------
// tile.store — Store data from local memory to GM
// ---------------------------------------------------------------------------
def TileIR_StoreOp : TileIR_Op<"store", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Store data from local memory to global memory";
  let description = [{
    Stores a tensor from local memory (e.g., UB) to a GM pointer destination.

    Example:
    ```mlir
    tile.store %tensor → %ptr {engine=#tile.mte3, src_layout=#tile.nd,
               dst_layout=#tile.nd}
    ```
  }];

  let arguments = (ins
    AnyType:$src,
    TT_Ptr:$dst,
    OptionalAttr<TileIR_Engine>:$engine,
    OptionalAttr<TileIR_Layout>:$src_layout,
    OptionalAttr<TileIR_Layout>:$dst_layout,
    OptionalAttr<StrAttr>:$comment
  );

  let assemblyFormat = [{
    $src `->` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

// ---------------------------------------------------------------------------
// tile.vec_fill — Fill a buffer with a scalar value (Vector engine)
// ---------------------------------------------------------------------------
def TileIR_VecFillOp : TileIR_Op<"vec_fill", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Fill a buffer with a scalar value on the Vector engine";
  let description = [{
    Initializes a buffer by broadcasting a scalar value. Typically used to
    reset running max/sum states or zero-fill accumulators.

    Example:
    ```mlir
    tile.vec_fill %buf, %ZERO_F32 {compute_unit=#tile.vector_core}
    ```
  }];

  let arguments = (ins
    AnyType:$buf,
    AnyType:$value,
    OptionalAttr<TileIR_Engine>:$compute_unit
  );

  let assemblyFormat = "$buf `,` $value attr-dict `:` type($buf) `,` type($value)";
}

//===----------------------------------------------------------------------===//
// Compute Operations
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// tile.cube_launch — Async matrix multiply on Cube engine
// ---------------------------------------------------------------------------
def TileIR_CubeLaunchOp : TileIR_Op<"cube_launch", [MemoryEffects<[MemWrite]>]> {
  let summary = "Launch async matrix multiply on the Cube engine";
  let description = [{
    Asynchronously launches a matrix multiplication on the Cube engine.
    The operands a/b are from L1, acc from L0C, stage_a/stage_b from L0A/L0B
    (data staging), and dst is the GM output pointer.

    Returns a token that can be used with tile.cube_wait to synchronize.

    Example:
    ```mlir
    tile.cube_launch a(%q_l1) b(%k_l1) acc(%l0c0)
      stage_a(%l0a0) stage_b(%l0b0) dst(%m1_slot)
      {transpose_b=true, init=true, mma="128x128x128_f32_f16_f16"}
      : (!tile.buf<...>, !tile.buf<...>, ..., !tt.ptr<f32>)
    ```
  }];

  let arguments = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$acc,
    AnyType:$stage_a,
    AnyType:$stage_b,
    TT_Ptr:$dst,
    OptionalAttr<UnitAttr>:$transpose_a,
    OptionalAttr<UnitAttr>:$transpose_b,
    OptionalAttr<UnitAttr>:$init,
    OptionalAttr<StrAttr>:$mma,
    OptionalAttr<StrAttr>:$comment
  );

  let assemblyFormat = [{
    `a` `(` $a `)` `b` `(` $b `)` `acc` `(` $acc `)`
    `stage_a` `(` $stage_a `)` `stage_b` `(` $stage_b `)` `dst` `(` $dst `)`
    attr-dict `:` `(` type($a) `,` type($b) `,` type($acc) `,`
    type($stage_a) `,` type($stage_b) `,` type($dst) `)`
  }];
}

// ---------------------------------------------------------------------------
// tile.cube_wait — Wait for Cube async operation completion
// ---------------------------------------------------------------------------
def TileIR_CubeWaitOp : TileIR_Op<"cube_wait", []> {
  let summary = "Wait for a Cube async operation to complete";
  let description = [{
    Blocks until the Cube async operation identified by the token completes.
    Used to drain the pipeline after cube_launch.

    Example:
    ```mlir
    tile.cube_wait %token
    ```
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
}

// ---------------------------------------------------------------------------
// tile.reduce — Reduce a tensor along an axis
// ---------------------------------------------------------------------------
def TileIR_ReduceOp : TileIR_Op<"reduce", [Pure]> {
  let summary = "Reduce a tensor along a specified axis";
  let description = [{
    Performs a reduction along a given axis using the specified arithmetic
    operation. The scope attribute specifies which engine performs the reduce.

    Example:
    ```mlir
    %res = tile.reduce %tensor {axis=1, op=@arith.maximumf, scope=#tile.vector}
             : !tile.tensor<[128,128],f32,#tile.UB> -> !tile.tensor<[128,1],f32,#tile.UB>
    ```
  }];

  let arguments = (ins
    TileIR_TensorType:$src,
    I64Attr:$axis,
    FlatSymbolRefAttr:$op,
    OptionalAttr<TileIR_Engine>:$scope
  );

  let results = (outs TileIR_TensorType:$result);

  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
}

// ---------------------------------------------------------------------------
// tile.elemwise — Element-wise operation
// ---------------------------------------------------------------------------
def TileIR_ElemwiseOp : TileIR_Op<"elemwise", [Pure]> {
  let summary = "Element-wise operation (math function applied to tensor)";
  let description = [{
    Applies an element-wise operation (e.g., @math.exp) to input tensors.
    The function symbol references an MLIR standard function.

    Example:
    ```mlir
    %res = tile.elemwise @math.exp(%arg)
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<TileIR_TensorType>:$inputs
  );

  let results = (outs TileIR_TensorType:$result);

  let assemblyFormat = "$func `(` $inputs `)` attr-dict `:` functional-type($inputs, $result)";
}

// ---------------------------------------------------------------------------
// tile.broadcast — Broadcast a tensor to a larger shape
// ---------------------------------------------------------------------------
def TileIR_BroadcastOp : TileIR_Op<"broadcast", [Pure]> {
  let summary = "Broadcast a tensor to a larger shape along an axis";
  let description = [{
    Broadcasts a tensor along a specified axis to match a target shape.
    Used to expand (e.g., [128,1] → [128,128]) for element-wise operations.

    Example:
    ```mlir
    %bc = tile.broadcast %tensor {axis=1, shape=[128,128]}
            : !tile.tensor<[128,1],f32,#tile.UB> -> !tile.tensor<[128,128],f32,#tile.UB>
    ```
  }];

  let arguments = (ins
    TileIR_TensorType:$src,
    I64Attr:$axis,
    I64ArrayAttr:$shape
  );

  let results = (outs TileIR_TensorType:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// ---------------------------------------------------------------------------
// tile.cast — Type conversion (e.g., f32 → f16)
// ---------------------------------------------------------------------------
def TileIR_CastOp : TileIR_Op<"cast", [Pure]> {
  let summary = "Type conversion between element types";
  let description = [{
    Converts element types of a tensor (e.g., f32 → f16 for GM write).

    Example:
    ```mlir
    %f16 = tile.cast %f32 : !tile.tensor<[128,128],f32,#tile.UB>
               -> !tile.tensor<[128,128],f16,#tile.UB>
    ```
  }];

  let arguments = (ins
    TileIR_TensorType:$src
  );

  let results = (outs TileIR_TensorType:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Tensor Conversion Operations
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// tile.to_tensor — Convert a buffer to an immutable tensor view
// ---------------------------------------------------------------------------
def TileIR_ToTensorOp : TileIR_Op<"to_tensor", [Pure]> {
  let summary = "Convert a buffer to a ranked tensor value";
  let description = [{
    Materializes a buffer as a standard ranked tensor so it can feed Triton
    compute ops (tt.dot, arith.*, math.*, reduce, ...). The result is a builtin
    RankedTensorType (not a !tile.* type), which is what those ops require.

    Example:
    ```mlir
    %t = tile.to_tensor %buf : !tile.buf<[128,128],f16,l0a> -> tensor<128x128xf16>
    ```
  }];

  let arguments = (ins
    TileIR_BufType:$src
  );

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// ---------------------------------------------------------------------------
// tile.store_tensor — Store a standard tensor into a TileIR buffer
// ---------------------------------------------------------------------------
def TileIR_StoreTensorOp : TileIR_Op<"store_tensor", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "Store a tensor value into a buffer";
  let description = [{
    Writes a standard ranked tensor value back into a mutable TileIR buffer.
    This is the inverse boundary of tile.to_tensor, whose result intentionally
    stays a standard tensor so Triton/arith/math ops can consume it directly.

    Example:
    ```mlir
    tile.store_tensor %tensor → %buf : tensor<128x128xf32>, !tile.buf<[128,128],f32,#tile.UB>
    ```
  }];

  let arguments = (ins
    TT_Tensor:$src,
    TileIR_BufType:$dst
  );

  let assemblyFormat = "$src `->` $dst attr-dict `:` type($src) `,` type($dst)";
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// tile.set_flag — Set a cross-engine synchronization flag
// ---------------------------------------------------------------------------
def TileIR_SetFlagOp : TileIR_Op<"set_flag", []> {
  let summary = "Set a cross-engine synchronization flag";
  let description = [{
    Signals that a producer pipe has completed its work, enabling the
    consumer pipe to proceed. Maps to AscendC SetFlag<HardEvent::X>().

    Example:
    ```mlir
    tile.set_flag [producer=PIPE_MTE3, consumer=PIPE_MTE2, event=EVENT_ID2]
    ```
  }];

  let arguments = (ins
    TileIR_Pipe:$producer,
    TileIR_Pipe:$consumer,
    TileIR_EventID:$event
  );

  let assemblyFormat = [{
    `[` $producer `,` $consumer `,` $event `]` attr-dict
  }];
}

// ---------------------------------------------------------------------------
// tile.wait_flag — Wait for a cross-engine synchronization flag
// ---------------------------------------------------------------------------
def TileIR_WaitFlagOp : TileIR_Op<"wait_flag", []> {
  let summary = "Wait for a cross-engine synchronization flag";
  let description = [{
    Blocks until the specified producer pipe has set the event flag.
    Maps to AscendC WaitFlag<HardEvent::X>().

    Example:
    ```mlir
    tile.wait_flag [producer=PIPE_MTE2, consumer=PIPE_V, event=EVENT_ID3]
    ```
  }];

  let arguments = (ins
    TileIR_Pipe:$producer,
    TileIR_Pipe:$consumer,
    TileIR_EventID:$event
  );

  let assemblyFormat = [{
    `[` $producer `,` $consumer `,` $event `]` attr-dict
  }];
}

// ---------------------------------------------------------------------------
// tile.pipe_barrier — Intra-engine pipeline barrier
// ---------------------------------------------------------------------------
def TileIR_PipeBarrierOp : TileIR_Op<"pipe_barrier", []> {
  let summary = "Intra-engine pipeline barrier (RAW hazard)";
  let description = [{
    Inserts a barrier within a single pipe/engine to ensure read-after-write
    hazards are resolved. Maps to AscendC PipeBarrier<PIPE_X>().

    Example:
    ```mlir
    tile.pipe_barrier [PIPE_V]
    ```
  }];

  let arguments = (ins
    TileIR_Pipe:$pipe
  );

  let assemblyFormat = "`[` $pipe `]` attr-dict";
}

#endif // TILE_IR_OPS
