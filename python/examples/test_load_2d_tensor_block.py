# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

import triton
import triton.language as tl


"""

|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|

Each instance loads BLOCK_SIZE_ROW * BLOCK_SIZE_COL
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_0,
    stride_1,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    input_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=[n_rows, n_cols],
        strides=[stride_0, stride_1],
        offsets=[pid0 * BLOCK_SIZE_ROW, pid1 * BLOCK_SIZE_COL],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
        order=[1, 0],
    )
    x = tl.load(input_ptr)
    x = (2 * x) + 1
    output_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=[n_rows, n_cols],
        strides=[stride_0, stride_1],
        offsets=[pid0 * BLOCK_SIZE_ROW, pid1 * BLOCK_SIZE_COL],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
        order=[1, 0],
    )
    tl.store(output_ptr, x)


def test(device):
    n_rows = 512
    n_cols = 256
    x = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    output = torch.full([n_rows, n_cols], -1, device=device, dtype=x.dtype)
    BLOCK_SIZE_ROW = 4
    BLOCK_SIZE_COL = 2

    grid = lambda meta: (n_rows // BLOCK_SIZE_ROW, n_cols // BLOCK_SIZE_COL)

    kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    expected = (2 * x) + 1

    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
