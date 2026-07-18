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

Each instance loads the entire column
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    input_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=[n_rows, n_cols],
        strides=[BLOCK_SIZE_COL, 1],
        offsets=[0, pid0],
        block_shape=[BLOCK_SIZE_ROW, 1],
        order=[1, 0],
    )
    x = tl.load(input_ptr)
    output_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=[n_rows, n_cols],
        strides=[BLOCK_SIZE_COL, 1],
        offsets=[0, pid0],
        block_shape=[BLOCK_SIZE_ROW, 1],
        order=[1, 0],
    )
    tl.store(output_ptr, x)


def test(device):
    n_rows = 4
    n_cols = 2
    x = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    output = torch.full([n_rows, n_cols], -1, device=device, dtype=x.dtype)
    BLOCK_SIZE_ROW = n_rows
    BLOCK_SIZE_COL = n_cols

    grid = lambda meta: (n_cols,)

    kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    torch.testing.assert_close(output, x, rtol=0.001, atol=1e-5)
