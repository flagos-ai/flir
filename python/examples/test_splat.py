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


@triton.jit
def splat(
    f32_val,
    f32_out,
    stride_row,
    stride_col,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    x = tl.full((2, BLOCK_SIZE_COL), f32_val, dtype=tl.float32)
    offs_row = 2 * pid0 + tl.arange(0, 2)
    offs_col = tl.arange(0, BLOCK_SIZE_COL)
    a_ptrs = f32_out + (offs_row[:, None] * stride_row + offs_col[None, :] * stride_col)
    tl.store(a_ptrs, x)


def test(device):
    n_rows = 256
    n_cols = 512
    fill_value = 123.456
    expected_result = torch.full((n_rows, n_cols), fill_value, dtype=torch.float32)
    output = torch.empty([n_rows, n_cols], device=device, dtype=expected_result.dtype)
    grid = lambda meta: (n_rows // 2,)

    splat[grid](
        fill_value,
        output,
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_ROW=n_rows,
        BLOCK_SIZE_COL=n_cols,
    )

    torch.testing.assert_close(output, expected_result, rtol=0.001, atol=1e-5)
