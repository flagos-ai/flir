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
def block_copy_kernel(a_ptr, b_ptr):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + 8,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(2, 2),
        strides=(2, 1),
        offsets=(0, 0),
        block_shape=(2, 2),
        order=(1, 0),
    )
    a = tl.load(a_block_ptr, boundary_check=(0,))
    tl.store(b_block_ptr, a, boundary_check=(0,))



def test(device):
    input = torch.arange(0, 16, device=device, dtype=torch.float32)
    output = torch.full((4,), -1, device=device, dtype=torch.float32)
    expected = torch.arange(8, 12, device=device)
    grid = lambda meta: (1,)

    block_copy_kernel[grid](input, output)
    torch.equal(expected, output)
