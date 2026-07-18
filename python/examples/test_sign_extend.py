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

from triton.backends.triton_shared.driver import CPUDriver

@triton.jit
def sign_extend(off, in0, out0, in0_size):
    offset = tl.load(off).to(tl.int64)
    offsets = offset + tl.arange(0, 4)
    a = tl.load(in0 + offsets, mask=offsets < in0_size, other=11)
    tl.store(out0 + tl.arange(0, 4), a)

def compile():
    src = triton.compiler.ASTSource(
        fn=sign_extend,
        signature="*i32,*fp32,*fp32,i32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])

def test_sign_extend(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 4
    offsets = torch.full((1, ), 1, device=device, dtype=torch.int32)
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1,)
    print(output)
    sign_extend[grid](offsets, input, output, SIZE)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([1, 2, 3, 11], device=device, dtype=torch.int32), output)
