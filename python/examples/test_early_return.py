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
def early_return(in0, out0):
    pid = tl.program_id(0)
    id = tl.load(in0 + pid)
    if id == -1:
        return
    offs = 1 + tl.arange(0, 4)
    out_offs = tl.arange(0, 4)
    tl.store(out0 + out_offs, offs)

def compile(device):
    src = triton.compiler.ASTSource(
        fn=early_return,
        signature="*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])


def test_return_case(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 8
    input = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1,)
    print(output)
    early_return[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([ -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32), output)

def test_normal_case(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 8
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE,), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1,)
    print(output)
    early_return[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([ 1,  2,  3,  4, -1, -1, -1, -1], dtype=torch.int32), output)
