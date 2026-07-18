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
def addptr(in0, out0):
    for i in range(0, 10, 2):
        in1 = in0 + 1 + i
        in2 = in1 + 1

        out1 = out0 + 1 + i
        out2 = out1 + 1

        a1 = tl.load(in1)
        a2 = tl.load(in2)

        tl.store(out1, a1)
        tl.store(out2, a2)



def test(device):
    input = torch.arange(0, 11, device=device, dtype=torch.float32)
    output = torch.full((11,), 0, device=device, dtype=torch.float32)
    grid = lambda meta: (1,)

    print(output)
    addptr[grid](input, output)
    print(input)
    print(output)
    assert torch.equal(input, output)

    # TODO: need to check some conditions otherwise the code below does not make any difference for the test
    src = triton.compiler.ASTSource(
        fn=addptr,
        signature="*fp32,*fp32",
    )
    ret = triton.compile(
        src,
    )
    print(ret.asm["ttir"])
