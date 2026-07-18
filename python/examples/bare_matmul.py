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

# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation

import torch
import triton
import triton.language as tl
import benchmark


@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # block row id
    pid_y = tl.program_id(1)  # block column id

    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])

    z = tl.dot(x, y)

    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


@benchmark.measure()
def bench_matmul(N, provider):
    device = 'cpu'
    dtype = torch.float32
    a = torch.randn((N, N), device=device, dtype=dtype)
    b = torch.randn((N, N), device=device, dtype=dtype)
    c = torch.empty((N, N), device=device, dtype=dtype)
    if provider == 'torch' or provider == 'test':
        c_ref = torch.matmul(a, b)
    if provider == 'triton' or provider == 'test':
        bare_matmul[(1,)](a, b, c, N, N, N, N)
        if provider == 'test':
            torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    benchmark.select_cpu_backend()
    for X in [2**i for i in range(7, 10, 1)]:
        for provider in ['test', 'torch', 'triton']:
            bench_matmul(X, provider)
