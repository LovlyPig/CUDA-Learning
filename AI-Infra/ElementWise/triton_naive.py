import torch
import triton
import triton.language as tl
import time

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'] # 根据输入大小N选择最优配置
)
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous()

    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, N)
    return out

def main():
    N = 1<<26
    a = torch.rand(N, device='cuda', dtype=torch.float32)
    b = torch.rand(N, device='cuda', dtype=torch.float32)

    c = add(a, b)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    c = add(a, b)
    torch.cuda.synchronize()
    end = time.time()

    bw = (3*N*4) / (end - start) / 1e9
    print(f"Time: {end - start:.3f} seconds, Bandwidth: {bw:.3f} GB/s.")

if __name__ == "__main__":    main()