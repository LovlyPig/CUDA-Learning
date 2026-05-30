import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 标量版本
        triton.Config({'BLOCK_SIZE': 256, 'VEC_WIDTH': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_WIDTH': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_WIDTH': 1}, num_warps=16),
        # 向量化版本
        triton.Config({'BLOCK_SIZE': 256, 'VEC_WIDTH': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_WIDTH': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_WIDTH': 4}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_WIDTH': 4}, num_warps=16),
    ],
    key=['N'] # 根据输入大小N选择最优配置
)
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr, VEC_WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE * VEC_WIDTH
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * VEC_WIDTH
    
    if VEC_WIDTH == 4:
        # 将每个线程的偏移扩展为4个连续的地址
        vec_offsets = offsets[:, None] + tl.arange(0, VEC_WIDTH)[None, :]
        # 将二维偏移展平为一维
        flat_offsets = tl.ravel(vec_offsets)
        mask = flat_offsets < N

        a = tl.load(a_ptr + flat_offsets, mask=mask)
        b = tl.load(b_ptr + flat_offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + flat_offsets, c, mask=mask)
    else:
        mask = offsets < N
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = a + b
        tl.store(c_ptr + offsets, c, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous()

    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE'] * meta['VEC_WIDTH']),)
    add_kernel[grid](x, y, out, N)
    return out

if __name__ == "__main__":
    N = 1<<26
    a = torch.rand(N, device='cuda', dtype=torch.float32)
    b = torch.rand(N, device='cuda', dtype=torch.float32)

    c_triton = add(a, b)
    torch.cuda.synchronize()
    c_torch = a + b
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-3, atol=1e-5)
    print("Triton Vector Add is correct!")

    print("Benchmarking Triton Vector Add...")
    ms_triton = triton.testing.do_bench(lambda: add(a, b))
    ms_torch = triton.testing.do_bench(lambda: a + b)

    bytes_total = 3*N*a.element_size(); # 读取a和b，写入c，每个元素4字节
    bw_triton = (bytes_total / 1e9) / (ms_triton / 1e3)
    bw_torch = (bytes_total / 1e9) / (ms_torch / 1e3)

    print("Triton Vector Add: {:.3f} ms, {:.2f} GB/s".format(ms_triton, bw_triton))
    print("PyTorch Vector Add: {:.3f} ms, {:.2f} GB/s".format(ms_torch, bw_torch))

    print("\nBest config chosen by autotune:")
    best_config = add_kernel.best_config
    print(f"BLOCK_SIZE: {best_config.kwargs['BLOCK_SIZE']}, VEC_WIDTH: {best_config.kwargs['VEC_WIDTH']}, num_warps: {best_config.num_warps}")