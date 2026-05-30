import torch

N = 1 << 26
a = torch.rand(N, device='cuda', dtype=torch.float32)
b = torch.rand(N, device='cuda', dtype=torch.float32)

_ = a+b

def benchmark_add(func, *args, name="ADD", n_warmup=5, n_reapt=20):

    for _ in range(n_warmup):
        func(*args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_reapt):
        func(*args)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / n_reapt
    return elapsed_ms

ms = benchmark_add(lambda x, y: x+y, a, b, name="PyTorch add")

bytes_total = 3*N*4;
bw = (bytes_total / 1e9) / (ms / 1e3)

print("PyTorch Vector Add: {:.3f} ms, {:.2f} GB/s".format(ms, bw))


