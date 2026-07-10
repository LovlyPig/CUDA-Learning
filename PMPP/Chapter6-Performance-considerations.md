# Chapter 6 Performance Considerations

## 6.1 Memory coalescing

[内存访问合并](../CUDA%20Programming%20Guide/2.2.4.%20Memory-Performance.md)
[内存访问合并](../CUDA%20Programming%20Guide/2.2CUDA-SIMT-Kernels.md#2234Local-Memory)

[tiled-mm](Chapter5-Memory-architecture-and-data-locality.md#53-tiling-for-reduced-memory-traffic)
在第5节的分块矩阵乘法中，因为数组的存放是行主序的，所以对矩阵a的访问是连续的，而对矩阵b的访问是不连续，即矩阵b的访问是 uncoalesced access pattern。
所以如果对b矩阵进行转置（或列主序存储），就也能使其的访问是连续的。

```C++
__global__ void TiledMatrixMulColMajor(float *M, float *N, float *P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    for (int t = 0; t < numTiles; ++t) {
        Mds[ty][tx] = M[row*n + t*TILE_WIDTH + tx];
        // N存储为列主序
        Nds[ty][tx] = N[col*n + t*TILE_WIDTH + ty];
        // 另一种方式是转置 每个线程加载一行（连续）的数据，然后按列存储到 Nds
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
    }
}
```

## 6.2 Hiding memory latency

DRAM systems typically employ two more forms of parallel organization: banks and channels. At the highest level, a processor contains one or more channels. Each channel is a memory controller with a bus that connects a set of DRAM banks to the processor. In real systems a processor tpically has one to eight channels, and a large number of banks is connected to each channel.

The data transfer bandwidth of a bus is defined by its width and clock frequency. Modern *double data rate*(DDR) busses perform two data transfers per clock cycle: one at the rising edge and one at the falling edge of each clock cycle. For each channel, the number of banks that is connected to it is determined by the number of banks required to fully utilize the data transfer bandwidth of the bus.
实际系统中，数据访问延迟比数据传输时间要长得多，只有一个bank的组织方式显然无法有效利用channel总线的数据传输带宽。例如，假设DRAM存储阵列的访问延迟与数据传输时间的比例是20：1，则channel总线的最大利用率是1/21=4.8%；对于一个16GB/s的channel总线数据传输到处理器的速率不会大于0.76GB/s。

In general, if the ratio of the cell array access latency and data transfer time is R, we need to have at least R+1 banks if we hope to fully utilize the data transfer bandwidth of the channel bus. The number of banks connected to each channel bus needs to be larger than R for two reasons. One is that having more banks reduces the probability of multiple simultaneous accesses targeting the same bank, a phenomenon called *bank conflict*. Since each bank can serve only one access at a time, the cell array access latency can no longer be overlapped for these conflicting accesses. The second reason is that the size of each cell array is set to achieve reasonable latency and manufacturability. This limits the number of cells that each bank can provide. One may need many banks just to be able to support the memory size that is required.

## 6.3 Thread coarsening(线程粗化)

The advantage of parallelizing work across threads at the finest granularity is that it enhances transparent scalability. If the hardware has enough resources to perform all the work in parllel, then the application has exposed enough parallelism to fully utilize the hardware.

在细粒度线程上并行化工作的劣势发生于当并行化这个工作存在“代价”时。这个并行模式的代价可以有很多形式，如不同的线程块重复读取相同的数据，重复的冗余计算，同步开销等等。如果在硬件上线程确实是并行执行的，那么这些开销是合理的。但是如果最终是串行执行导致资源的不充分利用，那么这些开销就是非必要的。In this case, it is better for programmer to partially serialize the work and reduce the price that is paid for parallelism. This can be done by assigning each thread multiple units of work,  which is often referred to as *thread coarsening*.

例如，在分块矩阵乘法中，会访问同一块矩阵M与不同块的矩阵N，导致不同的线程块在其共享内存存储了相同的矩阵M块，但这是我们为了让两个不同块并行计算所必需的开销。但如果在这些线程块实际是串行执行的，那么这个开销将是不合理的。所以我们可以让一个线程块做多个块的计算，使M矩阵块重复利用。

```C++
#define TILE_WIDTH 32
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH+1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }

    for (int t = 0; t < width/TILE_WIDTH; ++t) {
        Mds[ty][tx] = M[row*width + t*TILE_WIDTH +tx];

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c*TILE_WIDTH;

            Nds[tx][ty] = N[(t*TILE_WIDTH + ty)*width + col]; // 内存合并，转置映射
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k]*Nds[tx][k]; // padding解决bank conflict
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c*TILE_WIDTH;
        P[row*width+col] = Pvalue[c];
    }
}
```

Thread coarsening is a powerful optimization that can result in substantial performance improvement for many applications. There are several pitfalls to avoid in applying thread coarsening. First, recall that thread coarsening is beneficial when there is a price paid for parallelization that can be reduced with coarsening, such as r**edundant loading of data, redundant work, synchronization overhead, or others.** Second, recall that exposing as much parallelism as possible to the hardware enables transparent scalability. When programmers coarsen threads, they reduce the amount of parallelism that is exposed to the hardware. If the coarsening factor is too high, resulting in some parallel execution resources bing unutilized. Third, avoid increasing resource consumption to such an extent that it hurts occupancy. Thread coarsening may require using more registers per thread or more shared memory per thread block. The performance penalty from reducing occupancy may be more detrimental than the performance benefit that thread coarsening may offer.

## 6.4 A checklist of optimizations

This checklist is not an exhaustive one, but it contains many of the universal optimizations that are common across different applications and that programmers should first consider.








