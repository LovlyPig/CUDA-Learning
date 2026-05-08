# Chapter 4 Compute architecture and scheduling

## 4.6 Warp scheduling and latency tolerance

当一个线程块（block）被调度到SM上执行时，该块就成为该 SM 上的一个活跃块（active block）。一个 SM 上能够同时容纳的活跃块数量，主要受限于其 L1 cache/共享内存的大小。在块被分配的同时，其内部所有线程束（warp）所需的硬件资源（包括寄存器和共享内存）就会被一次性预先分配并锁定。

一个块内的线程会被划分为若干个 warp，每个 warp 中的线程由 SM 内的执行单元以单指令多线程（SIMT）方式执行相同的指令。当发生分支发散（divergence）时，未进入当前执行路径的线程会被屏蔽（mask），暂时不参与执行。

SM 内部维护一个就绪 warp 列表，里面存放当前可以执行指令的 warp。在每个时钟周期内，warp 调度器不需要执行保存或恢复上下文的操作，只需要从这个列表里选出一个 warp，然后发射它的下一条指令。这个过程是零开销的——因为所有 warp 需要的资源（如寄存器文件）在块分配时就已经固定下来，切换warp本质上只是改变一个“基址指针”，可以在发射指令的同时完成。

这种“零开销切换”正是 GPU 实现延迟隐藏（latency hiding）的核心硬件基础。当一个 warp 因为长延迟操作（例如访问全局内存）而进入等待状态时，硬件可以立即切换到其他就绪的 warp 继续执行。只要一个 SM 上的活跃 warp 数量足够多，SM 上的执行单元就能始终保持忙碌，长延迟操作的影响就被有效“掩盖”了。

PS：在GPU中SM的理论最大并发能力与实际驻留的warp数并不相等。Ampere架构，一个SM最多可有 64 个并发线程束，但实际能“装下”的驻留线程束数可能只有 48。

## 4.7 Resource partitioning and occupancy

从4.6节我们知道如果能容纳足够多的活跃线程束，就可能掩盖更多的长延迟操作。但是实际上一个SM上调度的warp数并不一定能达到SM所支持的最大数量，这个比率通常称为occupancy。

SM的运行资源包括寄存器，共享内存，thread block slots和 thread slots，这些资源被动态分配给线程以满足其运行要求。
例如，Ampere A100 GPU最大支持 32 blocks per SM，64 warps per SM（2048 threads），1024 threads per block。这些资源的动态分配决定了SM上可以同时存在多个block（few threads per block）或少量block（more threads per block）。
比如我假设32 threads per block，从最大 warps来看SM上最多可以调度64个 active block，但是这个架构实际上最大支持32 blocks per SM，所以通过减小每个线程块的线程数来获得更多的warp也是有极限的。而且在这种配置下，实际上就只有32*32=1024 thread slots被利用，occupancy只有1024/2048=50%。所以为了最大利用thread slots和实现最大occupancy，至少应该需要2048/32 = 64 threads per block。

同时每个线程所需的寄存器数量和每个block的共享内存数量也限制着active block的数量。比如A100 支持最大 65536 registers per SM，如果最大利用则每个线程只能使用65536/2048 = 32 registers，如果需要64 registers per thread，那么SM上最大的线程数为1024，则SM 的occupancy 为50%（编译器会通过将变量溢出到全局内存来减少每个线程所使用的寄存器，从而提高可以调度的线程数来提高occupancy，但这样很可能会因为全局内存访问而降低性能）。

可以通过使用`cudaOccupancyMaxPotentialBlockSize`API获得最大occupany的最优block size和 grid size。但是API的目标是最大线程束级占用，并不一定是最优，其他性能影响因素如内存带宽、指令延迟、warp 分歧、bank conflict 等同样需要关注。(使用Nsight Compute分析)
如：
场景1：计算密集型kernel
- 每线程用大量寄存器做复杂计算
- 占用率可能只有30%
- 但计算单元利用率高，性能好

场景2：内存密集型kernel
- 每线程操作简单，主要在读写内存
- 需要高占用率来隐藏延迟
- 50%占用率可能不够

从 4.6 节可知，SM 上能够并发执行的活跃线程束（warp）越多，就越能有效掩盖长延迟操作。然而在实际中，SM 上实际调度的 warp 数量往往达不到硬件支持的最大值，这个实际比例称为占用率（occupancy）。

SM 的运行资源包括寄存器、共享内存、线程块槽位（block slots）和线程槽位（thread slots）。这些资源会被动态分配给线程，以满足其执行需求。

以 Ampere A100 GPU 为例，其每个 SM 的最大资源限制为：

最多 32 个活跃线程块

最多 64 个活跃线程束（即 2048 个线程）

每个线程块最多 1024 个线程

这些资源限制共同决定了 SM 上可以同时存在多个小线程块（每块线程数少）或少量大线程块（每块线程数多）。例如，假设每个线程块只包含 32 个线程，那么从线程束上限（64 warps）来看，理论上可以调度 64 个活跃块。但硬件规定最多只能有 32 个块，因此实际最多只有 32 个块，每个块 1 个 warp，总共 32 个 warp（1024 个线程），占用率仅为 1024/2048 = 50%。为了充分利用线程槽位并达到 100% 占用率，每个线程块至少需要包含 2048/32 = 64 个线程（即 2 个 warp）。

此外，每个线程所需的寄存器数量以及每个块所需的共享内存大小，也会限制活跃块的数量。例如，A100 每个 SM 最多支持 65536 个寄存器。若要达到最大占用率，每个线程平均只能使用 65536/2048 = 32 个寄存器。如果每个线程需要 64 个寄存器，那么 SM 上最多只能容纳 1024 个线程（因为 64 × 1024 = 65536），此时占用率为 50%。编译器有时会通过将寄存器变量溢出（spill）到全局内存来减少每线程寄存器使用量，从而允许更多线程并发，提高占用率；但这往往会因访问全局内存而降低性能。

我们可以使用 cudaOccupancyMaxPotentialBlockSize API 来获得达到最高占用率的最优线程块大小和网格大小。但需要注意，该 API 的目标是最大化线程束级占用率，这并不总是等同于点性能最优。其他性能影响因素——如内存带宽、指令延迟、线程束发散（warp divergence）、共享内存 bank conflict 等——同样需要关注。实际分析中建议使用 Nsight Compute 等工具。

以下两个场景可以说明高占用率不等于高性能：

场景 1：计算密集型 kernel
每个线程使用大量寄存器进行复杂计算，占用率可能只有 30%。但由于计算单元持续处于忙碌状态，整体性能依然很好。

场景 2：内存密集型 kernel
每个线程操作简单，主要进行全局内存读写。此时需要较高的占用率来隐藏访存延迟，若占用率仅 50%，可能不足以掩盖延迟，导致计算单元空闲。

```C++
cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int       *minGridSize,    // 输出: 达到最大占用所需的最小 grid size
    int       *blockSize,      // 输出: 计算出的最优 block size
    const void *func,          // 输入: 你的 kernel 函数名
    size_t    dynamicSMemSize, // 输入: kernel 使用的动态共享内存 (extern __shared__) 字节数
    int       blockSizeLimit   // 输入: 对 blockSize 的上限约束, 0 表示无限制
);


__global__ void MyKernel(float* data, int N) {
    // 
}

int main() {
    // 

    int minGridSize, blockSize;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                        MyKernel, 0, 0);

    int totalThreads = N; // 总计算量
    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    // 确保 gridSize 不小于 minGridSize 以最大化占用
    if (gridSize < minGridSize) {
        gridSize = minGridSize;
    }

    MyKernel<<<gridSize, blockSize>>>(d_data, N);
    // ...


    // 计算给定配置的占用率
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
                                                myKernel, blockSize, 0);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    float occupancy = (float)(numBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;
    printf("占用率: %.2f%%\n", occupancy * 100);
}
```
