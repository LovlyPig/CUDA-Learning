# Chapter 6 Performance Considerations

# 6.1 Memory coalescing

[内存访问合并](../CUDA%20Programming%20Guide/2.2.4.%20Memory-Performance.md)
[内存访问合并](../CUDA%20Programming%20Guide/2.2CUDA-SIMT-Kernels.md#2234Local-Memory)

[tiled-mm](Chapter5-Memory-architecture-and-data-locality.md#53-tiling-for-reduced-memory-traffic)
在第5节的分块矩阵乘法中，因为数组的存放是行主序的，所以对矩阵a的访问是连续的，而对矩阵b的访问是不连续，即矩阵b的访问是 uncoalesced access pattern。
所以如果对b矩阵进行转置，就也能使其的访问是连续的。

# 6.2 Hiding memory latency






