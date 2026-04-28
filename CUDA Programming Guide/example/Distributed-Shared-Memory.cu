/*A standard way of computing histograms is to perform the 
computation in the shared memory of each thread block and 
then perform global memory atomics. A limitation of this 
approach is the shared memory capacity. Once the histogram 
bins no longer fit in the shared memory, a user needs to 
directly compute histograms and hence the atomics in the 
global memory. With distributed shared memory, CUDA provides
an intermediate step, where depending on the histogram bins 
size, the histogram can be computed in shared memory, 
distributed shared memory or global memory directly. */

#include <cuda_runtime.h>
#include <iostream>
#include <cooperative_groups.h>

// distributed shared memory histogram
__global__ void clusterHist_kernel(int* bins, const int nbins, const int bins_per_block,
                                    const int*__restrict__ input, size_t size) {

    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    cg::cluster_group cluster = cg::this_cluster();
    uint32_t clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        smem[i] = 0;
    }
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];

        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;
        
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;
        int* dst_smem  = cluster.map_shared_rank(smem, dst_block_rank);

        atomicAdd(&dst_smem[dst_offset], 1);
    }

    cluster.sync();

    int* lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        atomicAdd(&lbins[i], smem[i]);
}

/*The above kernel can be launched at runtime with a cluster
 size depending on the amount of distributed shared memory 
 required. If the histogram is small enough to fit in shared 
 memory of just one block, the user can launch the kernel 
 with cluster size 1.*/

int main() {

    int array_size = 1 << 20;
    int threads_per_block = 256;
    int nbins = 1024;

    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    int cluster_size = 2;
    int nbins_per_block = nbins / cluster_size;

    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    CUDA_CHECK(::cudaFuncSetAttribute((void*)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    cudaLaunchAttributes attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cluster_size;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attr;

    cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);

}