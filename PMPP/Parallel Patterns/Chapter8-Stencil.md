# Stencil

## Background

The first step in using computers to numerically evaluate and solve functions, models, variables, and equations is to convert them into a discrete representation. 

In mathematics a stencil is a geometric pattern of weights applied at each point of a structured grid. The pattern specifies how the values of the grid point of interest can be derived from the values at neighboring points using a numerical approximation routine. 

For example, assume that we have `f(x)` discretized into a 1D grid array F and we would like to calculate the discretized derivative of `f(x), f'(x)`. We can use the classic finite difference approximation for the first derivate:
$$
    f'(x) = \frac{f(x+h)-f(x-h)}{2h}+O(h^2)
$$

Since the grid spacing is h, the current estimated f(x-h), f(x) and f(x+h) values are in F[i-1], F[i] and F[i+1], respectively, where x=i*h. Therefore:
$$
    FD[i] = \frac{F[i+1]-F[i-1]}{2h} = -1/2h*F[i-1]+1/2h*F[i+1]
$$
That is, the calculation of the estimated function derivative value at grid point involves the current estimated function values at grid points [i-1, i, i+1] with coefficients [-1/2h, 0, 1/2h], which defines a 1D three-point stencil.

If the partial differential equation involves exclusively the partial derivatives by only one of the variables, for example, $\frac{\partial f(x,y)}{\partial x}, \frac{\partial f(x,y)}{\partial y}$, but not $\frac{f(x,y)}{\partial x \partial y}$, we can use 2D stencils whose selected grid points are all along the x axis and y axis.

## Basic Parallel stencil

For simplicity we assume that there is no dependence between output grid points when generating the output grid point values within a stencil sweep. We further assume that the grid point values on the boundaries store boumdary conditions and will not change from input ti output.

```C++
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

    if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
        out[i*N*N+j*N+k] = c0*in[i*N*N+j*N+k] + c1*in[i*N*N+j*N+k-1] + c2*in[i*N*N+j*N+k+1] + c3*in[i*N*N+(j-1)*N+k] + c4*in[i*N*N+(j+1)*N+k] + c5*in[(i-1)*N*N+j*N+k] + c6*in[(i+1)*N*N+j*N+k];
    }
}
```
Each thread performs 13 floating-point operations and load seven input values. Therefore the floating-point to global memory access ratio for this kernel is $13/(7*4) = 0.46 OP/B$.

## Shared memory tiling for stencil sweep

因为 stencil 相比 convolution 的计算强度更低，如2D 3x3 convolution 是4.5OP/B，而2D five-point stencil 只有 2.5OP/B；同样，3D 19-point stencil 是9.5OP/B, 3D 7x7x7 convolution 则为 171.5 OP/B.
所以使用共享内存的收益相对要更低，this small but significant difference motivates the use of thread coarsening and register tiling in the third dimension.

```C++
// blockDim = IN_TILE_DIM
// OUT_TILE_DIM = IN_TILE_DIM - 1
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int i = blockIdx.z*OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM];
    if (i>=0 && i<N && j>=0 && j<N && k>=0 && k<N)
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i*N*N+j*N+k];
    __syncthreads();

    if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
        if (threadIdx.z>=1 && threadIdx.z<OUT_TILE_DIM && threadIdx.y>=1 && threadIdx.y<OUT_TILE_DIM && threadIdx.x>=1 && threadIdx.x<OUT_TILE_DIM) {
            out[i*N*N+j*N+k] = c0*in_s[threadIdx.z][threadIdx.y][threadIdx.x] + c1*in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] + c2*in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] + c3*in_s[threadIdx.z][threadIdx.y-1][threadId.x] + c4*in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] + c5*in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] + c6*in_s[threadIdx.z+1][threadIdx.y][threadIdx.x];
        }
    }
}
```

We assume that each input tile is a cube with T grid points in each dimension and that each output tile has T-2 grid points in each dimension. Therefore each block has $(T-2)^3$ active threads calculating output grid point values, and each active thread performs 13 floating-point operations(3D 7-point). Therefore:
$$
    \frac{13*(T-2)^3}{4*T^3} = 13/4(1-2/T)^3 OP/B
$$
That is, the larger the Tvalue, the more the input grid point values are reused. The upper bound on the ratio as T increases asymptotically is $13/4=3.25 OP/B$.

1. 每个线程块最大1024个线程，以及共享内存大小限制了T的大小，如8 * 8 * 8的block只有1.37 OP/B；
2. 对于半径为1的filter，32x32 tile的convolution对应1024输入元素和30x30=900输出元素，halo elements 约12%左右；而8x8x8 3D是512输入元素，输出元素约6x6x6=216，halo elements 占到了58%
3. 8x8x8 tile，每个线程束的线程会访问4个不同的行，这种访问模式很难合并，无法充分利用DRAM带宽。

## Thread Coarsening

In this case, the price that is paid for parallelism is the low data reuse due to the loading of halo elements by each blocks.

```C++
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z*OUT_TILE_DIM;
    int j = blockIdx.y*OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart-1>=0 && iStart-1<N && j>=0 && j<N && k>=0 && k<N)
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart-1)*N*N + j*N + k];
    if (iStart>=0 && iStart<N && j>=0 && j<N && k>=0 && k<N)
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart*N*N + j*N + k];
    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i+1>=0 && i+1<N && j>=0 && j<N && k>=0 && k<N) 
            inNext_s[threadIdx.y][threadIdx.x] = in[(i+1)*N*N + j*N + k];
        __syncthreads();

        if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
            if (threadIdx.y>=1 && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x<IN_TILE_DIM-1) {
                out[i*N*N + j*N + k] = c0*inCurr_s[threadIdx.y][threadIdx.x] + c1*inCurr_s[threadIdx.y][threadIdx.x-1] + c2*inCurr_s[threadIdx.y][threadIdx.x+1] + c3*inCurr_s[threadIdx.y-1][threadIdx.x] + c4*inCurr_s[threadIdx.y+1][threadIdx.x] + c5*inPrev_s[threadIdx.y][threadIdx.x] + c6*inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}
```

This shows a kernel with thread coarsening in the z diretion for a 3D seven-point stencil sweep. The advantages of the thread coatsening kernel are that it increases the tile size without increasing the number of the threads and that it does not require all planes of the input tile to be present in the shared memory. **The thread block size is now only $T^2$ instead of $T^3$**, so we can use a much larger T value. Moreover, at any point in time, only three layers of the input tile need to be in the shared memory. The shared memory capacity requirement is now $3T^2$ elements instead of $T^3$ elements.

## Register Tiling

Here, we present an optimization that can be especially effective for stencil patterns that involve only neighbors along the x,y and z diretions of the center point.

Each inPrev_s and inNext_s element is used by only one thread in the calculation of the output tile grid point with the same x-y indices. Only the inCurr_s elements are accessed by multiple threads and truly need to be in the shared memory. The z neighbors in inPrev_s and inNext_s can instead stay in the registers of the single user thread. We take advantage of this property with the register tiling kernel.

```C++
__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM；
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inPrev, inCurr, inNext;

    if (iStart-1>=0 && iStart-1<N && j>=0 && j<N && k>=0 && k<N)
        inPrev = in[(iStart-1)*N*N + j*N + k];
    if (iStart>=0 && iStart<N && j>=0 && j<N && k>=0 && k<N) {
        inCurr = in[iStart*N*N + j*N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }

    for (int i = iStart; i < iStart+OUT_TILE_DIM; ++i) {
        if (i+1>=0 && i+1<N && j>=0 && j<N && k>=0 && k<N) 
            inNext = in[(i+1)*N*N + j*N + k];
        __syncthreads();

        if (i>=1 && i<N-1 && j>=1 && j<N-1 && k>=1 && k<N-1) {
            if (threadIdx.y>=1 && threadIdx.y<IN_TILE_DIM-1 && threadIdx.x>=1 && threadIdx.x，IN_TILE_DIM-1) {
                out[i*N*N + j*N + k] = co*inCurr + c1*inCurr_s[threadIdx.y][threadIdx.x+1] + c2*inCurr_s[threadIdx.y][threadIdx.x-1] + c3*inCurr_s[threadIdx.y+1][threadIdx.x] + c4*inCurr_s[threadIdx.y-1][threadIdx.x] + c5*inPrev + c6*inNext;
            }
        }

        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}
```

This scenario a common tradeoff that often needs to be made between shared memory and register usage.



