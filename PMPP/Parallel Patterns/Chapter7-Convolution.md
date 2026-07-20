# Convolution

**An introduction to constant memory and caching**

## background

1D convolution, taking an input data array of n elements $[x_0,x_1,\dots,x_{n-1}]$ and afilter attay of 2r+1 elements $[f_0,f_1,\dots,f_{2r}]$ and returns an output data array y:
$$
    y_i = \sum_{-r}^{r} y_{j+r} \times x_{i+j}    
$$

2D convolution,  the filter is $2r_x+1$ in the x dimension and $2r_y+1$ in the y dimension, the calcution of each P element can be expressed as follows:
$$
    P_{y,x} = \sum_{j=-ry}^{r_y}\sum_{k=-r_x}^{r_x}f_{y+j,x+k}\times N_{y,x}    
$$

3D convolution is same as 1D and 2D.

## basic algorithm

```C++
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2*r+1; ++fRow) {
        for (int fCol = 0; fCol < 2*r+1; ++fCol) {
            inRow = outRow-r+fRow;
            inCol = outCol-r+fCol;
            if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width)
                Pvalue += F[fRow][fCol] * N[inRow*width+inCol];
        }
    }

    P[outRow][outCol] = Pvalue;
}
```

Question:
1. 显然对于靠近边界的矩阵元素计算时，会导致严重的control divergence，
2. 计算强度低，对于2次存取操作只有一次加法一次乘法，即2 2 OP/8B = 0.25 OP/B

## Constant memory and Caching

First, the size of F is typically small; the radius of most convolution filters is 7 or smaller (even in 3D convolution, $7^3 * 4B = 1372B$ 仅仅1.34MB). Second, the contents of F do not change throught the execution of the convolution kernel. Third, all threads access the F elements in the same order, starting from F[0][0] and moving by one element at a time through the iterations of the doubly nested for-loop.

```C++
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1]
//.......

cudaMemcpyToSymbol(F, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float))
```

Kernel functions access constant memory variables as global variables. Therefore their pointers do not need to be passed to the kernel as arguments. Constant memory variables are also located in DRAM. However, because the CUDA runtime knows that constant memory variabes are not modified during kernel execution, it directs the hardwares to aggressively cache the constant memory variables during the kernel execution.

With the use of constant memory and caching, we have effectively doubled the ratio of floating-point arthmetic to memory access to around 0.5 OP/B.

## Tiled convolution with halo cells

We also can adress the memory bandwidth bottleneck of convolution with a tiled convolution algorithm.

```C++
// thread blocks matches the input tiles
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
// 将output tile 映射到input tile中间，input array和output array 实际上是一样的大小，所以只有实际范围是要计算的
// 如col 和 row 超出边界部分就是input tile 的 ghost cell
// tileCol和tileRow 同样指示的是input tile 与实际数组的相对位置
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width+col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (col>=0 && col<width && row>=0 && row<height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0 && tileRow<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; ++fRow) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; ++fCol) {
                    Pvalue += F_c[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }

}

// thread blocks matches the output tiles
#define OUT_TILE_DIM 32
#define IN_TILE_DIM ((OUT_TILE_DIM) + 2*(FILTER_RADIUS))

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    float *N_ss = reinterpret_cast<float*>(&N_s);
    N_ss += tid*2*FILTER_RADIUS;
    
    int start_row = threadIdx.y - FILTER_RADIUS + tid*2*FILTER_RADIUS/IN_TILE_DIM;
    int start_col = threadIdx.x - FILTER_RADIUS + tid*2*FILTER_RADIUS%IN_TILE_DIM;
    for (int r = 0; r < 2*FILTER_RADIUS; ++r) {
        if (start_row>=0 && start_row<height && start_col+r>=0 && start_col+r <width)
            N_ss[r] = N[start_row][start_col+r];
        else
            N_ss[r] = 0.0f;
    }
    __syncthreads();

    float Pvalue = 0.0f;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;
    for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; ++fRow) {
        for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; ++fCol) {
            if (tileRow+fRow<0 || tileRow+fRow>=height || tileCol+fCol<0 || tileCol+fCol>width)
                continue;
            Pvalue += F_c[fRow][fCol] * N_s[tileRow+fRow][tileCol+fCol];
        }
    }
    P[row*width+col] = Pvalue;

}

```

We now calculate the arithmetic-to-global memory access ratio for the tiled kernel(thread blocks matches the input tiles). Every thread that is assigned to an output tile element performs one multiplication and one addition for every element of the filter.
Therefore the threads in a internal block collectively perform $OUT\_TILE\_DIM^2 * (2*FILTER\_RADIUS+1)^2 * 2$ arithmetic operations(we ignore the effect of ghost cells, because for large input arrays, the effect of ghost cells for small mask sizes will be insignificant), and $IN\_TILE\_DIM^2 * 4B = (OUT\_TILE\_DIM + 2*FILTER\_RADIUS)^2 * 4B$ are loaded by each internal block. Therefore the arithmetic-to-global memory access ratio for the tiled kernel is
$$
    \frac{OUT\_TILE\_DIM^2 * (2 * FILTER\_RADIUS+1)^2 * 2}{(OUT\_TILE\_DIM + 2*FILTER\_RADIUS)^2 * 4}    
$$
For example with a 5x5 filter and 32x32 input tiles(28x28 output tiles), the ratio is 9.57 OP/B.

## Tiled convolution using caches for halo cells

Recall that the halo cells of an input tile of a block are also the internal elements of neighboring tiles. As a result, the memory accesses to theses halo cells may be naturally served from L2 cache without causing additional DRAM traffic. **That is, we can leave the accesses to these halo cells in the original N elements rather than loading them into the N_s.**

```C++
#define TILE_DIM 32
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N. float *P, int width, int height) {
    int col = blockIdx.x*TILE_DIM + threadIdx.x;
    int row = blockIdx.y*TILE_DIM + threadIdx.y;

    __shared__ N_s[TILE_DIM][TILE_DIM];
    if (row<height && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (col<width && row<height) {
        float Pvalue = 0.0f;

        for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; ++fRow) {
            for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; ++fCol) {
                int a = threadIdx.x - FILTER_RADIUS + fCol;
                int b = threadIdx.y - FILTER_RADIUS + fRow;
                int c = row - FILTER_RADIUS + fRow;
                int d = col - FILTER_RADIUS + fCol;

                if (a>=0 && a <width && b>=0 && b<height) {
                    Pvalue += F_c[fRow][fCol] * N_s[b][a];
                } else {
                    if (c>=0 && c<height && d>=0 && d<width) {
                        Pvalue += F_c[fRow][fCol] * N[c*width + d];
                    }
                }
            }
        }
        P[row*width + col] = Pvalue;
    }

}
```


