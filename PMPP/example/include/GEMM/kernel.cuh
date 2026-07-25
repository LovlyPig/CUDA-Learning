#include <vector>

#define TILE_WIDTH 32
// M*K x K*N -> M*N
__global__ void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gemm_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < ((K + TILE_WIDTH-1) / TILE_WIDTH); ++t) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_WIDTH + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 线程粗化需要在计算任务足够大时才有意义
// 按块粗化，如每个线程计算 2x2 子块
#define COARSENING_FACTOR 2
__global__ void gemm_tiled_optimized(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH+1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSENING_FACTOR + tx;

    float sum[COARSENING_FACTOR] = {0.0f};

    for (int t = 0; t < ((K + TILE_WIDTH-1) / TILE_WIDTH); ++t) {
        if (row < M && t * TILE_WIDTH + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        for (int c = 0; c < COARSENING_FACTOR; ++c) {
            int col = colStart + c * TILE_WIDTH;
            if (col < N && t * TILE_WIDTH + ty < K) {
                Bs[tx][ty] = B[(t * TILE_WIDTH + ty) * N + col];
            } else {
                Bs[tx][ty] = 0.0f;
            }
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                sum[c] += As[ty][k] * Bs[tx][k];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSENING_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        if (row < M && col < N) {
            C[row * N + col] = sum[c];
        }
    }
}

// 向量化
__global__ void gemm_tiled_vectorized(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int M, int N, int K) {
   __shared__ float As[TILE_WIDTH][TILE_WIDTH];
   __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

   int bx = blockIdx.x, by = blockIdx.y;
   int tx = threadIdx.x, ty = threadIdx.y;

   int row = by * TILE_WIDTH + ty;
   int col = bx * TILE_WIDTH + tx;

   float sum = 0.0f;

   int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
   for (int t = 0; t < num_tiles; ++t) {
        int A_global_col = t*TILE_WIDTH + 4*tx;
        if (row < M && A_global_col + 3 < K) {
            float4 a_vec = *reinterpret_cast<const float4*>(&A[row * K + A_global_col]);
            As[ty][4*tx] = a_vec.x;
            As[ty][4*tx + 1] = a_vec.y; 
            As[ty][4*tx + 2] = a_vec.z;
            As[ty][4*tx + 3] = a_vec.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (A_global_col + i < K && row < M) {
                    As[ty][4*tx + i] = A[row * K + A_global_col + i];
                } else {
                    As[ty][4*tx + i] = 0.0f;
        }

        int B_global_row = t*TILE_WIDTH + 4*ty;
        if (col < N && B_global_row + 3 < K) {
            float4 b_vec = *reinterpret_cast<const float4*>(&B[(B_global_row) * N + col]);
            // 转置存入共享内存
            Bs[4*ty][tx] = b_vec.x;
            Bs[4*ty + 1][tx] = b_vec.y;
            Bs[4*ty + 2][tx] = b_vec.z;
            Bs[4*ty + 3][tx] = b_vec.w;
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (B_global_row + i < K && col < N) {
                    Bs[4*ty + i][tx] = B[(B_global_row + i) * N + col];
                } else {
                    Bs[4*ty + i][tx] = 0.0f;    
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            float4 a = *reinterpret_cast<float4*>(&As[ty][k]);
            float4 b = *reinterpret_cast<float4*>(&Bs[tx][k]);
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        __syncthreads();
   }

   if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

