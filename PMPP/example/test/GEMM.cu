#include "gpu-utils/gpu-support.cuh"
#include "GEMM/matrix.h"
#include "GEMM/kernel.cuh"
#include "utils.h"

#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    GPUConfig config;

    int M = 1024;
    int N = 1024;
    int K = 1024;

    if (argc >= 2) {
        M = N = K = std::atoi(argv[1]);
    }

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_cpu(M * N, 0.0f);
    std::vector<float> C(M * N, 0.0f);

    random_matrix(A, M, K);
    random_matrix(B, K, N);

    float *d_A, *d_B; 
    float *d_C1, *d_C2, *d_C3;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C1, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C2, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C3, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    CudaTimer timer1, timer2, timer3;
    double elapsed_time_naive, elapsed_time_tiled, elapsed_time_optimized;

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize(ceil_div(N, TILE_WIDTH), ceil_div(M, TILE_WIDTH));
    timer1.start();
    gemm_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C1, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    elapsed_time_naive =  timer1.stop();

    timer2.start();
    gemm_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C2, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    elapsed_time_tiled = timer2.stop();

    dim3 gridSizeOptimized(ceil_div(N, TILE_WIDTH * COARSENING_FACTOR), ceil_div(M, TILE_WIDTH));
    timer3.start();
    gemm_tiled_optimized<<<gridSizeOptimized, blockSize>>>(d_A, d_B, d_C3, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    elapsed_time_optimized = timer3.stop();

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    double elapsed_time_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    gemm_cpu(A, B, C_cpu, M, N, K);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    elapsed_time_cpu = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    std::cout << "CPU time: " << elapsed_time_cpu << " ms" << std::endl;

    CUDA_CHECK(cudaMemcpy(C.data(), d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Checking results for naive kernel...\n");
    check(C_cpu, C, M, N);

    CUDA_CHECK(cudaMemcpy(C.data(), d_C2, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Checking results for tiled kernel...\n");
    check(C_cpu, C, M, N);

    CUDA_CHECK(cudaMemcpy(C.data(), d_C3, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Checking results for optimized tiled kernel...\n");
    check(C_cpu, C, M, N);

    std::cout << "Naive kernel time: " << elapsed_time_naive << " ms" << std::endl;
    std::cout << "Tiled kernel time: " << elapsed_time_tiled << " ms" << std::endl;
    std::cout << "Optimized tiled kernel time: "<< elapsed_time_optimized << " ms" << std::endl;  

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);

    return 0;
}