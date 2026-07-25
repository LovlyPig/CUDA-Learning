#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <thread>
#include <random>

#include <omp.h>

static void random_matrix(std::vector<float> &matrix, int rows, int cols) {
    thread_local std::minstd_rand rng(std::random_device{}());
    constexpr float min_val = -1.0f;
    constexpr float range = 2.0f;

    matrix.resize(rows * cols);
    for (auto &val : matrix) {
        val = min_val + range * (static_cast<float>(rng()) / static_cast<float>(rng.max()));
    }
}

static void check(const std::vector<float> &a, const std::vector<float> &b, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float diff = std::abs(a[i * cols + j] - b[i * cols + j]);
            if (diff > 1e-3) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                          << a[i * cols + j] << " vs " << b[i * cols + j]
                          << ", diff = " << diff << std::endl;
                return;
            }
        }
    }
}

void gemm_cpu(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int M, int N, int K) {
    const int BLOCK = 64; // 
    #pragma omp parallel for collapse(2)
    for (int bi = 0; bi < M; bi += BLOCK) {
        for (int bj = 0; bj < N; bj += BLOCK) {
            
            int max_i = std::min(bi + BLOCK, M);
            int max_j = std::min(bj + BLOCK, N);
            
            for (int i = bi; i < max_i; ++i) {
                for (int j = bj; j < max_j; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }
}


#endif