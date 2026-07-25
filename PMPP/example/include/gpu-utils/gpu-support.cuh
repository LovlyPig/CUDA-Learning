#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), file, line);
        throw std::runtime_error("GPUassert failed");
        exit(err);
    }
}

template<typename T>
__host__ __device__ void print_mem(const T &val) {
    const uint8_t *ptr = reinterpret_cast<const uint8_t*>(&val);
    for (size_t i = 0; i < sizeof(T); ++i) {
        printf("%02x ", ptr[i]);    
    }
    printf("\n");
}

template<typename F, typename... Args> __global__ void launch_kernel(F func, Args... args) {
    func(args...);
}

class GPUConfig {
public:
    int device_id;
    int num_sms;
    int max_threads_per_sm;
    int max_threads_per_block;
    int max_blocks_per_sm;
    size_t shared_mem_per_block;

    GPUConfig(int device_id = 0) : device_id(device_id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        num_sms = prop.multiProcessorCount;
        max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        max_threads_per_block = prop.maxThreadsPerBlock;
        max_blocks_per_sm = max_threads_per_sm / max_threads_per_block;
        shared_mem_per_block = prop.sharedMemPerBlock;
    }
};

class CudaTimer {
    cudaEvent_t start_event, stop_event;
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }
    void start() {
        cudaEventRecord(start_event, 0);
    }

    double stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        return static_cast<double>(milliseconds);
    }
};
