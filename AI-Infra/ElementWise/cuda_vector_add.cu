#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_naive(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// 跨步循环访问
__global__ void add_stride(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// 向量化访问，处理4个元素为一组
__global__ void add_vectorized(const float* a, const float* b, float* c, int N) {
    const float4* a_vec = reinterpret_cast<const float4*>(a);
    const float4* b_vec = reinterpret_cast<const float4*>(b);
    float4* c_vec = reinterpret_cast<float4*>(c);

    int N4 = N/4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N / 4; i += stride) {
        float4 a_val = a_vec[i];
        float4 b_val = b_vec[i];
        c_vec[i] = make_float4(a_val.x + b_val.x, a_val.y + b_val.y, a_val.z + b_val.z, a_val.w + b_val.w);
    }

    // 处理剩余元素
    int remain = N4*4;
    for (int i = idx+remain; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

struct Data {
    float *ha, *hb, *hc;
    float *da, *db, *dc;
    int N;
};

Data init(int N) {
    Data data;
    data.N = N;
    size_t size = N * sizeof(float);
    data.ha = (float*)malloc(size);
    data.hb = (float*)malloc(size);
    data.hc = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        data.ha[i] = static_cast<float>(i);
        data.hb[i] = static_cast<float>(i * 2);
    }

    cudaMalloc(&data.da, size);
    cudaMalloc(&data.db, size);
    cudaMalloc(&data.dc, size);

    cudaMemcpy(data.da, data.ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(data.db, data.hb, size, cudaMemcpyHostToDevice);

    return data;
}

void cleanup(Data& data) {
    free(data.ha);
    free(data.hb);
    free(data.hc);
    cudaFree(data.da);
    cudaFree(data.db);
    cudaFree(data.dc);
}

float launch(void (*kernel)(const float*, const float*, float*, int), Data& data, int threads, int blocks, int iter = 100) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        kernel<<<blocks, threads>>>(data.da, data.db, data.dc, data.N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds / iter / 1000.0f; // 转换为秒
}

bool verify(Data& data) {
    cudaMemcpy(data.hc, data.dc, data.N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < data.N; i++) {
        if (data.hc[i] != data.ha[i] + data.hb[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1<<26;
    const int threads = 256;

    const int blocks_1 = (N + threads - 1) / threads;
    const int blocks_2 = blocks_1; 
    const int blocks_3 = (N/4 + threads - 1) / threads;

    auto data = init(N);

    float t1 = launch(add_naive, data, threads, blocks_1);
    bool v1 = verify(data);
    printf("Naive: %f seconds, Verify: %s\n", t1, v1 ? "PASS" : "FAIL");
    float t2 = launch(add_stride, data, threads, blocks_2);
    bool v2 = verify(data);
    printf("Stride: %f seconds, Verify: %s\n", t2, v2 ? "PASS" : "FAIL");
    float t3 = launch(add_vectorized, data, threads, blocks_3);
    bool v3 = verify(data);
    printf("Vectorized: %f seconds, Verify: %s\n", t3, v3 ? "PASS" : "FAIL");

    cleanup(data);

    long bytes = N * sizeof(float) * 3; // 读取a和b，写入c
    printf("Naive Bandwidth: %f GB/s\n", bytes / t1 / 1e9);
    printf("Stride Bandwidth: %f GB/s\n", bytes / t2 / 1e9);
    printf("Vectorized Bandwidth: %f GB/s\n", bytes / t3 / 1e9);
}