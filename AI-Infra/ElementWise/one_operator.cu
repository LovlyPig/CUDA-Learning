// 一元操作统一模板

#include <stdio.h>
#include <cuda_runtime.h>

// 向量化类型映射
template<typename T, int vecWidth> struct VecType;
template<> struct VecType<float, 4> {
    using Type = float4;
};

template<typename T, typename Op>
__global__ void elementwise_kernel(const T* input, T* output, int N, Op op) {
    using VecT = typename VecType<T, 4>::Type;
    const VecT* input_vec = reinterpret_cast<const VecT*>(input);
    VecT* output_vec = reinterpret_cast<VecT*>(output);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N_vec = N / 4;

    for (int i = idx; i < N_vec; i += stride) {
        VecT in_val = input_vec[i];
        VecT out_val;
        out_val.x = op(in_val.x);
        out_val.y = op(in_val.y);
        out_val.z = op(in_val.z);
        out_val.w = op(in_val.w);
        output_vec[i] = out_val;
    }

    int remain = N_vec * 4;
    for (int i = idx + remain; i < N; i += stride) {
        output[i] = op(input[i]);
}

// 一元算子都可以仿函数实现
struct ReLU {
    __device__ float operator()(float x) const {
        return fmaxf(0.0f, x);
    }
};

struct GeLU {
    __device__ float operator()(float x) const {
        float cdf = 0.5f * (1.0f + tanhf((0.7978845608f * (x + 0.044715f * x * x * x))));
        return x * cdf;
    }
};

int main() {
    const int N = 1<<26;
    const int threads = 256;

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    const int sms = prop.multiProcessorCount;
    const int blocks = sms * 4; // 每SM分配4个block

    elementwise_kernel<<<blocks, threads>>>(/*input*/nullptr, /*output*/nullptr, N, ReLU());
    cudaDeviceSynchronize();
    
}