/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/



#ifndef ONEFLOW_CORE_CUDA_ELEMENTWISE_H_
#define ONEFLOW_CORE_CUDA_ELEMENTWISE_H_

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace oneflow {

namespace cuda {

namespace elementwise {

constexpr int kBlockSize = 256;           
constexpr int kNumWaves = 32;

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
  }
  *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                   sm_count * tpm / kBlockSize * kNumWaves));
  return cudaSuccess;
}

// std::aligned_storage<Len, Align> 生成一个 未初始化的 POD 类型，
// 其大小为 Len 字节，对齐要求为 Align 字节
template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

// 类型别名
// PackType<T, pack_size> 表示上述对齐存储类型
template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

// union使得storge和elem共享同一块内存
template<typename T, int pack_size>
union Pack {
  // 确保PackType的大小确实等于sizeof(T) * pack_size
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage; // 未初始化的原始内存块
  T elem[pack_size];
};

// 按指定对齐方式打包多个元素
template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ Packed() {
    // do nothing
  }
  union {
    T elem[pack_size];
  };
};

// 16字节是 NVIDIA GPU 向量加载指令的常见位宽
constexpr int kMaxPackBytes = 128 / 8; // 16字节 
// CUDA 内置向量类型支持的最大元素个数 如 half8 等
constexpr int kMaxPackSize = 8;  // 最大元素个数 8

constexpr int Min(int a, int b) { return a < b ? a : b; }

template<typename T>
constexpr int PackSize() {
  return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

// 递归变参模板
// 对传入的所有类型分别计算各自的 PackSize，并返回最小值
// 保证所有数组都能用相同的 pack_size 进行向量化访问
// 提供编译期常量，用于模板参数（前面的打包类型的pack_size）
template<typename T, typename U, typename... Args>
constexpr int PackSize() {
  return Min(PackSize<T>(), PackSize<U, Args...>());
}

// SFINAE 类型特征，用于在编译期检测类型 T 是否拥有 Apply2 成员函数
template<typename T>
class HasApply2 {
  typedef char one;
  struct two {
    char x[2];
  };

  template<typename C>
  static one test(decltype(&C::Apply2));
  template<typename C>
  static two test(...);

  // 如果 T::Apply2 存在，则 test<T>(0) 会匹配第一个重载（返回 one，大小为 1 字节），
  // 否则匹配第二个重载（返回 two，大小为 2 字节）。
  // 最终 value 为 true 表示该 Functor 支持 Apply2。
 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

// 条件：HasApply2 == true 且 pack_size 为偶数
template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == true && pack_size % 2 == 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
#pragma unroll
  for (int j = 0; j < pack_size; j += 2) { functor.Apply2(ret.elem + j, (in.elem + j)...); }
  return ret;
}

// 通用逐元素处理
// std::enable_if<B, T>：是一个类型模板，当 B 为 true 时，它内部有 type 成员，且 type 被定义为 T；
// 当 B 为 false 时，type 不存在
template<int pack_size, typename FunctorT, typename R, typename... IN>
__device__ typename std::enable_if<HasApply2<FunctorT>::value == false || pack_size % 2 != 0,
                                   Packed<R, pack_size>>::type
ApplyPack(const FunctorT& functor, const Packed<IN, pack_size>... in) {
  Packed<R, pack_size> ret;
#pragma unroll
  for (int j = 0; j < pack_size; ++j) { ret.elem[j] = functor((in.elem[j])...); }
  return ret;
}

template<int pack_size, typename FactoryT, typename R, typename... IN>
__global__ void __launch_bounds__(kBlockSize)
    ApplyGeneric(FactoryT factory, int64_t n_pack, Packed<R, pack_size>* pack_r,
                 const Packed<IN, pack_size>*... pack_in, int64_t n_tail, R* tail_r,
                 const IN*... tail_in) {
  auto functor = factory();
  const int global_tid = blockIdx.x * kBlockSize + threadIdx.x;
  for (int64_t i = global_tid; i < n_pack; i += blockDim.x * gridDim.x) {
    pack_r[i] = ApplyPack<pack_size, decltype(functor), R, IN...>(functor, (pack_in[i])...);
  }
  if (global_tid < n_tail) { tail_r[global_tid] = functor((tail_in[global_tid])...); }
}

// 将预先构造好的 functor 对象传递到内核，并在内核中重新获得它的副本。
// 它的存在是为了解耦构造时机，避免跨设备拷贝的复杂性
template<typename FunctorT>
struct SimpleFactory {
  explicit SimpleFactory(FunctorT functor) : tpl(functor) {}
  __device__ FunctorT operator()() const { return tpl; }

 private:
  FunctorT tpl;
};

template<size_t pack_size>
bool IsAlignedForPack() {
  return true;
}

template<size_t pack_size, typename T, typename... Args>
bool IsAlignedForPack(const T* ptr, const Args*... others) {
  return reinterpret_cast<uintptr_t>(ptr) % sizeof(Pack<T, pack_size>) == 0
         && IsAlignedForPack<pack_size, Args...>(others...);
}

template<size_t pack_size, typename FactoryT, typename R, typename... IN>
cudaError_t LaunchKernel(FactoryT factory, int64_t n, R* r, const IN*... in, cudaStream_t stream) {
  const int64_t n_pack = n / pack_size;
  const int64_t tail_offset = n_pack * pack_size;
  const int64_t n_tail = n - tail_offset;
  int num_blocks;
  {
    cudaError_t err = GetNumBlocks(n_pack, &num_blocks);
    if (err != cudaSuccess) { return err; }
  }
  ApplyGeneric<pack_size, FactoryT, R, IN...><<<num_blocks, kBlockSize, 0, stream>>>(
      factory, n_pack, reinterpret_cast<Packed<R, pack_size>*>(r),
      (reinterpret_cast<const Packed<IN, pack_size>*>(in))..., n_tail, r + tail_offset,
      (in + tail_offset)...);
  return cudaPeekAtLastError();
}

template<typename FactoryT, typename R, typename... IN>
struct GenericLauncher {
  static cudaError_t Launch(FactoryT factory, int64_t n, R* r, const IN*... in,
                            cudaStream_t stream) {
    constexpr int max_pack_size = PackSize<R, IN...>();
    if (IsAlignedForPack<max_pack_size, R, IN...>(r, in...)) {
      return LaunchKernel<max_pack_size, FactoryT, R, IN...>(factory, n, r, in..., stream);
    } else {
      return LaunchKernel<1, FactoryT, R, IN...>(factory, n, r, in..., stream);
    }
  }
};

template<typename FactoryT, typename R, typename A>
inline cudaError_t UnaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a,
                                    cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A>::Launch(factory, n, r, a, stream);
}

template<typename FunctorT, typename R, typename A>
inline cudaError_t Unary(FunctorT functor, int64_t n, R* r, const A* a, cudaStream_t stream) {
  return UnaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, stream);
}

template<typename FactoryT, typename R, typename A, typename B>
inline cudaError_t BinaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                     cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B>::Launch(factory, n, r, a, b, stream);
}

template<typename FunctorT, typename R, typename A, typename B>
inline cudaError_t Binary(FunctorT functor, int64_t n, R* r, const A* a, const B* b,
                          cudaStream_t stream) {
  return BinaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, stream);
}

template<typename FactoryT, typename R, typename A, typename B, typename C>
inline cudaError_t TernaryWithFactory(FactoryT factory, int64_t n, R* r, const A* a, const B* b,
                                      const C* c, cudaStream_t stream) {
  return GenericLauncher<FactoryT, R, A, B, C>::Launch(factory, n, r, a, b, c, stream);
}

template<typename FunctorT, typename R, typename A, typename B, typename C>
inline cudaError_t Ternary(FunctorT functor, int64_t n, R* r, const A* a, const B* b, const C* c,
                           cudaStream_t stream) {
  return TernaryWithFactory(SimpleFactory<FunctorT>(functor), n, r, a, b, c, stream);
}

}  // namespace elementwise

}  // namespace cuda

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CUDA_ELEMENTWISE_H_