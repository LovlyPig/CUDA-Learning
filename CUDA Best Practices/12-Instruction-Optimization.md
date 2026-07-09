## 12 Instruction Optimization

Best practices suggest that this optimization be performed after all higher-level optimizations have been completed.

### 12.1 Arithmetic Instructions

To maximize instruction throughput the application should:

- Minimize the use of arithmetic instructions with low throughput; this includes trading precision for speed when it does not affect the end result, **such as using intrinsic instead of regular functions, single-precision instead of double-precision, or flushing denormalized numbers to zero**; \url{https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html}
- Minimize divergent warps caused by control flow instructions;
- Reduce the number of instructions, for example, by optimizing out synchronization points（同步点） whenever possible or by using restricted pointers.
（当两个或多个指针指向相同或重叠的内存区域时，就发生了别名现象（Aliasing）。当编译器无法确定两个指针是否指向同一块内存时，它会做最保守的假设：认为它们指向同一块内存。为处理这种潜在的别名风险，编译器会放弃许多激进的优化。`__restrict__`关键字是一个指针限定符，用于修饰指针，告诉编译器：这个指针是访问其指向内存区域的唯一途径。在本作用域内，没有任何其他指针会指向或重叠这块内存区域。---> 缓存数据：将被 `__restrict__` 修饰的指针指向的数据大胆地缓存在寄存器或共享内存中，避免重复、昂贵的显存访问; 指令重排与优化：可以自由地重排指令顺序，或合并相同的计算（公共子表达式消除），而不用担心破坏程序的正确性。但是启用优化可能会增加寄存器使用量。在寄存器资源紧张的 CUDA 内核中，过度使用可能因占用率下降而导致性能不升反降。）

In this section, throughputs are given in number of operations per clock cycle per multiprocessor. For a warp size of 32, one instruction corresponds to 32 operations, so if N is the number of operations per clock cycle, the instruction throughput is N/32 instructions per clock cycle. All throughputs are for one multiprocessor. They must be multiplied by the number of multiprocessors in the device to get throughput for the whole device.

#### 12.1.1 Throught of Native Arithmetic Instructions

>This table reflects the max theoretical throughput of the mentioned operations.
>In some cases, these throughputs may only be achieved via specific sequences of instructions that require special care when using a compiler.
>Most rows provide a PTX instruction as an example of lower-level instruction expected to achieve the listed throughput. This PTX instruction may not be the only instructions able to perform the described operation(s).

table see \url{https://docs.nvidia.com/cuda/archive/13.2.0/cuda-c-best-practices-guide/index.html#instruction-optimization}.

In general, code compiled with `-ftz=true` (denormalized numbers are flushed to zero 将那些极其微小的非规格化浮点数直接当作 0 来处理) tends to have higher performance than code compiled with `-ftz=false`. 

**Single-Precision Floating-Point Division**
`__fdividef(x, y)` provides faster single-precision floating-point division than the division operator.

**Single-Precision Floating-Point Reciprocal Square Root**

To preserve IEEE-754 semantics the compiler can optimize `1.0/sqrtf()` into `rsqrtf()` only when both reciprocal and square root are approximate, (i.e., with -prec-div=false and -prec-sqrt=false). It is therefore recommended to invoke `rsqrtf()` directly where desired.
但是，`rsqrtf()` 有一个致命弱点：它不保证 IEEE-754 的精确舍入标准，通常只保证相对误差在千分之几以内（约 23 位精度，而非标准的 24 位）。

**Single-Precision Floating-Point Square Root**
Single-precision floating-point square root is implemented as a reciprocal square root followed by a reciprocal instead of a reciprocal square root followed by a multiplication so that it gives correct results for 0 and infinity.

**Sine and Cosine**

**Integer Arithmetic**
Integer division and modulo operation are costly as they compile to up to 20 instructions. They can be replaced with bitwise operations in some cases: If n is a power of 2, `(i/n)` is equivalent to `(i>>log2(n))` and `(i%n)` is equivalent to `(i&(n-1))`; the compiler will perform these conversions if n is literal（字面值）.

`__brev`(位反转) and `__popc`(数‘1’的个数) map to a single instruction and `__brevll` and `__popcll` to a few instructions.

**Half Precision Arithmetic**

**Type Conversion**
Functions operating on variables of type `char` or `short` whose operands generally need to be converted to `int`.

#### 12.1.2 Control Flow Instructions

Any flow control instruction (`if, switch, do, for, while`) can significantly impact the effective instruction throughput by causing threads of the same warp to diverge.

To obtain best performance in cases where the control flow depends on the thread ID, the controlling condition should be written so as to minimize the number of divergent warps. This is possible because the distribution of the warps across the block is deterministic.

#### 12.1.3 Synchronization Instruction

Throughput for `__syncthreads()` is 32 operations per clock cycle for devices of compute capability 6.0, 16 operations per clock cycle for devices of compute capability 7.x as well as 8.x and 64 operations per clock cycle for devices of compute capability 5.x, 6.1 and 6.2.

#### 12.1.4 Division Modulo Operations

#### 12.1.5 Loop Counters Signed vs. Unsigned

>**Low Medium Priority**: Use signed integers rather than unsigned integers as loop counters.

```C++
for (i = 0; i < n; ++i) {
    out[i] = in[offset + stride*i];
}
```

Here, the sub-expression `stride*i` could overflow a 32-bit integer, so if `i` is declared as unsigned, the overflow semantics prevent the compiler from using some optimizations that might otherwise have applied, such as strength reduction. If instead `i` is declared as signed, where the overflow semantics are undefined, the compiler has more leeway to use these optimizations.

#### 12.1.7 Other Arithmetic Instructions

### 12.2 Memory Instructions

> High Priority: Minimize the use of global memory. Prefer shared memory access where possible.

## 13 Control FLow

### 13.1 Branching and Divergence

> High Priority: Avoid different execution paths within the same warp.

For branches including just a few instructions, warp divergence generally results in marginal performance losses. For example, the compiler may use predication to avoid an actual branch. Instead, all instructions are scheduled, but a per-thread condition code or predicate controls which threads execute the instructions. Threads with a false predicate do not write results, and also do not evaluate addresses or read operands.

### 13.2 Branch Predication

When using branch predication, none of the instructions whose execution depends on the controlling condition is skipped. Instead, each such instruction is associated with a per-thread condition code or predicate that is set to true or false according to the controlling condition. Although each of these instructions is scheduled for execution, only the instructions with a true predicate are actually executed. Instructions with a false predicate do not write results, and they also do not evaluate addresses or read operands.

