# std::memory_order

```C++
// since C++11 until C++20
enum memory_order {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
};

// since C++20
enum class memory_order : {
    relaxed, consume, acquire, release, acq_rel, seq_cst
};
inline constexpr memory_order memory_order_relaxed = memory_order::relaxed;
inline constexpr memory_order memory_order_consume = memory_order::consume;
inline constexpr memory_order memory_order_acquire = memory_order::acquire;
inline constexpr memory_order memory_order_release = memory_order::release;
inline constexpr memory_order memory_order_acq_rel = memory_order::acq_rel;
inline constexpr memory_order memory_order_seq_cst = memory_order::seq_cst;
```

## Constants

Defined in header `<atomic>`

- memory_order_relaxed. Relaxed operation: There are no synchronization or ordering constraints imposed on other reads or writes, **only this operation's atomicity is guaranteed.**
- memory_order_consume(deprecated in C++26). Load operation: **no reads or writes in the current thread dependent on the value currently loaded can be reordered before this load.** Writes to data-dependent variables in other threads that release the same atomic variable are visible in the current thread. 
- memory_order_acquire. Load operation: **no reads or writes in the current thread can be reordered before this load.** All writes in other threads that release the same atomic variable are visible in the current thread.
- memory_order_release. Store operation: **no reads or writes in the current thread can be reordered after this store.** All writes in the current thread are visible in other threads that acquire the same atomic variable and writes that carry a dependency into the atomic variable become visible in other threads that consume the same atomic.
- memory_order_acq_rel. **A read-modify-write operation with this memory order is both an acquire operation and a release operation.** No memory reads or writes in the current thread can be reordered before the load, nor after the store. All writes in other threads that release the same atomic variable are visible before the modification and the modification is visible in other threads that acquire the same atomic variable.
- memory_order_seq_cst. A load operation with this memory order performs an acquire operation, a store performs a release operation, and read-modify-write performs both an acquire operation and a release operation, plus a single total order exists in which all threads observe all modifications in the same order.

## Formal description

### sequenced-before
### modification order
### release sequence
### synchronizes with
### inter-thread happens-before

### visible side-effects

The side-effect A on a scalar M (a write) is visible with respect to value computation B on M (a read) if both of the following are true:
1) A happens-before B
2) There is no other side effect X(write) to M where A happens-before X and X happens-before B.

Note: inter-thread synchronization boils down to preventing data races (by establishing happens-before relationships) and defining which side effects become visible under what conditions. Data visibility is strongly related to CPU caching.

### consume operation 
### acquire operation
### release operation

## Explanation

### Relaxed ordering

`memory_order_relaxed` are not synchronization operations; they do not impose an order among concurrent memory accesses. They only guarantee atomicity and modification order consistency.

```C++
// thread 1
r1 = y.load(std::memory_order_relaxed); // A
x.store(r1, std::memory_order_relaxed); // B
// thread2
r2 = x.load(std::memory_order_relaxed); // C
y.store(42, std::memory_order_relaxed); // D

// is allowed to produce r1 == r2 && r2 = 42.
```

**although A is sequenced-before B within thread 1 and C is sequenced before D within thread 2, nothing prevents D from appearing before A in the modification order of y, and B from appearing before C in the modification order of x.** The side-effect of D on y could be visible to the load A in thread 1 while the side effect of B on x could be visible to the load C in thread 2. In particular, this may occur if D is completed before C in thread 2, either due to compiler reordering or at runtime.

**Typical use for relaxed memory ordering is incrementing counters, such as the reference counters of std::shared_ptr, since this only requires atomicity, but not ordering or synchronization (note that decrementing the std::shared_ptr counters requires acquire-release synchronization with the destructor).**
（递增引用计数只需要保证计数值本身的原子累加正确性，不涉及任何的跨线程数据依赖；递减当计数减到0时必须销毁托管对象，此时需要确保其他线程对该对象的所有写入操作都对当前销毁线程可见，并且销毁操作不会被重排到这些写入之前，因此递减需要memory_order_acq_rel或更强的顺序来建立happens-before关系。）

### Release-Acquire ordering

If an atomic store in thread A is `memory_order_release`, an atomic load in thread B from the same variable is `memory_order_acquire`, and the load in thread B reads a value written by the store in thread A, then the store in thread A synchronizes-with the load in thread B.

**All memory writes (including non-atomic and relaxed atomic) that happened-before the atomic store from the point of view of thread A, become visible side-effects in thread B. That is, once the atomic load is completed, thread B is guaranteed to see everything thread A wrote to memory. This promise only holds if B actually returns the value that A stored, or a value from later in the release sequence.**

>Mutual exclusion locks, such as std::mutex or atomic spinlock, are an example of release-acquire synchronization: when the lock is released by thread A and acquired by thread B, everything that took place in the critical section (before the release) in the context of thread A has to be visible to thread B (after the acquire) which is executing the same critical section.

```C++
#include <atomic>
#include <cassert>
#include <string>
#include <thread>

std::atomic<std::string*> ptr;
int data;

void producer() {
    std::string* p = new std::string("Hello");
    // All memory writes that happened-before the atomic store from the point of view of thread A, become visible side-effects in thread B.
    data = 42;
    ptr.store(p, std::memory_order_release);
}

void consumer() {
    std::string* p2;
    // once the atomic load is completed, thread B is guaranteed to see everything thread A wrote to memory.
    while (!(p2 = ptr.load(std::memory_order_acquire)));

    assert(*p2 == "Hello"); // never fires
    assert(data == 42); // never fires
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    t1.join(); t2.join();
}
```

```C++
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

std::vector<int> data;
std::atomic<int> flag = {0};

void thread_1()
{
    data.push_back(42);
    flag.store(1, std::memory_order_release);
}

// 即使中间的 RMW 使用了 relaxed，只要有初始的 release 和最终的 acquire，并且中间全部是 RMW 操作（或 memory_order_seq_cst 的加载），就能保证可见性。它允许你在同步链中插入低开销的 relaxed RMW，而不会破坏 happens‑before 关系。
void thread_2()
{
    int expected = 1;
    // memory_order_relaxed is okay because this is an RMW,
    // and RMWs (with any ordering) following a release form a release sequence
    while (!flag.compare_exchange_strong(expected, 2, std::memory_order_relaxed))
    {
        expected = 1;
    }
}

void thread_3()
{
    while (flag.load(std::memory_order_acquire) < 2)
        ;
    // if we read the value 2 from the atomic flag, we see 42 in the vector
    assert(data.at(0) == 42); // will never fire
}

int main()
{
    std::thread a(thread_1);
    std::thread b(thread_2);
    std::thread c(thread_3);
    a.join(); b.join(); c.join();
}

```



