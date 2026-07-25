#include <iostream>

// 向上对齐2的幂
static inline size_t align_up(size_t n, size_t align) {
    // assert((align & (align - 1)) == 0); 
    return (n + align - 1) & ~(align - 1);
}

static inline size_t align_down(size_t n, size_t align) {
    // assert((align & (align - 1)) == 0); 
    return n & ~(align - 1);
}

static inline size_t ceil_div(size_t n, size_t d) {
    return (n + d - 1) / d;
}