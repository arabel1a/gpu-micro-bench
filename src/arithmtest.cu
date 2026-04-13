// GPU arithmetic instruction throughput microbenchmark
// Measures throughput and latency of individual SASS instructions via inline PTX.
//
// Usage: ./arithmtest <gpu_id> <n_iters> [n_warmup] [n_reps]
//
// Output: tagged CSV on stdout.
//   GPU,<name>,<sm>,<n_sms>,<l2_kb>
//   ARITH,<op>,<mode>,<n_iters>,<total_ns>,<ns_per_op>,<ops_per_ns>,<wall_ms>
//
// Each test: 1 block, 128 threads (4 warps), __launch_bounds__(128,1).
// Throughput (tput): 8 independent register chains — measures reciprocal throughput.
// Latency (lat): 1 dependent chain — measures pipeline latency.
// 8 ops per loop iteration in both modes.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(e));                   \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// ============================================================================
// Scalar kernel generation macros
// ============================================================================

// --- Float, 1 constant: op %dst, %dst, %const ---
#define KERN_F1(name, op, c_val)                                               \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(float* __restrict__ out, int n) {                           \
    float r0=1.f+1e-4f*threadIdx.x, r1=r0+.1f, r2=r0+.2f, r3=r0+.3f,       \
          r4=r0+.4f, r5=r0+.5f, r6=r0+.6f, r7=r0+.7f, c=(c_val);           \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8;\n\t" op " %1,%1,%8;\n\t"                              \
        op " %2,%2,%8;\n\t" op " %3,%3,%8;\n\t"                              \
        op " %4,%4,%8;\n\t" op " %5,%5,%8;\n\t"                              \
        op " %6,%6,%8;\n\t" op " %7,%7,%8;\n\t"                              \
        :"+f"(r0),"+f"(r1),"+f"(r2),"+f"(r3),                                \
         "+f"(r4),"+f"(r5),"+f"(r6),"+f"(r7):"f"(c));                        \
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;                \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(float* __restrict__ out, int n) {                            \
    float r=1.f+1e-4f*threadIdx.x, c=(c_val);                                \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        :"+f"(r):"f"(c));                                                     \
    out[threadIdx.y*32+threadIdx.x]=r;                                        \
}

// --- Float, 2 constants: op %dst, %dst, %a, %b ---
#define KERN_F2(name, op, a_val, b_val)                                        \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(float* __restrict__ out, int n) {                           \
    float r0=1.f+1e-4f*threadIdx.x, r1=r0+.1f, r2=r0+.2f, r3=r0+.3f,       \
          r4=r0+.4f, r5=r0+.5f, r6=r0+.6f, r7=r0+.7f;                       \
    float a=(a_val), b=(b_val);                                               \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8,%9;\n\t" op " %1,%1,%8,%9;\n\t"                        \
        op " %2,%2,%8,%9;\n\t" op " %3,%3,%8,%9;\n\t"                        \
        op " %4,%4,%8,%9;\n\t" op " %5,%5,%8,%9;\n\t"                        \
        op " %6,%6,%8,%9;\n\t" op " %7,%7,%8,%9;\n\t"                        \
        :"+f"(r0),"+f"(r1),"+f"(r2),"+f"(r3),                                \
         "+f"(r4),"+f"(r5),"+f"(r6),"+f"(r7):"f"(a),"f"(b));                \
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;                \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(float* __restrict__ out, int n) {                            \
    float r=1.f+1e-4f*threadIdx.x, a=(a_val), b=(b_val);                     \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        :"+f"(r):"f"(a),"f"(b));                                             \
    out[threadIdx.y*32+threadIdx.x]=r;                                        \
}

// --- Int, 1 constant: op %dst, %dst, %const ---
#define KERN_I1(name, op, c_val)                                               \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(int* __restrict__ out, int n) {                             \
    int r0=threadIdx.x+1, r1=r0+1, r2=r0+2, r3=r0+3,                        \
        r4=r0+4, r5=r0+5, r6=r0+6, r7=r0+7;                                 \
    int c=(c_val);                                                            \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8;\n\t" op " %1,%1,%8;\n\t"                              \
        op " %2,%2,%8;\n\t" op " %3,%3,%8;\n\t"                              \
        op " %4,%4,%8;\n\t" op " %5,%5,%8;\n\t"                              \
        op " %6,%6,%8;\n\t" op " %7,%7,%8;\n\t"                              \
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),                                \
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(c));                        \
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;                \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(int* __restrict__ out, int n) {                              \
    int r=threadIdx.x+1, c=(c_val);                                          \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        :"+r"(r):"r"(c));                                                     \
    out[threadIdx.y*32+threadIdx.x]=r;                                        \
}

// --- Int, 2 constants: op %dst, %dst, %a, %b ---
#define KERN_I2(name, op, a_val, b_val)                                        \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(int* __restrict__ out, int n) {                             \
    int r0=threadIdx.x+1, r1=r0+1, r2=r0+2, r3=r0+3,                        \
        r4=r0+4, r5=r0+5, r6=r0+6, r7=r0+7;                                 \
    int a=(a_val), b=(b_val);                                                 \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8,%9;\n\t" op " %1,%1,%8,%9;\n\t"                        \
        op " %2,%2,%8,%9;\n\t" op " %3,%3,%8,%9;\n\t"                        \
        op " %4,%4,%8,%9;\n\t" op " %5,%5,%8,%9;\n\t"                        \
        op " %6,%6,%8,%9;\n\t" op " %7,%7,%8,%9;\n\t"                        \
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),                                \
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(a),"r"(b));                \
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;                \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(int* __restrict__ out, int n) {                              \
    int r=threadIdx.x+1, a=(a_val), b=(b_val);                               \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        :"+r"(r):"r"(a),"r"(b));                                             \
    out[threadIdx.y*32+threadIdx.x]=r;                                        \
}

// --- Half2 (f16x2), 1 constant: op %dst, %dst, %const (packed in uint32, "r" constraint) ---
#define KERN_H1(name, op, c_val)                                               \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(int* __restrict__ out, int n) {                             \
    unsigned r0=0x3C003C00u, r1=0x3C013C01u, r2=0x3C023C02u, r3=0x3C033C03u, \
             r4=0x3C043C04u, r5=0x3C053C05u, r6=0x3C063C06u, r7=0x3C073C07u; \
    unsigned c=(c_val);                                                       \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8;\n\t" op " %1,%1,%8;\n\t"                              \
        op " %2,%2,%8;\n\t" op " %3,%3,%8;\n\t"                              \
        op " %4,%4,%8;\n\t" op " %5,%5,%8;\n\t"                              \
        op " %6,%6,%8;\n\t" op " %7,%7,%8;\n\t"                              \
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),                                \
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(c));                        \
    out[threadIdx.y*32+threadIdx.x]=(int)(r0+r1+r2+r3+r4+r5+r6+r7);         \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(int* __restrict__ out, int n) {                              \
    unsigned r=0x3C003C00u; unsigned c=(c_val);                               \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        op " %0,%0,%1;\n\t" op " %0,%0,%1;\n\t"                              \
        :"+r"(r):"r"(c));                                                     \
    out[threadIdx.y*32+threadIdx.x]=(int)r;                                   \
}

// --- Half2 (f16x2), 2 constants: op %dst, %dst, %a, %b ---
#define KERN_H2(name, op, a_val, b_val)                                        \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_tput(int* __restrict__ out, int n) {                             \
    unsigned r0=0x3C003C00u, r1=0x3C013C01u, r2=0x3C023C02u, r3=0x3C033C03u, \
             r4=0x3C043C04u, r5=0x3C053C05u, r6=0x3C063C06u, r7=0x3C073C07u; \
    unsigned a=(a_val), b=(b_val);                                            \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%8,%9;\n\t" op " %1,%1,%8,%9;\n\t"                        \
        op " %2,%2,%8,%9;\n\t" op " %3,%3,%8,%9;\n\t"                        \
        op " %4,%4,%8,%9;\n\t" op " %5,%5,%8,%9;\n\t"                        \
        op " %6,%6,%8,%9;\n\t" op " %7,%7,%8,%9;\n\t"                        \
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),                                \
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(a),"r"(b));                \
    out[threadIdx.y*32+threadIdx.x]=(int)(r0+r1+r2+r3+r4+r5+r6+r7);         \
}                                                                              \
__global__ void __launch_bounds__(128,1)                                       \
kern_##name##_lat(int* __restrict__ out, int n) {                              \
    unsigned r=0x3C003C00u, a=(a_val), b=(b_val);                             \
    for(int i=0;i<n;i++) asm volatile(                                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        op " %0,%0,%1,%2;\n\t" op " %0,%0,%1,%2;\n\t"                        \
        :"+r"(r):"r"(a),"r"(b));                                             \
    out[threadIdx.y*32+threadIdx.x]=(int)r;                                   \
}

// ============================================================================
// Scalar kernel instantiations
// ============================================================================

// FP32
KERN_F1(fadd, "add.f32",   0.000001f)
KERN_F1(fmul, "mul.f32",   0.999999f)
KERN_F2(ffma, "fma.rn.f32", 1.000001f, 0.000001f)

// FP16x2 (packed in uint32)
KERN_H1(hadd2, "add.f16x2",   0x00010001u)   // ~tiny f16
KERN_H1(hmul2, "mul.f16x2",   0x3BFF3BFFu)   // ~0.9998 in f16
KERN_H2(hfma2, "fma.rn.f16x2", 0x3C003C00u, 0x00010001u) // a=1.0, b=tiny

// INT32 arithmetic
KERN_I1(iadd, "add.s32",    1)
KERN_I2(imad, "mad.lo.s32", 3, 1)

// INT8 dot product (special: dp4a dst, a, b, acc — acc is the chain register)
// tput: 8 independent accumulators, shared a/b constants
__global__ void __launch_bounds__(128,1)
kern_dp4a_tput(int* __restrict__ out, int n) {
    int r0=threadIdx.x, r1=r0+1, r2=r0+2, r3=r0+3,
        r4=r0+4, r5=r0+5, r6=r0+6, r7=r0+7;
    int a=0x01010101, b=0x01010101;
    for(int i=0;i<n;i++) asm volatile(
        "dp4a.s32.s32 %0,%8,%9,%0;\n\t" "dp4a.s32.s32 %1,%8,%9,%1;\n\t"
        "dp4a.s32.s32 %2,%8,%9,%2;\n\t" "dp4a.s32.s32 %3,%8,%9,%3;\n\t"
        "dp4a.s32.s32 %4,%8,%9,%4;\n\t" "dp4a.s32.s32 %5,%8,%9,%5;\n\t"
        "dp4a.s32.s32 %6,%8,%9,%6;\n\t" "dp4a.s32.s32 %7,%8,%9,%7;\n\t"
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(a),"r"(b));
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;
}
__global__ void __launch_bounds__(128,1)
kern_dp4a_lat(int* __restrict__ out, int n) {
    int r=threadIdx.x, a=0x01010101;
    for(int i=0;i<n;i++) asm volatile(
        "dp4a.s32.s32 %0,%1,%1,%0;\n\t" "dp4a.s32.s32 %0,%1,%1,%0;\n\t"
        "dp4a.s32.s32 %0,%1,%1,%0;\n\t" "dp4a.s32.s32 %0,%1,%1,%0;\n\t"
        "dp4a.s32.s32 %0,%1,%1,%0;\n\t" "dp4a.s32.s32 %0,%1,%1,%0;\n\t"
        "dp4a.s32.s32 %0,%1,%1,%0;\n\t" "dp4a.s32.s32 %0,%1,%1,%0;\n\t"
        :"+r"(r):"r"(a));
    out[threadIdx.y*32+threadIdx.x]=r;
}

// Bitwise / shift
KERN_I1(shl,  "shl.b32",  1)
KERN_I1(shr,  "shr.s32",  1)
KERN_I1(band, "and.b32",  0x7FFFFFFFu)
KERN_I1(bor,  "or.b32",   0x01)
KERN_I1(bxor, "xor.b32",  0x55555555u)

// LOP3: 3-input logic, used heavily in MMVQ bit extraction
// lop3.b32 d, a, b, c, immLut — we use a=d (chain), b,c = constants
__global__ void __launch_bounds__(128,1)
kern_lop3_tput(int* __restrict__ out, int n) {
    int r0=threadIdx.x+1, r1=r0+1, r2=r0+2, r3=r0+3,
        r4=r0+4, r5=r0+5, r6=r0+6, r7=r0+7;
    int a=0xAAAAAAAA, b=0x55555555;
    for(int i=0;i<n;i++) asm volatile(
        "lop3.b32 %0,%0,%8,%9,0xE8;\n\t" "lop3.b32 %1,%1,%8,%9,0xE8;\n\t"
        "lop3.b32 %2,%2,%8,%9,0xE8;\n\t" "lop3.b32 %3,%3,%8,%9,0xE8;\n\t"
        "lop3.b32 %4,%4,%8,%9,0xE8;\n\t" "lop3.b32 %5,%5,%8,%9,0xE8;\n\t"
        "lop3.b32 %6,%6,%8,%9,0xE8;\n\t" "lop3.b32 %7,%7,%8,%9,0xE8;\n\t"
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(a),"r"(b));
    out[threadIdx.y*32+threadIdx.x]=r0+r1+r2+r3+r4+r5+r6+r7;
}
__global__ void __launch_bounds__(128,1)
kern_lop3_lat(int* __restrict__ out, int n) {
    int r=threadIdx.x+1, a=0xAAAAAAAA, b=0x55555555;
    for(int i=0;i<n;i++) asm volatile(
        "lop3.b32 %0,%0,%1,%2,0xE8;\n\t" "lop3.b32 %0,%0,%1,%2,0xE8;\n\t"
        "lop3.b32 %0,%0,%1,%2,0xE8;\n\t" "lop3.b32 %0,%0,%1,%2,0xE8;\n\t"
        "lop3.b32 %0,%0,%1,%2,0xE8;\n\t" "lop3.b32 %0,%0,%1,%2,0xE8;\n\t"
        "lop3.b32 %0,%0,%1,%2,0xE8;\n\t" "lop3.b32 %0,%0,%1,%2,0xE8;\n\t"
        :"+r"(r):"r"(a),"r"(b));
    out[threadIdx.y*32+threadIdx.x]=r;
}

// PRMT: byte permute, used in MMVQ for bit extraction
__global__ void __launch_bounds__(128,1)
kern_prmt_tput(int* __restrict__ out, int n) {
    unsigned r0=threadIdx.x+1, r1=r0+1, r2=r0+2, r3=r0+3,
             r4=r0+4, r5=r0+5, r6=r0+6, r7=r0+7;
    unsigned src=0xDEADBEEFu, sel=0x3210u;
    for(int i=0;i<n;i++) asm volatile(
        "prmt.b32 %0,%0,%8,%9;\n\t" "prmt.b32 %1,%1,%8,%9;\n\t"
        "prmt.b32 %2,%2,%8,%9;\n\t" "prmt.b32 %3,%3,%8,%9;\n\t"
        "prmt.b32 %4,%4,%8,%9;\n\t" "prmt.b32 %5,%5,%8,%9;\n\t"
        "prmt.b32 %6,%6,%8,%9;\n\t" "prmt.b32 %7,%7,%8,%9;\n\t"
        :"+r"(r0),"+r"(r1),"+r"(r2),"+r"(r3),
         "+r"(r4),"+r"(r5),"+r"(r6),"+r"(r7):"r"(src),"r"(sel));
    out[threadIdx.y*32+threadIdx.x]=(int)(r0+r1+r2+r3+r4+r5+r6+r7);
}
__global__ void __launch_bounds__(128,1)
kern_prmt_lat(int* __restrict__ out, int n) {
    unsigned r=threadIdx.x+1, src=0xDEADBEEFu, sel=0x3210u;
    for(int i=0;i<n;i++) asm volatile(
        "prmt.b32 %0,%0,%1,%2;\n\t" "prmt.b32 %0,%0,%1,%2;\n\t"
        "prmt.b32 %0,%0,%1,%2;\n\t" "prmt.b32 %0,%0,%1,%2;\n\t"
        "prmt.b32 %0,%0,%1,%2;\n\t" "prmt.b32 %0,%0,%1,%2;\n\t"
        "prmt.b32 %0,%0,%1,%2;\n\t" "prmt.b32 %0,%0,%1,%2;\n\t"
        :"+r"(r):"r"(src),"r"(sel));
    out[threadIdx.y*32+threadIdx.x]=(int)r;
}

// ============================================================================
// Tensor core MMA kernels
//
// Warp-cooperative: all 32 threads in warp participate.
// Throughput: 4 independent D-register chains × 2 rounds = 8 MMA ops/iter.
// Latency: 1 D chain, 8 dependent MMA ops/iter.
// ============================================================================

// Helper: MMA instruction prefix strings

// sm_80+ F16/BF16 shape: m16n8k16, D=4, A=4, B=2
#define MMA_F16_80   "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
#define MMA_BF16_80  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
// Tput macros for A=4, B=2: A=%16-19, B=%20-21
#define MMA_K16A    "{%16,%17,%18,%19}, "
#define MMA_K16B    "{%20,%21}, "

// sm_75 F16 shape: m8n8k4, D=8, A=2, B=2
#define MMA_F16_75   "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "

// sm_80+ TF32 shape: m16n8k4, D=4, A=2(.b32), B=1(.b32)
#define MMA_TF32_OP "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
#define MMA_TF32A   "{%16,%17}, "
#define MMA_TF32B   "{%18}, "

#define MMA_S8_OP  "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "

// --- HMMA F16 (sm_80+: m16n8k16, sm_75: m8n8k4) ---
__global__ void __launch_bounds__(128,1)
kern_hmma_f16_tput(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    // m16n8k16: D=4, A=4, B=2.  4 chains × 4 D = 16 regs; A=%16-19, B=%20-21
    float d0=0,d1=0,d2=0,d3=0, d4=0,d5=0,d6=0,d7=0,
          d8=0,d9=0,d10=0,d11=0, d12=0,d13=0,d14=0,d15=0;
    unsigned a0=0x3C003C00u,a1=a0,a2=a0,a3=a0, b0=0x3C003C00u,b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_F16_80 "{%0,%1,%2,%3}, "     MMA_K16A MMA_K16B "{%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%4,%5,%6,%7}, "     MMA_K16A MMA_K16B "{%4,%5,%6,%7};\n\t"
        MMA_F16_80 "{%8,%9,%10,%11}, "   MMA_K16A MMA_K16B "{%8,%9,%10,%11};\n\t"
        MMA_F16_80 "{%12,%13,%14,%15}, " MMA_K16A MMA_K16B "{%12,%13,%14,%15};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, "     MMA_K16A MMA_K16B "{%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%4,%5,%6,%7}, "     MMA_K16A MMA_K16B "{%4,%5,%6,%7};\n\t"
        MMA_F16_80 "{%8,%9,%10,%11}, "   MMA_K16A MMA_K16B "{%8,%9,%10,%11};\n\t"
        MMA_F16_80 "{%12,%13,%14,%15}, " MMA_K16A MMA_K16B "{%12,%13,%14,%15};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),
         "+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),
         "+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0+d4+d8+d12;
#else
    // m8n8k4: D=8, A=2, B=2.  2 chains × 8 D = 16 regs; A=%16-17, B=%18-19
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0,
          d8=0,d9=0,d10=0,d11=0,d12=0,d13=0,d14=0,d15=0;
    unsigned a0=0x3C003C00u, a1=a0, b0=0x3C003C00u, b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, "     "{%16,%17}, " "{%18,%19}, " "{%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%8,%9,%10,%11,%12,%13,%14,%15}, " "{%16,%17}, " "{%18,%19}, " "{%8,%9,%10,%11,%12,%13,%14,%15};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, "     "{%16,%17}, " "{%18,%19}, " "{%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%8,%9,%10,%11,%12,%13,%14,%15}, " "{%16,%17}, " "{%18,%19}, " "{%8,%9,%10,%11,%12,%13,%14,%15};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, "     "{%16,%17}, " "{%18,%19}, " "{%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%8,%9,%10,%11,%12,%13,%14,%15}, " "{%16,%17}, " "{%18,%19}, " "{%8,%9,%10,%11,%12,%13,%14,%15};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, "     "{%16,%17}, " "{%18,%19}, " "{%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%8,%9,%10,%11,%12,%13,%14,%15}, " "{%16,%17}, " "{%18,%19}, " "{%8,%9,%10,%11,%12,%13,%14,%15};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),"+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15)
        :"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0+d8;
#endif
}
__global__ void __launch_bounds__(128,1)
kern_hmma_f16_lat(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    // m16n8k16: 1 chain, D=4, A=4, B=2
    float d0=0,d1=0,d2=0,d3=0;
    unsigned a0=0x3C003C00u,a1=a0,a2=a0,a3=a0, b0=0x3C003C00u,b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_F16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0;
#else
    // m8n8k4: 1 chain, D=8, A=2, B=2
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
    unsigned a0=0x3C003C00u, a1=a0, b0=0x3C003C00u, b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        MMA_F16_75 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
        :"r"(a0),"r"(a1),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0;
#endif
}

// --- HMMA BF16 (m16n8k16, sm_80+) ---
// A=4 regs, B=2 regs, C/D=4 regs (same layout as F16)
__global__ void __launch_bounds__(128,1)
kern_hmma_bf16_tput(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    float d0=0,d1=0,d2=0,d3=0, d4=0,d5=0,d6=0,d7=0,
          d8=0,d9=0,d10=0,d11=0, d12=0,d13=0,d14=0,d15=0;
    unsigned a0=0x3F803F80u,a1=a0,a2=a0,a3=a0, b0=0x3F803F80u,b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_BF16_80 "{%0,%1,%2,%3}, "     MMA_K16A MMA_K16B "{%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%4,%5,%6,%7}, "     MMA_K16A MMA_K16B "{%4,%5,%6,%7};\n\t"
        MMA_BF16_80 "{%8,%9,%10,%11}, "   MMA_K16A MMA_K16B "{%8,%9,%10,%11};\n\t"
        MMA_BF16_80 "{%12,%13,%14,%15}, " MMA_K16A MMA_K16B "{%12,%13,%14,%15};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, "     MMA_K16A MMA_K16B "{%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%4,%5,%6,%7}, "     MMA_K16A MMA_K16B "{%4,%5,%6,%7};\n\t"
        MMA_BF16_80 "{%8,%9,%10,%11}, "   MMA_K16A MMA_K16B "{%8,%9,%10,%11};\n\t"
        MMA_BF16_80 "{%12,%13,%14,%15}, " MMA_K16A MMA_K16B "{%12,%13,%14,%15};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),
         "+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),
         "+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0+d4+d8+d12;
#endif
}
__global__ void __launch_bounds__(128,1)
kern_hmma_bf16_lat(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    float d0=0,d1=0,d2=0,d3=0;
    unsigned a0=0x3F803F80u,a1=a0,a2=a0,a3=a0, b0=0x3F803F80u,b1=b0;
    for(int i=0;i<n;i++) asm volatile(
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        MMA_BF16_80 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    out[threadIdx.y*32+threadIdx.x]=d0;
#endif
}

// --- HMMA TF32 (m16n8k4, sm_80+) ---
// A=2 regs (.b32), B=1 reg (.b32), C/D=4 regs (.f32)
__global__ void __launch_bounds__(128,1)
kern_hmma_tf32_tput(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    float d0=0,d1=0,d2=0,d3=0, d4=0,d5=0,d6=0,d7=0,
          d8=0,d9=0,d10=0,d11=0, d12=0,d13=0,d14=0,d15=0;
    unsigned a0=0x3F800000u, a1=a0, b0=0x3F800000u; // 1.0f as bits
    for(int i=0;i<n;i++) asm volatile(
        MMA_TF32_OP "{%0,%1,%2,%3}, "     MMA_TF32A MMA_TF32B "{%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%4,%5,%6,%7}, "     MMA_TF32A MMA_TF32B "{%4,%5,%6,%7};\n\t"
        MMA_TF32_OP "{%8,%9,%10,%11}, "   MMA_TF32A MMA_TF32B "{%8,%9,%10,%11};\n\t"
        MMA_TF32_OP "{%12,%13,%14,%15}, " MMA_TF32A MMA_TF32B "{%12,%13,%14,%15};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, "     MMA_TF32A MMA_TF32B "{%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%4,%5,%6,%7}, "     MMA_TF32A MMA_TF32B "{%4,%5,%6,%7};\n\t"
        MMA_TF32_OP "{%8,%9,%10,%11}, "   MMA_TF32A MMA_TF32B "{%8,%9,%10,%11};\n\t"
        MMA_TF32_OP "{%12,%13,%14,%15}, " MMA_TF32A MMA_TF32B "{%12,%13,%14,%15};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),
         "+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7),
         "+f"(d8),"+f"(d9),"+f"(d10),"+f"(d11),
         "+f"(d12),"+f"(d13),"+f"(d14),"+f"(d15)
        :"r"(a0),"r"(a1),"r"(b0));
    out[threadIdx.y*32+threadIdx.x]=d0+d4+d8+d12;
#endif
}
__global__ void __launch_bounds__(128,1)
kern_hmma_tf32_lat(float* __restrict__ out, int n) {
#if __CUDA_ARCH__ >= 800
    float d0=0,d1=0,d2=0,d3=0;
    unsigned a0=0x3F800000u, a1=a0, b0=0x3F800000u;
    for(int i=0;i<n;i++) asm volatile(
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        MMA_TF32_OP "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n\t"
        :"+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
        :"r"(a0),"r"(a1),"r"(b0));
    out[threadIdx.y*32+threadIdx.x]=d0;
#endif
}

// --- IMMA S8 (m8n8k16, sm_75+) ---
// A=1 reg, B=1 reg, C/D=2 regs. Tput: 4 chains × 2 D regs = %0-%7
__global__ void __launch_bounds__(128,1)
kern_imma_s8_tput(int* __restrict__ out, int n) {
    int d0=0,d1=0, d2=0,d3=0, d4=0,d5=0, d6=0,d7=0;
    int a0=0x01010101, b0=0x01010101;
    for(int i=0;i<n;i++) asm volatile(
        MMA_S8_OP "{%0,%1}, {%8}, {%9}, {%0,%1};\n\t"
        MMA_S8_OP "{%2,%3}, {%8}, {%9}, {%2,%3};\n\t"
        MMA_S8_OP "{%4,%5}, {%8}, {%9}, {%4,%5};\n\t"
        MMA_S8_OP "{%6,%7}, {%8}, {%9}, {%6,%7};\n\t"
        MMA_S8_OP "{%0,%1}, {%8}, {%9}, {%0,%1};\n\t"
        MMA_S8_OP "{%2,%3}, {%8}, {%9}, {%2,%3};\n\t"
        MMA_S8_OP "{%4,%5}, {%8}, {%9}, {%4,%5};\n\t"
        MMA_S8_OP "{%6,%7}, {%8}, {%9}, {%6,%7};\n\t"
        :"+r"(d0),"+r"(d1),"+r"(d2),"+r"(d3),
         "+r"(d4),"+r"(d5),"+r"(d6),"+r"(d7)
        :"r"(a0),"r"(b0));
    out[threadIdx.y*32+threadIdx.x]=d0+d2+d4+d6;
}
__global__ void __launch_bounds__(128,1)
kern_imma_s8_lat(int* __restrict__ out, int n) {
    int d0=0, d1=0, a0=0x01010101, b0=0x01010101;
    for(int i=0;i<n;i++) asm volatile(
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        MMA_S8_OP "{%0,%1}, {%2}, {%3}, {%0,%1};\n\t"
        :"+r"(d0),"+r"(d1):"r"(a0),"r"(b0));
    out[threadIdx.y*32+threadIdx.x]=d0;
}

// ============================================================================
// Runner
// ============================================================================

struct Test {
    const char* name;
    void (*tput_f)(float*, int);
    void (*tput_i)(int*, int);
    void (*lat_f)(float*, int);
    void (*lat_i)(int*, int);
    bool is_int;
    int min_sm;  // minimum compute capability × 10 (e.g. 75, 80, 86)
};

void run_test(const Test& t, int n_iters, int n_warmup, int n_reps, int gpu_sm) {
    if (gpu_sm < t.min_sm) {
        fprintf(stderr, "  %s: skipped (needs sm_%d, have sm_%d)\n", t.name, t.min_sm, gpu_sm);
        return;
    }

    float* d_fout; int* d_iout;
    CHECK(cudaMalloc(&d_fout, 128 * sizeof(float)));
    CHECK(cudaMalloc(&d_iout, 128 * sizeof(int)));

    dim3 block(32, 4);
    int ops_per_iter = 8;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto bench_f = [&](const char* mode, void (*kern)(float*, int)) {
        for (int i = 0; i < n_warmup; i++) kern<<<1, block>>>(d_fout, n_iters);
        CHECK(cudaDeviceSynchronize());
        float total_ms = 0;
        for (int rep = 0; rep < n_reps; rep++) {
            CHECK(cudaEventRecord(start));
            kern<<<1, block>>>(d_fout, n_iters);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        float avg_ms = total_ms / n_reps;
        double total_ns = avg_ms * 1e6;
        long long total_ops = (long long)n_iters * ops_per_iter;
        printf("ARITH,%s,%s,%d,%.1f,%.3f,%.4f,%.1f\n",
               t.name, mode, n_iters, total_ns, total_ns/total_ops,
               total_ops/total_ns, total_ms);
    };

    auto bench_i = [&](const char* mode, void (*kern)(int*, int)) {
        for (int i = 0; i < n_warmup; i++) kern<<<1, block>>>(d_iout, n_iters);
        CHECK(cudaDeviceSynchronize());
        float total_ms = 0;
        for (int rep = 0; rep < n_reps; rep++) {
            CHECK(cudaEventRecord(start));
            kern<<<1, block>>>(d_iout, n_iters);
            CHECK(cudaEventRecord(stop));
            CHECK(cudaEventSynchronize(stop));
            float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }
        float avg_ms = total_ms / n_reps;
        double total_ns = avg_ms * 1e6;
        long long total_ops = (long long)n_iters * ops_per_iter;
        printf("ARITH,%s,%s,%d,%.1f,%.3f,%.4f,%.1f\n",
               t.name, mode, n_iters, total_ns, total_ns/total_ops,
               total_ops/total_ns, total_ms);
    };

    if (t.is_int) { bench_i("tput", t.tput_i); bench_i("lat", t.lat_i); }
    else          { bench_f("tput", t.tput_f); bench_f("lat", t.lat_f); }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_fout));
    CHECK(cudaFree(d_iout));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gpu_id> <n_iters> [n_warmup] [n_reps]\n", argv[0]);
        return 1;
    }

    int gpu_id = atoi(argv[1]);
    int n_iters = atoi(argv[2]);
    int n_warmup = argc > 3 ? atoi(argv[3]) : 10;
    int n_reps = argc > 4 ? atoi(argv[4]) : 50;

    CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    int l2_kb = prop.l2CacheSize / 1024;
    int gpu_sm = prop.major * 10 + prop.minor;

    printf("GPU,%s,sm_%d%d,%d,%d\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, l2_kb);
    fprintf(stderr, "GPU: %s (sm_%d), SMs: %d\n", prop.name, gpu_sm, prop.multiProcessorCount);
    fprintf(stderr, "n_iters=%d, n_warmup=%d, n_reps=%d, ops_per_iter=8\n",
            n_iters, n_warmup, n_reps);

    // F = float output, I = int output, min_sm
    #define F_TEST(name, sm) {#name, (void(*)(float*,int))kern_##name##_tput, nullptr, \
                              (void(*)(float*,int))kern_##name##_lat, nullptr, false, sm}
    #define I_TEST(name, sm) {#name, nullptr, (void(*)(int*,int))kern_##name##_tput, \
                              nullptr, (void(*)(int*,int))kern_##name##_lat, true, sm}

    Test tests[] = {
        // FP32 scalar
        F_TEST(fadd, 75),
        F_TEST(fmul, 75),
        F_TEST(ffma, 75),
        // FP16x2 scalar
        I_TEST(hadd2, 75),
        I_TEST(hmul2, 75),
        I_TEST(hfma2, 75),
        // INT32 arithmetic
        I_TEST(iadd, 75),
        I_TEST(imad, 75),
        // INT8 dot product
        I_TEST(dp4a, 75),
        // Bitwise / shift
        I_TEST(shl, 75),
        I_TEST(shr, 75),
        I_TEST(band, 75),
        I_TEST(bor, 75),
        I_TEST(bxor, 75),
        I_TEST(lop3, 75),
        I_TEST(prmt, 75),
        // Tensor core MMA
        F_TEST(hmma_f16, 75),
        F_TEST(hmma_bf16, 80),
        F_TEST(hmma_tf32, 80),
        I_TEST(imma_s8, 75),
    };

    for (auto& t : tests)
        run_test(t, n_iters, n_warmup, n_reps, gpu_sm);

    return 0;
}
