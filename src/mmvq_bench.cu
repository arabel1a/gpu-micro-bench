// Isolated MMVQ Q1_0_g128 kernel benchmark.
// Extracts the exact kernel from llama.cpp and runs it on random data
// to measure pure MMVQ throughput without scheduler/framework overhead.
//
// Usage: ./mmvq_bench <gpu_id> <n_warmup> <n_reps> <layers_csv>
//   layers_csv: "name:nrows:ncols:count,name:nrows:ncols:count,..."
//
// Output (CSV to stdout):
//   GPU,<name>,<sm>,<n_sms>,<l2_kb>
//   LAYER,<name>,<nrows>,<ncols>,<count>,<total_us>,<per_call_us>,<weight_mb>,<eff_bw_gbps>
//   SUMMARY,<us_per_token>,<equiv_tps>,<total_weight_mb>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <sstream>

// ============================================================================
// Block structs — exact copies from llama.cpp ggml-common.h
// ============================================================================

#define QK1_0_g128 128
#define QK8_1 32

struct block_q1_0_g128 {
    half d;                        // scale (2 bytes)
    uint8_t qs[QK1_0_g128 / 8];   // 1-bit quants (16 bytes)
};  // 18 bytes total
static_assert(sizeof(block_q1_0_g128) == 18, "wrong q1_0_g128 block size");

struct block_q8_1 {
    half2 ds;                      // delta + sum (4 bytes)
    int8_t qs[QK8_1];             // quants (32 bytes)
};  // 36 bytes total
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");

// ============================================================================
// Device helpers — exact copies from llama.cpp
// ============================================================================

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32];
}

// static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
// #if (__CUDA_ARCH__ >= 610) && !defined(DISABLE_DP4A)
//     return __dp4a(a, b, c);
// #else
//     const int8_t * a8 = (const int8_t *) &a;
//     const int8_t * b8 = (const int8_t *) &b;
//     return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
// #endif
// }
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(DP4A_REPL_DP2A)
    int8_t a0 = (int8_t)(a >> 0);
    int8_t a1 = (int8_t)(a >> 8);
    int8_t a2 = (int8_t)(a >> 16);
    int8_t a3 = (int8_t)(a >> 24);

    int16_t a_lo = (int16_t)a0 | ((int16_t)a1 << 8);
    int16_t a_hi = (int16_t)a2 | ((int16_t)a3 << 8);
    int32_t a_packed = ((int32_t)a_hi << 16) | (uint32_t)a_lo;

    int part_lo = __dp2a_lo(a_packed, b, c);
    int part_hi = __dp2a_hi(a_packed, b, 0);
    return part_lo + part_hi;

#elif (__CUDA_ARCH__ >= 610) && !defined(DISABLE_DP4A)
    return __dp4a(a, b, c);
#else
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}
template<int width = 32>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

// ============================================================================
// vec_dot_q1_0_g128_q8_1 — exact copy from llama.cpp vecdotq.cuh
// ============================================================================

static __device__ __forceinline__ float vec_dot_q1_0_g128_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_q1_0_g128 * bq1_0_g128 = (const block_q1_0_g128 *) vbq + kbx;

    const float d1 = bq1_0_g128->d;

    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    const int offset = iqs * 4;
    const int v = bq1_0_g128->qs[offset + 0] | (bq1_0_g128->qs[offset + 1] << 8) |
                  (bq1_0_g128->qs[offset + 2] << 16) | (bq1_0_g128->qs[offset + 3] << 24);

    int vi_bytes[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int shift = j * 4;
        const int bits4 = (v >> shift) & 0x0F;
        const int b0 = (bits4 & 0x01) ? 1 : -1;
        const int b1 = (bits4 & 0x02) ? 1 : -1;
        const int b2 = (bits4 & 0x04) ? 1 : -1;
        const int b3 = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int u = get_int_b4(bq8_1_chunk->qs, j);
        sumi = ggml_cuda_dp4a(vi_bytes[j], u, sumi);
    }

    const float2 ds8f = __half22float2(bq8_1_chunk->ds);
    return d1 * ds8f.x * sumi;
}

// ============================================================================
// Simplified MMVQ kernel — from llama.cpp mmvq.cu
// ncols_dst=1, rows_per_cuda_block=1, no fusion, no multi-token-id
// ============================================================================

constexpr int NWARPS = 4;
constexpr int WARP_SZ = 32;
constexpr int QI = QK1_0_g128 / 32;  // = 4
constexpr int VDR = 1;
constexpr int BLOCKS_PER_ITER = VDR * NWARPS * WARP_SZ / QI;  // = 32

__launch_bounds__(NWARPS * WARP_SZ, 1)
__global__ void mmvq_q1_0_g128(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols_x,
        const int stride_row_x) {

    const int tid = WARP_SZ * threadIdx.y + threadIdx.x;
    const int row0 = blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK1_0_g128;

    const block_q8_1 * y = (const block_q8_1 *) vy;
    const int kbx_offset = row0 * stride_row_x;

    float tmp = 0.0f;

    for (int kbx = tid / (QI/VDR); kbx < blocks_per_row_x; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK1_0_g128 / QK8_1);  // = kbx * 4
        const int kqs = VDR * (tid % (QI/VDR));

        tmp += vec_dot_q1_0_g128_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
    }

    // Cross-warp reduction via shared memory
    __shared__ float tmp_shared[NWARPS - 1][WARP_SZ];

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

#pragma unroll
    for (int l = 0; l < NWARPS - 1; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row0] = tmp;
    }
}

// ============================================================================
// Layer descriptor
// ============================================================================

struct Layer {
    std::string name;
    int nrows;      // output dim (= number of CUDA blocks)
    int ncols;      // input dim (= QK1_0_g128 blocks along this axis)
    int count;      // how many times per token (e.g. 32 for 32 transformer blocks)
};

// ============================================================================
// Helpers
// ============================================================================

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static void fill_random(void *buf, size_t bytes) {
    auto *p = (uint8_t *)buf;
    for (size_t i = 0; i < bytes; i++) p[i] = rand() & 0xFF;
}

static std::vector<Layer> parse_layers(const char *csv) {
    std::vector<Layer> layers;
    std::istringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        Layer l;
        // format: name:nrows:ncols:count
        char name[64];
        sscanf(token.c_str(), "%63[^:]:%d:%d:%d", name, &l.nrows, &l.ncols, &l.count);
        l.name = name;
        layers.push_back(l);
    }
    return layers;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <gpu_id> <n_warmup> <n_reps> <layers_csv>\n", argv[0]);
        return 1;
    }

    int gpu_id = atoi(argv[1]);
    int n_warmup = atoi(argv[2]);
    int n_reps = atoi(argv[3]);
    auto layers = parse_layers(argv[4]);

    CHECK_CUDA(cudaSetDevice(gpu_id));

    // Print GPU info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, gpu_id));
    printf("GPU,%s,sm_%d%d,%d,%d\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.l2CacheSize / 1024);
    fflush(stdout);

    fprintf(stderr, "[mmvq_bench] GPU %d: %s (%d SMs, L2=%d KB)\n",
            gpu_id, prop.name, prop.multiProcessorCount, prop.l2CacheSize / 1024);

    // Find max dimensions across all layers for allocation
    size_t max_weight_bytes = 0;
    int max_ncols = 0;
    int max_nrows = 0;
    for (auto &l : layers) {
        // Weight: nrows × (ncols/128) blocks × 18 bytes each
        size_t w = (size_t)l.nrows * (l.ncols / QK1_0_g128) * sizeof(block_q1_0_g128);
        if (w > max_weight_bytes) max_weight_bytes = w;
        if (l.ncols > max_ncols) max_ncols = l.ncols;
        if (l.nrows > max_nrows) max_nrows = l.nrows;
    }
    // Activation: (max_ncols/32) Q8_1 blocks
    size_t act_bytes = (max_ncols / QK8_1) * sizeof(block_q8_1);
    // Output buffer
    size_t out_bytes = max_nrows * sizeof(float);

    // Allocate device buffers
    void *d_weight, *d_act;
    float *d_out;
    CHECK_CUDA(cudaMalloc(&d_weight, max_weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_act, act_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, out_bytes));

    // Fill with random data on host, copy to device
    void *h_weight = malloc(max_weight_bytes);
    void *h_act = malloc(act_bytes);
    fill_random(h_weight, max_weight_bytes);
    fill_random(h_act, act_bytes);
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight, max_weight_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_act, h_act, act_bytes, cudaMemcpyHostToDevice));
    free(h_weight);
    free(h_act);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    dim3 block_dims(WARP_SZ, NWARPS, 1);  // (32, 4, 1) = 128 threads

    double total_us_per_token = 0.0;
    double total_weight_mb = 0.0;

    for (auto &l : layers) {
        int blocks_per_row = l.ncols / QK1_0_g128;  // stride_row_x
        dim3 grid(l.nrows, 1, 1);
        int n_calls = l.count;
        double weight_bytes = (double)l.nrows * blocks_per_row * sizeof(block_q1_0_g128);
        double weight_mb = weight_bytes / (1024.0 * 1024.0);

        // Copy fresh random weights for this layer size
        CHECK_CUDA(cudaMemcpy(d_weight, d_weight, (size_t)(weight_bytes), cudaMemcpyDeviceToDevice));

        // Warmup
        for (int i = 0; i < n_warmup; i++) {
            mmvq_q1_0_g128<<<grid, block_dims>>>(
                d_weight, d_act, d_out, l.ncols, blocks_per_row);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Benchmark: time n_reps "tokens", each token = n_calls kernel launches
        CHECK_CUDA(cudaEventRecord(start));
        for (int rep = 0; rep < n_reps; rep++) {
            for (int c = 0; c < n_calls; c++) {
                mmvq_q1_0_g128<<<grid, block_dims>>>(
                    d_weight, d_act, d_out, l.ncols, blocks_per_row);
            }
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        double total_us = ms * 1000.0;
        double per_call_us = total_us / (n_reps * n_calls);
        double per_token_us = total_us / n_reps;
        // Effective BW: bytes read per call / time per call
        double eff_bw_gbps = (weight_bytes / per_call_us) * 1e6 / (1024.0*1024.0*1024.0);

        printf("LAYER,%s,%d,%d,%d,%.1f,%.2f,%.2f,%.2f\n",
               l.name.c_str(), l.nrows, l.ncols, n_calls,
               per_token_us, per_call_us, weight_mb, eff_bw_gbps);
        fflush(stdout);

        fprintf(stderr, "  %-12s %5dx%-5d  x%-3d  %8.1f us/tok  %7.2f us/call  %6.1f MB  %6.1f GB/s\n",
                l.name.c_str(), l.nrows, l.ncols, n_calls,
                per_token_us, per_call_us, weight_mb, eff_bw_gbps);

        total_us_per_token += per_token_us;
        total_weight_mb += weight_mb * n_calls;
    }

    double equiv_tps = 1e6 / total_us_per_token;
    printf("SUMMARY,%.1f,%.2f,%.2f\n", total_us_per_token, equiv_tps, total_weight_mb);
    fflush(stdout);
    fprintf(stderr, "\n  TOTAL: %.1f us/token = %.2f TPS (%.1f MB weights/token)\n",
            total_us_per_token, equiv_tps, total_weight_mb);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_act));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
