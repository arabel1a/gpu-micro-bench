// GPU memory subsystem microbenchmark — memtest style
// Measures latency per cache level using pointer-chase with PTX cache hints.
//
// Usage: ./memtest <gpu_id> [size1,size2,...] [n_hops_min]
//
// Output: tagged CSV on stdout. Columns: ca, cg, cs, cn (no-cache = L2 flushed).
//   GPU,<name>,<sm>,<n_sms>,<l2_kb>
//   LAT_WARM,<size>,<ca>,<cg>,<cs>,<cn>,<wall_ms>
//   LAT_MULTI,<size>,<ca>,<cg>,<cs>,<cn>,<wall_ms>
//   BW_WARM,<size>,<ca>,<cg>,<cs>,<cn>,<wall_ms>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
#include <omp.h>

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,       \
                    __LINE__, cudaGetErrorString(e));                   \
            exit(1);                                                   \
        }                                                              \
    } while (0)

// ============================================================
// PTX load intrinsics
// ============================================================

__device__ __forceinline__ int ld_ca(const int* addr) {
    int val;
    asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ int ld_cg(const int* addr) {
    int val;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

__device__ __forceinline__ int ld_cs(const int* addr) {
    int val;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(val) : "l"(addr));
    return val;
}

// ============================================================
// Kernels
// ============================================================

enum CacheHint { CA, CG, CS };

template <CacheHint hint>
__global__ void pointer_chase(const int* __restrict__ chain, int n_hops, int* out) {
    int idx = 0;
    for (int i = 0; i < n_hops; ++i) {
        const int* addr = chain + idx;
        if constexpr (hint == CA) idx = ld_ca(addr);
        else if constexpr (hint == CG) idx = ld_cg(addr);
        else                           idx = ld_cs(addr);
    }
    *out = idx;
}

template <CacheHint hint>
__global__ void pointer_chase_multi(const int* __restrict__ chain,
                                    int n_hops, int* out) {
    if (threadIdx.x != 0) return;
    int idx = blockIdx.x % 1024;
    for (int i = 0; i < n_hops; ++i) {
        const int* addr = chain + idx;
        if constexpr (hint == CA) idx = ld_ca(addr);
        else if constexpr (hint == CG) idx = ld_cg(addr);
        else                           idx = ld_cs(addr);
    }
    out[blockIdx.x] = idx;
}

template <CacheHint hint>
__global__ void seq_bw_kernel(const int* __restrict__ data,
                              size_t n_elems, int* out) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    int acc = 0;
    for (size_t i = tid; i < n_elems; i += stride) {
        const int* addr = data + i;
        if constexpr (hint == CA) acc += ld_ca(addr);
        else if constexpr (hint == CG) acc += ld_cg(addr);
        else                           acc += ld_cs(addr);
    }
    if (tid == 0) *out = acc;
}

// L2 flush kernel: read the entire buffer, accumulate, write result.
// Compiler cannot optimize this out because the result escapes to global memory.
__global__ void flush_read_kernel(const int* __restrict__ buf, size_t n_elems, int* out) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    int acc = 0;
    for (size_t i = tid; i < n_elems; i += stride)
        acc += buf[i];
    atomicAdd(out, acc);
}

// ============================================================
// Host helpers
// ============================================================

void build_pointer_chase(int* h_chain, size_t n_elems, int seed) {
    std::vector<int> perm(n_elems);
    for (size_t i = 0; i < n_elems; ++i) perm[i] = (int)i;
    std::mt19937 rng(seed);
    for (size_t i = n_elems - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        std::swap(perm[i], perm[dist(rng)]);
    }
    for (size_t i = 0; i < n_elems - 1; ++i)
        h_chain[perm[i]] = perm[i + 1];
    h_chain[perm[n_elems - 1]] = perm[0];
}

struct ChainSet {
    std::vector<size_t> sizes;
    std::vector<int*> d_chains;

    void build(const std::vector<size_t>& sz_list) {
        sizes = sz_list;
        d_chains.resize(sizes.size());
        std::vector<int*> h_chains(sizes.size());

        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < sizes.size(); ++i) {
            h_chains[i] = (int*)malloc(sizes[i]);
            build_pointer_chase(h_chains[i], sizes[i] / sizeof(int), 42 + (int)i);
        }

        for (size_t i = 0; i < sizes.size(); ++i) {
            CHECK(cudaMalloc(&d_chains[i], sizes[i]));
            CHECK(cudaMemcpy(d_chains[i], h_chains[i], sizes[i], cudaMemcpyHostToDevice));
            free(h_chains[i]);
        }
    }

    void free_all() {
        for (auto p : d_chains) CHECK(cudaFree(p));
    }
};

// L2 flusher: reads a buffer 4× L2 size through all SMs.
// The read result is copied back to host to prevent any DCE.
struct L2Flusher {
    int* d_buf = nullptr;
    int* d_acc = nullptr;
    size_t sz = 0;
    int n_sms = 1;

    void init(size_t l2_size, int sms) {
        n_sms = sms;
        sz = l2_size * 4;
        CHECK(cudaMalloc(&d_buf, sz));
        CHECK(cudaMalloc(&d_acc, sizeof(int)));
        // Fill with non-zero so reads are "real"
        CHECK(cudaMemset(d_buf, 0x42, sz));
    }

    void flush() {
        CHECK(cudaMemset(d_acc, 0, sizeof(int)));
        flush_read_kernel<<<n_sms * 4, 256>>>(d_buf, sz / sizeof(int), d_acc);
        // Copy result to host — the read is now observable, compiler can't skip it
        int h_acc;
        CHECK(cudaMemcpy(&h_acc, d_acc, sizeof(int), cudaMemcpyDeviceToHost));
        // h_acc is intentionally unused beyond this point; the memcpy is the fence
        (void)h_acc;
    }

    void destroy() {
        if (d_buf) CHECK(cudaFree(d_buf));
        if (d_acc) CHECK(cudaFree(d_acc));
    }
};

struct LatResult { double ns_per_hop; float wall_ms; };
struct BWResult  { double gb_per_s;   float wall_ms; };

template <CacheHint hint>
LatResult measure_latency(const int* d_chain, int n_hops, int* d_out,
                          L2Flusher& flusher, bool flush_before) {
    // Warmup: populate TLBs
    pointer_chase<hint><<<1, 1>>>(d_chain, n_hops, d_out);
    CHECK(cudaDeviceSynchronize());

    if (flush_before) flusher.flush();

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    pointer_chase<hint><<<1, 1>>>(d_chain, n_hops, d_out);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return { (double)ms * 1e6 / n_hops, ms };
}

template <CacheHint hint>
LatResult measure_latency_multi(const int* d_chain, int n_hops, int* d_out,
                                int n_sms, L2Flusher& flusher, bool flush_before) {
    pointer_chase_multi<hint><<<n_sms, 32>>>(d_chain, n_hops, d_out);
    CHECK(cudaDeviceSynchronize());

    if (flush_before) flusher.flush();

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    pointer_chase_multi<hint><<<n_sms, 32>>>(d_chain, n_hops, d_out);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return { (double)ms * 1e6 / n_hops, ms };
}

template <CacheHint hint>
BWResult measure_seq_bw(const int* d_data, size_t n_elems, int* d_out,
                        size_t bytes, int n_blocks, int n_threads,
                        L2Flusher& flusher, bool flush_before) {
    seq_bw_kernel<hint><<<n_blocks, n_threads>>>(d_data, n_elems, d_out);
    CHECK(cudaDeviceSynchronize());

    if (flush_before) flusher.flush();

    int n_reps = 10;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int r = 0; r < n_reps; ++r)
        seq_bw_kernel<hint><<<n_blocks, n_threads>>>(d_data, n_elems, d_out);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return { (double)bytes * n_reps / ((double)ms / 1000.0) / 1e9, ms };
}

// ============================================================
// CLI
// ============================================================

std::vector<size_t> parse_sizes(const char* csv) {
    std::vector<size_t> out;
    const char* p = csv;
    while (*p) {
        out.push_back(strtoull(p, nullptr, 10));
        p = strchr(p, ',');
        if (!p) break;
        ++p;
    }
    return out;
}

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    CHECK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, gpu_id));
    size_t l2_size = prop.l2CacheSize;
    int n_sms = prop.multiProcessorCount;

    std::vector<size_t> sizes;
    if (argc > 2) sizes = parse_sizes(argv[2]);
    else for (size_t s = 2*1024; s <= 256*1024*1024; s *= 2) sizes.push_back(s);

    int n_hops_min = argc > 3 ? atoi(argv[3]) : 50000;

    printf("GPU,%s,sm_%d%d,%d,%d\n",
           prop.name, prop.major, prop.minor, n_sms, (int)(l2_size / 1024));

    fprintf(stderr, "Building %zu chains with %d threads...\n",
            sizes.size(), omp_get_max_threads());

    ChainSet chains;
    chains.build(sizes);

    L2Flusher flusher;
    flusher.init(l2_size, n_sms);

    int* d_out;
    CHECK(cudaMalloc(&d_out, sizeof(int) * std::max(n_sms, 1)));

    fprintf(stderr, "Running tests...\n");

    // Test 1: Single-thread pointer-chase
    // ca/cg/cs = warm (caches populated), cn = L2 flushed before measurement
    for (size_t i = 0; i < sizes.size(); ++i) {
        size_t n_elems = sizes[i] / sizeof(int);
        int n_hops = std::max((int)(n_elems * 4), n_hops_min);
        int* dc = chains.d_chains[i];

        auto ca = measure_latency<CA>(dc, n_hops, d_out, flusher, false);
        auto cg = measure_latency<CG>(dc, n_hops, d_out, flusher, false);
        auto cs = measure_latency<CS>(dc, n_hops, d_out, flusher, false);
        auto cn = measure_latency<CA>(dc, n_hops, d_out, flusher, true);

        printf("LAT_WARM,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
               sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop, cn.ns_per_hop,
               ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        fflush(stdout);
    }

    // Test 2: Multi-SM pointer-chase
    for (size_t i = 0; i < sizes.size(); ++i) {
        int* dc = chains.d_chains[i];
        int n_hops = 20000;

        auto ca = measure_latency_multi<CA>(dc, n_hops, d_out, n_sms, flusher, false);
        auto cg = measure_latency_multi<CG>(dc, n_hops, d_out, n_sms, flusher, false);
        auto cs = measure_latency_multi<CS>(dc, n_hops, d_out, n_sms, flusher, false);
        auto cn = measure_latency_multi<CA>(dc, n_hops, d_out, n_sms, flusher, true);

        printf("LAT_MULTI,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
               sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop, cn.ns_per_hop,
               ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        fflush(stdout);
    }

    // Test 3: Sequential bandwidth
    int n_blocks = n_sms * 4;
    int n_threads = 256;

    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < 64 * 1024) continue;
        size_t n_elems = sizes[i] / sizeof(int);
        int* d_data = chains.d_chains[i];

        auto ca = measure_seq_bw<CA>(d_data, n_elems, d_out, sizes[i],
                                     n_blocks, n_threads, flusher, false);
        auto cg = measure_seq_bw<CG>(d_data, n_elems, d_out, sizes[i],
                                     n_blocks, n_threads, flusher, false);
        auto cs = measure_seq_bw<CS>(d_data, n_elems, d_out, sizes[i],
                                     n_blocks, n_threads, flusher, false);
        auto cn = measure_seq_bw<CA>(d_data, n_elems, d_out, sizes[i],
                                     n_blocks, n_threads, flusher, true);

        printf("BW_WARM,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
               sizes[i], ca.gb_per_s, cg.gb_per_s, cs.gb_per_s, cn.gb_per_s,
               ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        fflush(stdout);
    }

    flusher.destroy();
    chains.free_all();
    CHECK(cudaFree(d_out));
    fprintf(stderr, "Done.\n");
    return 0;
}
