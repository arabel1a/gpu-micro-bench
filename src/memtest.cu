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

// Cache line stride: 128 bytes / sizeof(int) = 32 ints.
// Chain nodes are spaced 1 cache line apart so every pointer-chase hop
// is a guaranteed cache line miss.
static constexpr int CL_STRIDE = 128 / sizeof(int);  // 32

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
    int idx = (blockIdx.x % 1024) * CL_STRIDE;
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

// Block dispatch overhead test.
// 128 threads (4 warps × 32), matches MMVQ launch_bounds.
// Reads a small buffer that fits in L1 (4 KB), does minimal compute,
// reduces via shared memory + __syncthreads__ + warp shuffle.
// Structure mirrors the real MMVQ kernel: load → compute → smem reduce → write.
//
// Anti-optimization measures:
// - All loads via inline PTX (ld.global.ca) — compiler can't hoist or eliminate
// - Data-dependent addressing: each load index depends on the previous load value
// - blockIdx.x mixed into address to prevent cross-block CSE
// - Shared memory values are data-dependent (not predictable at compile time)
// - Final result escapes to global memory
__launch_bounds__(128, 1)
__global__ void block_overhead_kernel(const int* __restrict__ data,
                                      float* __restrict__ out,
                                      int data_elems) {
    __shared__ float smem[3][32];

    int tid = threadIdx.y * 32 + threadIdx.x;

    // Load from L1-cached buffer with data-dependent addressing.
    // Start index depends on blockIdx to prevent CSE across blocks.
    int idx = (tid + (blockIdx.x & 0xF)) % data_elems;
    int acc = 0;
    for (int i = 0; i < 8; ++i) {
        int val = ld_ca(data + idx);
        acc += val;
        // Next index depends on loaded value — opaque to compiler
        idx = (unsigned)(idx + (val | 1)) % (unsigned)data_elems;
    }

    float tmp = (float)acc;

    // Cross-warp reduction via shared memory (MMVQ structure: nwarps=4)
    if (threadIdx.y > 0) {
        smem[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    for (int l = 0; l < 3; ++l) {
        tmp += smem[l][threadIdx.x];
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, offset);
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = tmp;
    }
}

// Warp concurrency: 1 block, N warps, thread 0 of each warp does pointer chase.
// Tests how well one SM hides DRAM latency by overlapping loads from different warps.
__global__ void warp_conc_chase(const int* __restrict__ chain, int n_hops,
                                int n_nodes, int* out) {
    if (threadIdx.x != 0) return;
    int warp_id = threadIdx.y;
    int idx = ((warp_id * 37) % n_nodes) * CL_STRIDE;
    for (int i = 0; i < n_hops; ++i) {
        idx = ld_cg(chain + idx);
    }
    out[warp_id] = idx;
}

// Block concurrency: N blocks, each 1 warp, thread 0 does pointer chase.
// Tests how block-level occupancy across all SMs helps hide DRAM latency.
__global__ void block_conc_chase(const int* __restrict__ chain, int n_hops,
                                 int n_nodes, int* out) {
    if (threadIdx.x != 0) return;
    int block_id = blockIdx.x;
    int idx = ((block_id * 37) % n_nodes) * CL_STRIDE;
    for (int i = 0; i < n_hops; ++i) {
        idx = ld_cg(chain + idx);
    }
    out[block_id] = idx;
}

// ============================================================
// Host helpers
// ============================================================

void build_pointer_chase(int* h_chain, size_t n_elems, int seed) {
    size_t n_nodes = n_elems / CL_STRIDE;  // one node per cache line
    std::vector<int> perm(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) perm[i] = (int)i;
    std::mt19937 rng(seed);
    for (size_t i = n_nodes - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        std::swap(perm[i], perm[dist(rng)]);
    }
    // Zero the buffer, then link nodes at cache-line-spaced positions
    memset(h_chain, 0, n_elems * sizeof(int));
    for (size_t i = 0; i < n_nodes - 1; ++i)
        h_chain[perm[i] * CL_STRIDE] = perm[i + 1] * CL_STRIDE;
    h_chain[perm[n_nodes - 1] * CL_STRIDE] = perm[0] * CL_STRIDE;
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
        sz = l2_size * 32;  // Must exceed L2 associativity (16-32 way) to evict all sets
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

// Cold (DRAM-only) latency: flush L2 before each single-pass rep.
// Chain has n_nodes = n_elems/CL_STRIDE nodes (one per cache line), so one
// cycle = n_nodes hops, each hitting a different cache line = guaranteed miss.
LatResult measure_latency_cold(const int* d_chain, size_t n_elems, int* d_out,
                               L2Flusher& flusher) {
    int n_hops = (int)(n_elems / CL_STRIDE);  // one cycle through the chain
    int n_reps = 5;

    // Warmup: populate TLBs
    pointer_chase<CG><<<1, 1>>>(d_chain, n_hops, d_out);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int r = 0; r < n_reps; ++r) {
        flusher.flush();
        CHECK(cudaEventRecord(start));
        pointer_chase<CG><<<1, 1>>>(d_chain, n_hops, d_out);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    float avg_ms = total_ms / n_reps;
    return { (double)avg_ms * 1e6 / n_hops, total_ms };
}

LatResult measure_latency_cold_multi(const int* d_chain, size_t n_elems, int* d_out,
                                     int n_sms, L2Flusher& flusher) {
    int n_hops = (int)(n_elems / CL_STRIDE);
    int n_reps = 5;

    pointer_chase_multi<CG><<<n_sms, 32>>>(d_chain, n_hops, d_out);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int r = 0; r < n_reps; ++r) {
        flusher.flush();
        CHECK(cudaEventRecord(start));
        pointer_chase_multi<CG><<<n_sms, 32>>>(d_chain, n_hops, d_out);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    float avg_ms = total_ms / n_reps;
    return { (double)avg_ms * 1e6 / n_hops, total_ms };
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

BWResult measure_seq_bw_cold(const int* d_data, size_t n_elems, int* d_out,
                             size_t bytes, int n_blocks, int n_threads,
                             L2Flusher& flusher) {
    int n_reps = 5;

    // Warmup
    seq_bw_kernel<CG><<<n_blocks, n_threads>>>(d_data, n_elems, d_out);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int r = 0; r < n_reps; ++r) {
        flusher.flush();
        CHECK(cudaEventRecord(start));
        seq_bw_kernel<CG><<<n_blocks, n_threads>>>(d_data, n_elems, d_out);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    float avg_ms = total_ms / n_reps;
    return { (double)bytes / ((double)avg_ms / 1000.0) / 1e9, total_ms };
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
        size_t n_nodes = n_elems / CL_STRIDE;
        int n_hops = std::max((int)(n_nodes * 4), n_hops_min);
        int* dc = chains.d_chains[i];

        auto ca = measure_latency<CA>(dc, n_hops, d_out, flusher, false);
        auto cg = measure_latency<CG>(dc, n_hops, d_out, flusher, false);
        auto cs = measure_latency<CS>(dc, n_hops, d_out, flusher, false);

        if (sizes[i] >= 64 * 1024) {
            auto cn = measure_latency_cold(dc, n_elems, d_out, flusher);
            printf("LAT_WARM,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
                   sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop, cn.ns_per_hop,
                   ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        } else {
            printf("LAT_WARM,%zu,%.2f,%.2f,%.2f,,%.1f\n",
                   sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop,
                   ca.wall_ms + cg.wall_ms + cs.wall_ms);
        }
        fflush(stdout);
    }

    // Test 2: Multi-SM pointer-chase
    for (size_t i = 0; i < sizes.size(); ++i) {
        int* dc = chains.d_chains[i];
        int n_hops = 20000;

        auto ca = measure_latency_multi<CA>(dc, n_hops, d_out, n_sms, flusher, false);
        auto cg = measure_latency_multi<CG>(dc, n_hops, d_out, n_sms, flusher, false);
        auto cs = measure_latency_multi<CS>(dc, n_hops, d_out, n_sms, flusher, false);

        if (sizes[i] >= 64 * 1024) {
            auto cn = measure_latency_cold_multi(dc, sizes[i] / sizeof(int), d_out, n_sms, flusher);
            printf("LAT_MULTI,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
                   sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop, cn.ns_per_hop,
                   ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        } else {
            printf("LAT_MULTI,%zu,%.2f,%.2f,%.2f,,%.1f\n",
                   sizes[i], ca.ns_per_hop, cg.ns_per_hop, cs.ns_per_hop,
                   ca.wall_ms + cg.wall_ms + cs.wall_ms);
        }
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
        auto cn = measure_seq_bw_cold(d_data, n_elems, d_out, sizes[i],
                                     n_blocks, n_threads, flusher);

        printf("BW_WARM,%zu,%.2f,%.2f,%.2f,%.2f,%.1f\n",
               sizes[i], ca.gb_per_s, cg.gb_per_s, cs.gb_per_s, cn.gb_per_s,
               ca.wall_ms + cg.wall_ms + cs.wall_ms + cn.wall_ms);
        fflush(stdout);
    }

    // ============================================================
    // Test 4: Block dispatch overhead
    // ============================================================
    // Small buffer that fits in L1 (4 KB). Warm it, then run the
    // same fixed-work kernel with increasing block counts.
    // If per-block overhead differs between GPUs, the CMP/2080S
    // time ratio will grow with block count.
    {
        constexpr int BLK_DATA_ELEMS = 1024;  // 4 KB
        constexpr size_t BLK_DATA_BYTES = BLK_DATA_ELEMS * sizeof(int);
        int* h_bdata = (int*)malloc(BLK_DATA_BYTES);
        for (int i = 0; i < BLK_DATA_ELEMS; i++) h_bdata[i] = (i * 7 + 13) & 0xFF;
        int* d_bdata;
        CHECK(cudaMalloc(&d_bdata, BLK_DATA_BYTES));
        CHECK(cudaMemcpy(d_bdata, h_bdata, BLK_DATA_BYTES, cudaMemcpyHostToDevice));
        free(h_bdata);

        int max_blocks = n_sms * 64;
        float* d_bout;
        CHECK(cudaMalloc(&d_bout, sizeof(float) * max_blocks));

        dim3 bk_dim(32, 4, 1);  // 128 threads = 4 warps × 32

        // Warmup L1 with max_blocks launch
        block_overhead_kernel<<<max_blocks, bk_dim>>>(d_bdata, d_bout, BLK_DATA_ELEMS);
        CHECK(cudaDeviceSynchronize());

        fprintf(stderr, "Running block dispatch overhead tests...\n");

        std::vector<int> bcounts;
        bcounts.push_back(n_sms);       // 1 block/SM
        for (int m = 2; m <= 64; m *= 2)
            bcounts.push_back(n_sms * m);

        constexpr int BLK_REPS = 50;
        for (int nb : bcounts) {
            // Warmup this exact config
            block_overhead_kernel<<<nb, bk_dim>>>(d_bdata, d_bout, BLK_DATA_ELEMS);
            CHECK(cudaDeviceSynchronize());

            cudaEvent_t t0, t1;
            CHECK(cudaEventCreate(&t0));
            CHECK(cudaEventCreate(&t1));
            CHECK(cudaEventRecord(t0));
            for (int r = 0; r < BLK_REPS; ++r)
                block_overhead_kernel<<<nb, bk_dim>>>(d_bdata, d_bout, BLK_DATA_ELEMS);
            CHECK(cudaEventRecord(t1));
            CHECK(cudaEventSynchronize(t1));

            float ms;
            CHECK(cudaEventElapsedTime(&ms, t0, t1));
            double us_total = ms * 1000.0 / BLK_REPS;
            int bps = nb / n_sms;
            double us_per_block = us_total / bps;  // per SM, per block

            printf("BLOCK_COST,%d,%d,%.3f,%.3f,%.1f\n",
                   nb, bps, us_total, us_per_block, ms);
            fflush(stdout);

            CHECK(cudaEventDestroy(t0));
            CHECK(cudaEventDestroy(t1));
        }

        CHECK(cudaFree(d_bdata));
        CHECK(cudaFree(d_bout));
    }

    // ============================================================
    // Test 5: Warp concurrency — single SM, vary warps
    // ============================================================
    // Allocate a 64 MB chain for concurrency tests (>> any L2)
    constexpr size_t CONC_BUF = 64 * 1024 * 1024;
    size_t conc_n_elems = CONC_BUF / sizeof(int);
    size_t conc_n_nodes = conc_n_elems / CL_STRIDE;
    int* h_conc = (int*)malloc(CONC_BUF);
    build_pointer_chase(h_conc, conc_n_elems, 12345);
    int* d_conc;
    CHECK(cudaMalloc(&d_conc, CONC_BUF));
    CHECK(cudaMemcpy(d_conc, h_conc, CONC_BUF, cudaMemcpyHostToDevice));
    free(h_conc);

    // Need enough output slots for max blocks
    int max_conc_out = 16 * n_sms;
    int* d_conc_out;
    CHECK(cudaMalloc(&d_conc_out, sizeof(int) * std::max(max_conc_out, 32)));

    // Note: all concurrency tests flush L2 before timed run to ensure DRAM misses.
    // Without flush, warmup populates L2 and varying n_warps just varies the
    // working set relative to L2 capacity — measuring cache effects, not scheduling.

    fprintf(stderr, "Running warp concurrency tests...\n");
    int warp_counts[] = {1, 2, 4, 8, 16, 32};
    for (int nw : warp_counts) {
        dim3 block(32, nw);

        // Warmup TLBs
        warp_conc_chase<<<1, block>>>(d_conc, n_hops_min, (int)conc_n_nodes, d_conc_out);
        CHECK(cudaDeviceSynchronize());

        // Flush L2 so all hops are DRAM misses
        flusher.flush();

        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        warp_conc_chase<<<1, block>>>(d_conc, n_hops_min, (int)conc_n_nodes, d_conc_out);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        // ns_per_hop: per-warp latency (wall_time / n_hops_per_warp)
        // hops_per_us: aggregate throughput (n_warps * n_hops / wall_us)
        double ns_per_hop = (double)ms * 1e6 / n_hops_min;
        double hops_per_us = (double)nw * n_hops_min / (ms * 1000.0);
        printf("CONC_WARP,%d,%.2f,%.4f,%.1f\n", nw, ns_per_hop, hops_per_us, ms);
        fflush(stdout);
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    }

    // ============================================================
    // Test 5: Block concurrency — vary blocks across GPU
    // ============================================================
    fprintf(stderr, "Running block concurrency tests...\n");
    std::vector<int> block_counts;
    for (int b = 1; b < n_sms; b *= 2) block_counts.push_back(b);
    block_counts.push_back(n_sms);
    for (int m = 2; m <= 16; m *= 2) block_counts.push_back(n_sms * m);

    for (int nb : block_counts) {
        // Warmup TLBs
        block_conc_chase<<<nb, 32>>>(d_conc, n_hops_min, (int)conc_n_nodes, d_conc_out);
        CHECK(cudaDeviceSynchronize());

        // Flush L2
        flusher.flush();

        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        block_conc_chase<<<nb, 32>>>(d_conc, n_hops_min, (int)conc_n_nodes, d_conc_out);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));

        float ms;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double ns_per_hop = (double)ms * 1e6 / n_hops_min;
        double hops_per_us = (double)nb * n_hops_min / (ms * 1000.0);
        printf("CONC_BLOCK,%d,%.2f,%.4f,%.1f\n", nb, ns_per_hop, hops_per_us, ms);
        fflush(stdout);
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    }

    // ============================================================
    // Test 6: Inter-kernel L2 persistence
    // ============================================================
    fprintf(stderr, "Running L2 persistence tests...\n");
    constexpr int N_LAUNCHES = 10;
    int bw_blocks = n_sms * 4;
    int bw_threads = 256;

    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < 64 * 1024) continue;
        size_t n_el = sizes[i] / sizeof(int);
        int* d_data = chains.d_chains[i];

        // Warmup TLBs
        seq_bw_kernel<CG><<<bw_blocks, bw_threads>>>(d_data, n_el, d_out);
        CHECK(cudaDeviceSynchronize());

        // Flush L2, then launch N_LAUNCHES kernels back-to-back
        flusher.flush();

        std::vector<cudaEvent_t> ev(N_LAUNCHES + 1);
        for (int k = 0; k <= N_LAUNCHES; ++k) CHECK(cudaEventCreate(&ev[k]));

        CHECK(cudaEventRecord(ev[0]));
        for (int k = 0; k < N_LAUNCHES; ++k) {
            seq_bw_kernel<CG><<<bw_blocks, bw_threads>>>(d_data, n_el, d_out);
            CHECK(cudaEventRecord(ev[k + 1]));
        }
        CHECK(cudaEventSynchronize(ev[N_LAUNCHES]));

        float cold_ms;
        CHECK(cudaEventElapsedTime(&cold_ms, ev[0], ev[1]));
        float warm_total = 0;
        for (int k = 1; k < N_LAUNCHES; ++k) {
            float ms;
            CHECK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
            warm_total += ms;
        }
        float warm_avg_ms = warm_total / (N_LAUNCHES - 1);

        double cold_bw = (double)sizes[i] / ((double)cold_ms / 1000.0) / 1e9;
        double warm_bw = (double)sizes[i] / ((double)warm_avg_ms / 1000.0) / 1e9;

        printf("L2_PERSIST,%zu,%.2f,%.2f,%.1f\n",
               sizes[i], cold_bw, warm_bw, cold_ms + warm_total);
        fflush(stdout);

        for (int k = 0; k <= N_LAUNCHES; ++k) CHECK(cudaEventDestroy(ev[k]));
    }

    CHECK(cudaFree(d_conc));
    CHECK(cudaFree(d_conc_out));
    flusher.destroy();
    chains.free_all();
    CHECK(cudaFree(d_out));
    fprintf(stderr, "Done.\n");
    return 0;
}
