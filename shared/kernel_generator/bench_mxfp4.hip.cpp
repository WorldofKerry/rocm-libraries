// Standalone MXFP4 GEMM benchmark with rotating buffers
// Build: hipcc -o bench_mxfp4 bench_mxfp4.hip.cpp --offload-arch=gfx950
// Usage: bench_mxfp4 <code_object.co> <M> <N> <K> [warmup] [iters] [rotating_mb]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CHECK(x) do { hipError_t e = (x); if (e) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1); } } while(0)

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <kernel.co> <M> <N> <K> [warmup=5] [iters=20] [rotating_mb=512]\n", argv[0]);
        return 1;
    }

    const char* co_path = argv[1];
    int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    int warmup = argc > 5 ? atoi(argv[5]) : 5;
    int iters = argc > 6 ? atoi(argv[6]) : 20;
    int rotating_mb = argc > 7 ? atoi(argv[7]) : 512;

    // Our kernel args: A(8), B(8), D(8), M(4), N(4), K(4),
    //                  scaleA(8), scaleB(8), stride_sA(4), stride_sB(4) = 60 bytes
    int mx_block = 32;
    size_t a_bytes = (size_t)M * K / 2;  // FP4: 0.5 bytes/elem
    size_t b_bytes = (size_t)N * K / 2;
    size_t d_bytes = (size_t)M * N * 2;  // FP16 output
    size_t sa_bytes = (size_t)M * (K / mx_block);
    size_t sb_bytes = (size_t)N * (K / mx_block);
    size_t total_per_set = a_bytes + b_bytes + d_bytes + sa_bytes + sb_bytes;

    // Rotating buffer: allocate multiple copies to defeat L2 cache
    int num_sets = 1;
    if (rotating_mb > 0) {
        size_t rotating_bytes = (size_t)rotating_mb * 1024 * 1024;
        num_sets = std::max(1, (int)(rotating_bytes / total_per_set));
    }
    printf("Problem: %dx%dx%d MXFP4\n", M, N, K);
    printf("Buffers: A=%zuB B=%zuB D=%zuB sA=%zuB sB=%zuB (total=%zuB)\n",
           a_bytes, b_bytes, d_bytes, sa_bytes, sb_bytes, total_per_set);
    printf("Rotating: %d sets (%zu MB)\n", num_sets, num_sets * total_per_set / (1024*1024));

    // Allocate rotating buffer sets
    struct BufSet { void *A, *B, *D, *sA, *sB; };
    std::vector<BufSet> sets(num_sets);
    for (int i = 0; i < num_sets; i++) {
        CHECK(hipMalloc(&sets[i].A, a_bytes));
        CHECK(hipMalloc(&sets[i].B, b_bytes));
        CHECK(hipMalloc(&sets[i].D, d_bytes));
        CHECK(hipMalloc(&sets[i].sA, sa_bytes));
        CHECK(hipMalloc(&sets[i].sB, sb_bytes));
        // Init: random-ish data
        CHECK(hipMemset(sets[i].A, 0x22, a_bytes));   // FP4 = 1.0
        CHECK(hipMemset(sets[i].B, 0x22, b_bytes));
        CHECK(hipMemset(sets[i].D, 0, d_bytes));
        CHECK(hipMemset(sets[i].sA, 0x7F, sa_bytes)); // E8M0 = 1.0
        CHECK(hipMemset(sets[i].sB, 0x7F, sb_bytes));
    }

    // Load kernel
    hipModule_t module;
    CHECK(hipModuleLoad(&module, co_path));
    hipFunction_t func;
    CHECK(hipModuleGetFunction(&func, module, "gemm_kernel"));

    int tile_m = 128, tile_n = 128;
    int grid_m = (M + tile_m - 1) / tile_m;
    int grid_n = (N + tile_n - 1) / tile_n;
    int block_size = 256;
    int stride_sa = K / mx_block;
    int stride_sb = K / mx_block;

    printf("Grid: %dx%d, Block: %d\n", grid_m, grid_n, block_size);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        auto& s = sets[i % num_sets];
        struct __attribute__((packed)) {
            void* A; void* B; void* D;
            int M, N, K;
            void* sA; void* sB;
            int stride_sA, stride_sB;
        } args = {s.A, s.B, s.D, M, N, K, s.sA, s.sB, stride_sa, stride_sb};
        void* config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, (void*)sizeof(args),
            HIP_LAUNCH_PARAM_END
        };
        // Need to cast sizeof to void* properly
        size_t sz = sizeof(args);
        void* config2[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
            HIP_LAUNCH_PARAM_END
        };
        CHECK(hipModuleLaunchKernel(func,
            grid_m, grid_n, 1,
            block_size, 1, 1,
            0, nullptr, nullptr, (void**)config2));
    }
    CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t start, stop;
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));

    CHECK(hipEventRecord(start, nullptr));
    for (int i = 0; i < iters; i++) {
        auto& s = sets[i % num_sets];
        struct __attribute__((packed)) {
            void* A; void* B; void* D;
            int M, N, K;
            void* sA; void* sB;
            int stride_sA, stride_sB;
        } args = {s.A, s.B, s.D, M, N, K, s.sA, s.sB, stride_sa, stride_sb};
        size_t sz = sizeof(args);
        void* config[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
            HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
            HIP_LAUNCH_PARAM_END
        };
        CHECK(hipModuleLaunchKernel(func,
            grid_m, grid_n, 1,
            block_size, 1, 1,
            0, nullptr, nullptr, (void**)config));
    }
    CHECK(hipEventRecord(stop, nullptr));
    CHECK(hipEventSynchronize(stop));

    float elapsed_ms;
    CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_us = elapsed_ms * 1000.0 / iters;
    double tflops = 2.0 * M * N * K / (avg_us * 1e6) / 1e12;

    printf("\nResults (%d iters, %d warmup, %d-set rotating buffer):\n", iters, warmup, num_sets);
    printf("  Time:   %.1f us\n", avg_us);
    printf("  TFLOPS: %.0f\n", tflops);

    // Cleanup
    for (auto& s : sets) {
        hipFree(s.A); hipFree(s.B); hipFree(s.D); hipFree(s.sA); hipFree(s.sB);
    }
    hipModuleUnload(module);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return 0;
}
