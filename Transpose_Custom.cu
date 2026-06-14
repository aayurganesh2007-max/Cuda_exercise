#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define FOLD_TILE   32
#define NUM_WARMUP   3
#define NUM_TIMED   10

__global__ void kernel_perfect_coalesced_fold(const float* __restrict__ A, float* __restrict__ C, int N)
{
    if (blockIdx.x > blockIdx.y) return;//This ensures we are computing the tiles only for the lower triangular

    // Pad by +1 to maintain zero bank conflicts during internal transposition
    __shared__ float tile_A[FOLD_TILE][FOLD_TILE + 1];
    __shared__ float tile_B[FOLD_TILE][FOLD_TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;

    // Original symmetric read coordinates
    int gy_L = blockIdx.y * FOLD_TILE + ty;
    int gx_L = blockIdx.x * FOLD_TILE + tx;
    int gy_U = blockIdx.x * FOLD_TILE + ty;
    int gx_U = blockIdx.y * FOLD_TILE + tx;

    // 1. Fully coalesced read from Global Memory into Shared Tiles
    tile_A[ty][tx] = (gy_L < N && gx_L < N) ? A[gy_L * N + gx_L] : 0.f;
    tile_B[ty][tx] = (gy_U < N && gx_U < N) ? A[gy_U * N + gx_U] : 0.f;
    __syncthreads();

    // 2. Compute the fused sum inside registers
    float sum = tile_A[ty][tx] + tile_B[tx][ty];
    __syncthreads(); // Synchronize before reusing shared memory tracks

    // 3. Double-buffer the results back into shared memory
    // tile_A holds the normal layout, tile_B stores the transposed view locally
    tile_A[ty][tx] = sum;
    tile_B[tx][ty] = sum; 
    __syncthreads();

    // 4. Write out Target 1 (Lower Triangle) - Coalesced Row Wise
    if (gy_L < N && gx_L < N) {
        C[gy_L * N + gx_L] = tile_A[ty][tx];
    }

    // 5. Write out Target 2 (Upper Triangle) - NOW FULLY COALESCED ROW WISE
    // Notice we swap global mapping indexing matching the flipped shared memory buffer layout
    if (blockIdx.x != blockIdx.y && gy_U < N && gx_U < N) {
        C[gy_U * N + gx_U] = tile_B[ty][tx];
    }
}

float bench_perfect(const float* d_A, float* d_C, int N) {
    dim3 blk(FOLD_TILE, FOLD_TILE);
    dim3 grd((N + FOLD_TILE - 1) / FOLD_TILE, (N + FOLD_TILE - 1) / FOLD_TILE);

    for (int w = 0; w < NUM_WARMUP; w++) {
        kernel_perfect_coalesced_fold<<<grd, blk>>>(d_A, d_C, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int r = 0; r < NUM_TIMED; r++) {
        kernel_perfect_coalesced_fold<<<grd, blk>>>(d_A, d_C, N);
    }

    cudaEventRecord(t1); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / NUM_TIMED;
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int ns = 6;

    printf("Final Results: CUDA PERFECT COALESCED FOLD\n");
    printf("%-11s %-11s %-17s %-17s\n", "Matrix Size", "Memory (MB)", "Mean Runtime (ms)", "Throughput (GB/s)");

    for (int s = 0; s < ns; s++) {
        int N = sizes[s];
        size_t matrix_bytes = (size_t)N * N * 4;
        double memory_mb = (double)matrix_bytes / (1024.0 * 1024.0);

        float *d_A, *d_C;
        cudaMalloc(&d_A, matrix_bytes); cudaMalloc(&d_C, matrix_bytes);

        float perf_ms = bench_perfect(d_A, d_C, N);
        double bytes_transferred = 3.0 * matrix_bytes; 
        double throughput_gb_s = (bytes_transferred / (perf_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char size_str[20]; sprintf(size_str, "%d x %d", N, N);
        printf("%-11s %-11.2f %-17.4f %-17.2f\n", size_str, memory_mb, perf_ms, throughput_gb_s);

        cudaFree(d_A); cudaFree(d_C);
    }
    return 0;
}
