%%writefile triangular_fold_bench.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define FOLD_TILE   32
#define NUM_WARMUP   3
#define NUM_TIMED   10

__global__ void kernel_diagonal_fold(const float* __restrict__ A, float* __restrict__ C, int N)
{
    if (blockIdx.x > blockIdx.y) return;

    __shared__ float tile_A[FOLD_TILE][FOLD_TILE + 1];
    __shared__ float tile_B[FOLD_TILE][FOLD_TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;

    int gy_L = blockIdx.y * FOLD_TILE + ty;
    int gx_L = blockIdx.x * FOLD_TILE + tx;

    int gy_U = blockIdx.x * FOLD_TILE + ty;
    int gx_U = blockIdx.y * FOLD_TILE + tx;

    tile_A[ty][tx] = (gy_L < N && gx_L < N) ? A[gy_L * N + gx_L] : 0.f;
    tile_B[ty][tx] = (gy_U < N && gx_U < N) ? A[gy_U * N + gx_U] : 0.f;
    __syncthreads();

    if (gy_L >= N || gx_L >= N) return;

    float sum = tile_A[ty][tx] + tile_B[tx][ty];
    C[gy_L * N + gx_L] = sum;

    if (blockIdx.x != blockIdx.y && gx_L < N && gy_L < N)
        C[gx_L * N + gy_L] = sum;
}

float bench_fold(const float* d_A, float* d_C, int N) {
    dim3 blk(FOLD_TILE, FOLD_TILE);
    dim3 grd((N + FOLD_TILE - 1) / FOLD_TILE, (N + FOLD_TILE - 1) / FOLD_TILE);

    for (int w = 0; w < NUM_WARMUP; w++) {
        kernel_diagonal_fold<<<grd, blk>>>(d_A, d_C, N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int r = 0; r < NUM_TIMED; r++) {
        kernel_diagonal_fold<<<grd, blk>>>(d_A, d_C, N);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / NUM_TIMED;
}

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int ns = 6;

    printf("Final Results: CUDA DIAGONAL FOLD\n");
    printf("%-11s %-11s %-17s %-17s\n", "Matrix Size", "Memory (MB)", "Mean Runtime (ms)", "Throughput (GB/s)");

    for (int s = 0; s < ns; s++) {
        int N = sizes[s];
        size_t matrix_bytes = (size_t)N * N * 4;
        double memory_mb = (double)matrix_bytes / (1024.0 * 1024.0);

        float *d_A, *d_C;
        cudaMalloc(&d_A, matrix_bytes); 
        cudaMalloc(&d_C, matrix_bytes);

        float fold_ms = bench_fold(d_A, d_C, N);
        
        double bytes_transferred = 3.0 * matrix_bytes; 
        double throughput_gb_s = (bytes_transferred / (fold_ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char size_str[20];
        sprintf(size_str, "%d x %d", N, N);
        printf("%-11s %-11.2f %-17.4f %-17.2f\n", size_str, memory_mb, fold_ms, throughput_gb_s);

        cudaFree(d_A); cudaFree(d_C);
    }
    return 0;



# Compile the custom targets
!nvcc -O3 -arch=sm_75 -o fold_bench triangular_fold_bench.cu

# Execute to view results
!./fold_bench

import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# 1. Capture the console stdout stream from the fold binary
print("Extracting running metrics from the Diagonal Fold binary...")
fold_out = subprocess.run(["./fold_bench"], capture_output=True, text=True, check=True).stdout

# 2. Parse the tabular text layout into a DataFrame
def parse_cuda_metrics(raw_text):
    lines = raw_text.strip().split("\n")
    data = []
    for line in lines:
        if "Final Results" in line or "Matrix Size" in line or "===" in line or not line.strip():
            continue
        tokens = line.split()
        if len(tokens) == 6:  # Format: [N, 'x', N, Memory, Runtime, Throughput]
            size_n = int(tokens[0])
            runtime = float(tokens[4])
            throughput = float(tokens[5])
            data.append({"Matrix Size": size_n, "Mean Runtime (ms)": runtime, "Throughput (GB/s)": throughput})
    return pd.DataFrame(data)

df_fold = parse_cuda_metrics(fold_out)

# 3. Plot 1: Runtime Growth Curve
plt.figure(figsize=(6, 4))
plt.plot(df_fold['Matrix Size'], df_fold['Mean Runtime (ms)'], marker='s', color='darkorange', linewidth=2, label='CUDA Diagonal Fold')
plt.xscale('log', base=2)
plt.xlabel('Matrix Size (N)', fontweight='bold')
plt.ylabel('Mean Runtime (ms)', fontweight='bold')
plt.title('Execution Time Scaling (Diagonal Fold)')
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('fold_runtime_chart.png', dpi=150)
plt.show()

# 4. Plot 2: Throughput Curve
plt.figure(figsize=(6, 4))
plt.plot(df_fold['Matrix Size'], df_fold['Throughput (GB/s)'], marker='s', color='teal', linewidth=2, label='CUDA Diagonal Fold')
plt.xscale('log', base=2)
plt.xlabel('Matrix Size (N)', fontweight='bold')
plt.ylabel('Throughput (GB/s)', fontweight='bold')
plt.title('Memory Throughput Curve (Diagonal Fold)')
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('fold_throughput_chart.png', dpi=150)
plt.show()
}
