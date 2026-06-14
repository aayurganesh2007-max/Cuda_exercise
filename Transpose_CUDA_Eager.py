import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
MATRIX_SIZES = [256, 512, 1024, 2048, 4096, 8192]
NUM_WARMUP = 3 
NUM_TIMED_RUNS = 10 
DTYPE = torch.float32

# =Target Kaggle's GPU instead of CPU =
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running benchmarks on backend: {device}\n")

# Core computation (CUDA EAGER)
def transpose_add_cuda(A: torch.Tensor) -> torch.Tensor:
    """
    Computes A + A^T in eager mode on the GPU.
    """
    return A + A.T

def benchmark_size(N: int) -> dict:
    # =Allocate inp ut matrix directly on the GPU VRAM =
    A = torch.randn(N, N, dtype=DTYPE, device=device)
    
    matrix_bytes = N * N * 4
    memory_mb = matrix_bytes / (1024 ** 2)

    # Warm-up phase
    for _ in range(NUM_WARMUP):
        _ = transpose_add_cuda(A)

    # = Synchronize CUDA stream before opening the clock =
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    for _ in range(NUM_TIMED_RUNS):
        _ = transpose_add_cuda(A)
        
    # = Synchronize CUDA stream before stopping the clock =
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    mean_ms = ((t_end - t_start) / NUM_TIMED_RUNS) * 1_000

    # Memory throughput calculation (remains 3x matrix size for Eager Mode)
    bytes_transferred = 3 * matrix_bytes
    throughput_gb_s = (bytes_transferred / (mean_ms / 1_000)) / (1024 ** 3)

    return {
        "Matrix Size": f"{N} x {N}",
        "Memory (MB)": round(memory_mb, 2),
        "Mean Runtime (ms)": round(mean_ms, 4),
        "Throughput (GB/s)": round(throughput_gb_s, 2)
    }

# Main execution
def main():
    results = []
    for N in MATRIX_SIZES:
        results.append(benchmark_size(N))
        
    df = pd.DataFrame(results)
    # = Updated print label =
    print("\nFinal Results: PYTORCH CUDA EAGER")
    print(df.to_string(index=False))
    
    # 1. Clean the data types for plotting
    df['Matrix Size'] = df['Matrix Size'].str.split(' ').str[0].astype(int)
    df['Mean Runtime (ms)'] = df['Mean Runtime (ms)'].astype(float)
    df['Throughput (GB/s)'] = df['Throughput (GB/s)'].astype(float)

    # 2. Plot 1: Runtime Growth Curve
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Mean Runtime (ms)'], marker='o', color='teal') # Color changed to teal for GPU separation
    plt.xscale('log', base=2)  
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Mean Runtime (ms)')
    plt.title('CUDA Eager Execution Time Scaling')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cuda_eager_runtime_chart.png')

    # 3. Plot 2: Throughput Curve
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Throughput (GB/s)'], marker='s', color='darkorange') # Color changed to orange
    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('CUDA Eager Memory Throughput Curve')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cuda_eager_throughput_chart.png')
    plt.show()

if __name__ == "__main__":
    main()
