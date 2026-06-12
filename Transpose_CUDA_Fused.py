import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
MATRIX_SIZES = [256, 512, 1024, 2048, 4096, 8192]
# === MINIMAL CHANGE 1: Increase warmups to allow torch.compile to finish compilation before timing ===
NUM_WARMUP = 5  
NUM_TIMED_RUNS = 10 
DTYPE = torch.float32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running benchmarks on backend: {device}\n")

# Base operation
def transpose_add_cuda(A: torch.Tensor) -> torch.Tensor:
    return A + A.T

# === MINIMAL CHANGE 2: Compile the function to fuse operations ===
# max-autotune forces the compiler to find the absolute fastest fused memory kernel layout
transpose_add_fused = torch.compile(transpose_add_cuda, mode="max-autotune")

def benchmark_size(N: int) -> dict:
    A = torch.randn(N, N, dtype=DTYPE, device=device)
    
    matrix_bytes = N * N * 4
    memory_mb = matrix_bytes / (1024 ** 2)

    # Warm-up phase (Compilation physically occurs on the first iteration here)
    for _ in range(NUM_WARMUP):
        # === MINIMAL CHANGE 3: Call the compiled function version ===
        _ = transpose_add_fused(A)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    for _ in range(NUM_TIMED_RUNS):
        # === MINIMAL CHANGE 4: Call the compiled function version ===
        _ = transpose_add_fused(A)
        
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    mean_ms = ((t_end - t_start) / NUM_TIMED_RUNS) * 1_000

    # Note: Throughput equation left as-is for parity in your report plotting, 
    # though fusing operations reduces physical VRAM traffic!
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
    # === MINIMAL CHANGE 5: Updated print label to FUSED ===
    print("\nFinal Results: PYTORCH CUDA FUSED")
    print(df.to_string(index=False))
    
    df['Matrix Size'] = df['Matrix Size'].str.split(' ').str[0].astype(int)
    df['Mean Runtime (ms)'] = df['Mean Runtime (ms)'].astype(float)
    df['Throughput (GB/s)'] = df['Throughput (GB/s)'].astype(float)

    # Plot 1: Fused Runtime
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Mean Runtime (ms)'], marker='o', color='purple') # Color changed to purple for fused mode
    plt.xscale('log', base=2)  
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Mean Runtime (ms)')
    plt.title('CUDA Fused Execution Time Scaling')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cuda_fused_runtime_chart.png')

    # Plot 2: Fused Throughput
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Throughput (GB/s)'], marker='s', color='forestgreen') # Color changed to green
    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('CUDA Fused Memory Throughput Curve')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('cuda_fused_throughput_chart.png')
    plt.show()

if __name__ == "__main__":
    main()
