import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

#Configuration
# Matrix sizes ranging from 256 to 8192 (powers of 2)
MATRIX_SIZES = [256, 512, 1024, 2048, 4096, 8192]
NUM_WARMUP = 3 #makes the system pre-allocate the memory buffers and stabilize its state.
NUM_TIMED_RUNS = 10 # this is for normalising the external noise
DTYPE = torch.float32

#Core Computation(CPU EAGER)
def transpose_add_cpu(A: torch.Tensor) -> torch.Tensor:
    """
    Computes A + A^T in eager mode on the CPU.
    """
    return A + A.T

#Benchmark Runner
def benchmark_size(N: int) -> dict:
    # 1. Allocate input matrix directly in system memory
    A = torch.randn(N, N, dtype=DTYPE)#Does a random initialisation of the matrix and uses more floating point to prevent the compiler to optimise through shortcuts
    # Calculate memory footprint of one input matrix (4 bytes per element)
    matrix_bytes = N * N * 4
    memory_mb = matrix_bytes / (1024 ** 2)#Conversion to MB

    # 2. Warm-up phase (allows memory pools to stabilize)
    for _ in range(NUM_WARMUP):
        _ = transpose_add_cpu(A)

    # 3. Timed execution phase
    t_start = time.perf_counter()
    for _ in range(NUM_TIMED_RUNS):
        _ = transpose_add_cpu(A)
    t_end = time.perf_counter()

    # Calculate average time per execution pass in milliseconds
    mean_ms = ((t_end - t_start) / NUM_TIMED_RUNS) * 1_000

    # Calculate memory throughput bandwidth (GB/s):
    # Kernel reads A, reads A^T, and writes the output tensor -> 3x data traffic
    bytes_transferred = 3 * matrix_bytes# we know the time but we need to calulate the speed which requires knowing the size of these 3 memory aceses
    throughput_gb_s = (bytes_transferred / (mean_ms / 1_000)) / (1024 ** 3)#in GB

    return {
        "Matrix Size": f"{N} x {N}",
        "Memory (MB)": round(memory_mb, 2),
        "Mean Runtime (ms)": round(mean_ms, 4),
        "Throughput (GB/s)": round(throughput_gb_s, 2)
    }

#Main Execution
def main():
    results = []
    for N in MATRIX_SIZES:
        results.append(benchmark_size(N))#testing for all sizes
        
    # Output an neatly formatted layout for easy data plotting and exporting
    df = pd.DataFrame(results)
    print("Final Results:PYTORCH CPU EAGER")
    print(df.to_string(index=False))
    

    # 1. Clean the data types (converts strings to numbers so they plot correctly)
    df['Matrix Size'] = df['Matrix Size'].str.split(' ').str[0].astype(int)
    df['Mean Runtime (ms)'] = df['Mean Runtime (ms)'].astype(float)
    df['Throughput (GB/s)'] = df['Throughput (GB/s)'].astype(float)

    # 2. Plot 1: Runtime Growth Curve
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Mean Runtime (ms)'], marker='o', color='crimson')
    plt.xscale('log', base=2)  # Helps spread out the sizes (256 to 8192) evenly
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Mean Runtime (ms)')
    plt.title('Execution Time Scaling')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('runtime_chart.png')

    # 3. Plot 2: Throughput Curve (Cache & TLB Effects)
    plt.figure(figsize=(6, 4))
    plt.plot(df['Matrix Size'], df['Throughput (GB/s)'], marker='s', color='royalblue')
    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Throughput (GB/s)')
    plt.title('Memory Throughput Curve')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('throughput_chart.png')
    plt.show()

if __name__ == "__main__":
    main()
