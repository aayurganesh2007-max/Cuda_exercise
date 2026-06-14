Implementing a kernel which, given a matrix A, computes A + A_transpose. Evaluating the kernel with the other implementations (listed below) for matrix sizes ranging from 256 to 8192 (in powers of 2) by measuring the time taken for simply executing the computations.

•	PyTorch CPU Eager
•	PyTorch GPU (CUDA) Eager
•	PyTorch GPU (CUDA) Fused
•	Custom Fused CUDA Kernel
•	PyTorch iGPU (Intel / AMD) Eager
•	PyTorch iGPU (Intel / AMD) Fused
