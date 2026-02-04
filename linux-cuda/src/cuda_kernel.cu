#include <cuda_runtime.h>

// CUDA kernel to add two vectors
__global__ void add_vectors_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// C interface for the CUDA kernel
extern "C" void add_vectors_cuda(float* a, float* b, float* c, int n) {
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    add_vectors_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
}

