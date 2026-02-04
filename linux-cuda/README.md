# CUDA Hello World Example

This directory contains a simple CUDA-enabled Python extension built with pybind11 and CMake.

## Structure

- `CMakeLists.txt` - CMake configuration with CUDA support
- `src/hello_cuda.cpp` - Main C++ code with pybind11 bindings
- `src/cuda_kernel.cu` - CUDA kernel implementation
- `pyproject.toml` - Build configuration for scikit-build-core

## Features

- Simple hello world function that detects CUDA devices
- Vector addition using CUDA kernels
- Demonstrates CUDA integration with pybind11

## Usage

After building and installing:

```python
import hello_cuda

# Get CUDA info
print(hello_cuda.hello())

# Add vectors using CUDA
a = [1.0, 2.0, 3.0, 4.0]
b = [5.0, 6.0, 7.0, 8.0]
result = hello_cuda.add_vectors(a, b)
print(result)  # [6.0, 8.0, 10.0, 12.0]
```

## Building

This is configured to build with cibuildwheel in the GitHub Actions workflow (`.github/workflows/linux-cuda.yml`).

