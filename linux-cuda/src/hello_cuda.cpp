#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace py = pybind11;

// CUDA kernel declaration (defined in cuda_kernel.cu)
extern "C" void add_vectors_cuda(float* a, float* b, float* c, int n);

// Simple hello function
std::string hello() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    std::string result = "v" + std::string(HELLO_CUDA_VERSION) + ": Hello from C++ with CUDA!";
    
    if (err == cudaSuccess && device_count > 0) {
        result += "\nFound " + std::to_string(device_count) + " CUDA device(s)";
        
        // Get device info
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            result += "\nDevice 0: " + std::string(prop.name);
            result += " (Compute " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
        }
    } else {
        result += "\nCUDA devices not available";
    }
    
    return result;
}

// Python function that uses CUDA kernel
py::list add_vectors(py::list a, py::list b) {
    // Convert Python lists to vectors
    std::vector<float> vec_a, vec_b;
    for (auto item : a) {
        vec_a.push_back(py::cast<float>(item));
    }
    for (auto item : b) {
        vec_b.push_back(py::cast<float>(item));
    }
    
    if (vec_a.size() != vec_b.size()) {
        throw std::runtime_error("Vectors must have the same size");
    }
    
    int n = vec_a.size();
    if (n == 0) {
        return py::list();
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_a, vec_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, vec_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch CUDA kernel
    add_vectors_cuda(d_a, d_b, d_c, n);
    
    // Copy result back
    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Convert to Python list
    py::list py_result;
    for (float val : result) {
        py_result.append(val);
    }
    
    return py_result;
}

PYBIND11_MODULE(hello_cuda, m) {
    m.doc() = "Hello world pybind11 module with CUDA support";
    m.def("hello", &hello, "Return a friendly greeting from C++ with CUDA info");
    m.def("add_vectors", &add_vectors, "Add two vectors using CUDA",
          py::arg("a"), py::arg("b"));
}

