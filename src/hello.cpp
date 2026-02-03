#include <pybind11/pybind11.h>

namespace py = pybind11;

std::string hello() {
    return std::string("v") + std::string(HELLO_CPP_VERSION) + ": Hello from C++!";
}

PYBIND11_MODULE(hello_cpp, m) {
    m.doc() = "Minimal hello world pybind11 module";
    m.def("hello", &hello, "Return a friendly greeting from C++");
}
