cmake_minimum_required(VERSION 3.26)

function(split_version str MAJOR MINOR PATCH)
    # Parse the string directly into the names provided by the arguments
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ ${str})
    
    set(${MAJOR} ${CMAKE_MATCH_1})
    set(${MINOR} ${CMAKE_MATCH_2})
    set(${PATCH} ${CMAKE_MATCH_3})

    return(PROPAGATE ${MAJOR} ${MINOR} ${PATCH})
endfunction()

if(DEFINED SKBUILD_PROJECT_VERSION)
    set(HELLO_CUDA_VERSION ${SKBUILD_PROJECT_VERSION})
else()
    set(HELLO_CUDA_VERSION "0.9.0")
endif()

split_version(${HELLO_CUDA_VERSION} 
    HELLO_CUDA_VERSION_MAJOR
    HELLO_CUDA_VERSION_MINOR
    HELLO_CUDA_VERSION_PATCH)

# Enable CUDA language support
enable_language(CUDA)

project(hello_cuda VERSION ${HELLO_CUDA_VERSION} LANGUAGES CXX CUDA)

message(STATUS "HELLO_CUDA_VERSION_MAJOR: ${HELLO_CUDA_VERSION_MAJOR}")
message(STATUS "HELLO_CUDA_VERSION_MINOR: ${HELLO_CUDA_VERSION_MINOR}")
message(STATUS "HELLO_CUDA_VERSION_PATCH: ${HELLO_CUDA_VERSION_PATCH}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_FOUND}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA toolkit root: ${CUDA_TOOLKIT_ROOT_DIR}")

# Set CUDA architectures (can be overridden by cmake.args in pyproject.toml)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")
endif()

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the pybind11 module with CUDA support
pybind11_add_module(hello_cuda 
    src/hello_cuda.cpp
    src/cuda_kernel.cu
)

# Link CUDA libraries
target_link_libraries(hello_cuda PRIVATE 
    ${CUDA_LIBRARIES}
)

# Include CUDA directories
target_include_directories(hello_cuda PRIVATE
    ${CUDA_INCLUDE_DIRS}
)

# Compile definitions
target_compile_definitions(hello_cuda PUBLIC 
    HELLO_CUDA_VERSION="${HELLO_CUDA_VERSION}"
    CUDA_VERSION="${CUDA_VERSION}"
)

# Set CUDA separable compilation
set_property(TARGET hello_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(
  TARGETS hello_cuda
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
)

