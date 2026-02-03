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
    set(HELLO_CPP_VERSION ${SKBUILD_PROJECT_VERSION})
else()
    set(HELLO_CPP_VERSION "0.9.0")
endif()

split_version(${HELLO_CPP_VERSION} 
    HELLO_CPP_VERSION_MAJOR
    HELLO_CPP_VERSION_MINOR
    HELLO_CPP_VERSION_PATCH)

project(hello_cpp VERSION ${HELLO_CPP_VERSION} LANGUAGES CXX)

message(STATUS "HELLO_CPP_VERSION_MAJOR: ${HELLO_CPP_VERSION_MAJOR}")
message(STATUS "HELLO_CPP_VERSION_MINOR: ${HELLO_CPP_VERSION_MINOR}")
message(STATUS "HELLO_CPP_VERSION_PATCH: ${HELLO_CPP_VERSION_PATCH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)


pybind11_add_module(hello_cpp src/hello.cpp)
target_compile_definitions(hello_cpp PUBLIC HELLO_CPP_VERSION="${HELLO_CPP_VERSION}")

install(
  TARGETS hello_cpp
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}
  RUNTIME DESTINATION ${SKBUILD_PLATLIB_DIR}
  ARCHIVE DESTINATION ${SKBUILD_PLATLIB_DIR}
)
