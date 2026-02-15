cmake_minimum_required(VERSION 3.16)
project(GPMCube)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(src)

# Main
add_executable(gpmcube
    src/main.cpp
)

# Test executable
add_executable(test_datacube
    tests/test_datacube.cpp
)
