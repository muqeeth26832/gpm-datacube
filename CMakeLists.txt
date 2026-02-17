cmake_minimum_required(VERSION 3.16)
project(GPMCube)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(src)

find_package(ZLIB REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Main
add_executable(gpmcube
    src/main.cpp
    src/loader/zarr_loader.cpp
    src/builder/default_cube_builder.cpp
)
target_link_libraries(gpmcube PRIVATE ZLIB::ZLIB nlohmann_json::nlohmann_json)

# Test executable
add_executable(test_datacube
    tests/test_datacube.cpp
)
target_link_libraries(test_datacube PRIVATE nlohmann_json::nlohmann_json)
