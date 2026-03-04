#pragma once
#include "../cube/simple_cube.h"
#include <random>

namespace benchmark {

inline SimpleCube<float>
generate_synthetic_cube(size_t T, size_t LAT, size_t LON)
{
    SimpleCube<float> cube(T, LAT, LON);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t t = 0; t < T; ++t)
        for (size_t i = 0; i < LAT; ++i)
            for (size_t j = 0; j < LON; ++j)
                cube.at(t, i, j) = dist(gen);

    return cube;
}

}
