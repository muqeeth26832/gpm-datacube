#pragma once

#include "../cube/simple_cube.h"
#include <vector>
#include <string>

class OMPSimpleCubeBuilder
{
public:

    static SimpleCube<float> build(
        const std::vector<float>& lat,
        const std::vector<float>& lon,
        const std::vector<float>& nsr,
        const std::vector<std::string>& timestamps,
        unsigned int num_threads = 0
    );

private:

    struct BinData
    {
        size_t t;
        size_t lat_idx;
        size_t lon_idx;
        float value;
    };
};
