#pragma once

#include "../cube/simple_cube.h"
#include <vector>
#include <string>


class SimpleCubeBuilder {
public:
    static SimpleCube<float> build(
        const std::vector<float>& lat,
        const std::vector<float>& lon,
        const std::vector<float>& nsr,
        const std::vector<std::string>& timestamps
    );
};
