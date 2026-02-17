#pragma once

#include "../cube/datacube.h"
#include <vector>


class DefaultCubeBuilder {
public:
    static Datacube<float> build(
      const std::vector<float>& lat,
      const std::vector<float>& lon,
      const std::vector<float>& lnsr,
      const std::vector<std::string>& timestamps
    );
};
