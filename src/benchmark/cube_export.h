#pragma once
#include "../cube/simple_cube.h"
#include <fstream>

namespace benchmark {

template<typename Dtype>
void export_cube_csv(const SimpleCube<Dtype>& cube,
                     const std::string& path)
{
    std::ofstream file(path);

    size_t T = cube.time_dim();
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    file << "t,lat,lon,value\n";

    for(size_t t=0;t<T;t++)
    {
        for(size_t lat=0;lat<LAT;lat++)
        {
            for(size_t lon=0;lon<LON;lon++)
            {
                file << t << ","
                     << lat << ","
                     << lon << ","
                     << cube.at(t,lat,lon)
                     << "\n";
            }
        }
    }
}

}
