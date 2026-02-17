#include "default_cube_builder.h"
#include <unordered_map>
#include <cmath>
#include <iostream>

Datacube<float>
DefaultCubeBuilder::build(
    const std::vector<float>& lat,
    const std::vector<float>& lon,
    const std::vector<float>& nsr,
    const std::vector<std::string>& timestamps)
{
    // ---- Hardcoded defaults ----
    const double lat_min = 5.0;
    const double lat_max = 40.0;
    const double lon_min = 65.0;
    const double lon_max = 100.0;
    const double resolution = 0.25;

    size_t lat_bins =
        std::ceil((lat_max - lat_min) / resolution);
    size_t lon_bins =
        std::ceil((lon_max - lon_min) / resolution);

    // ---- Build hourly time index ----
    std::unordered_map<std::string, size_t> time_index;
    size_t time_counter = 0;

    for (const auto& ts : timestamps)
    {
        std::string hour = ts.substr(0, 13); // YYYY-MM-DDTHH

        if (time_index.find(hour) == time_index.end())
        {
            time_index[hour] = time_counter++;
        }
    }

    std::cout << "Building cube: "
              << time_counter << " × "
              << lat_bins << " × "
              << lon_bins << "\n";

    Datacube<float> cube(time_counter,
                         lat_bins,
                         lon_bins);

    Datacube<int> count(time_counter,
                        lat_bins,
                        lon_bins);

    cube.fill(0.0f);
    count.fill(0);

    // ---- Binning ----
    for (size_t i = 0; i < lat.size(); ++i)
    {
        const std::string hour = timestamps[i].substr(0, 13);
        size_t t = time_index[hour];

        size_t lat_idx =
            (lat[i] - lat_min) / resolution;

        size_t lon_idx =
            (lon[i] - lon_min) / resolution;

        if (lat_idx < lat_bins &&
            lon_idx < lon_bins)
        {
            cube.at(t, lat_idx, lon_idx) += nsr[i];
            count.at(t, lat_idx, lon_idx) += 1;
        }
    }

    // ---- Normalize ----
    for (size_t t = 0; t < time_counter; ++t)
    {
        for (size_t la = 0; la < lat_bins; ++la)
        {
            for (size_t lo = 0; lo < lon_bins; ++lo)
            {
                if (count.at(t, la, lo) > 0)
                {
                    cube.at(t, la, lo) /=
                        count.at(t, la, lo);
                }
            }
        }
    }

    return cube;
}
