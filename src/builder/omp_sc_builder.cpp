#include "omp_sc_builder.h"

#include <unordered_map>
#include <cmath>
#include <iostream>
#include <omp.h>

SimpleCube<float>
OMPSimpleCubeBuilder::build(
    const std::vector<float>& lat,
    const std::vector<float>& lon,
    const std::vector<float>& nsr,
    const std::vector<std::string>& timestamps,
    unsigned int num_threads)
{
    if (num_threads == 0)
        num_threads = omp_get_max_threads();

    omp_set_num_threads(num_threads);

    const double lat_min = 5.0;
    const double lat_max = 40.0;
    const double lon_min = 65.0;
    const double lon_max = 100.0;
    const double resolution = 0.25;

    size_t lat_bins = std::ceil((lat_max - lat_min) / resolution);
    size_t lon_bins = std::ceil((lon_max - lon_min) / resolution);

    // ----------------------------
    // Build hourly time index
    // ----------------------------

    std::unordered_map<std::string,size_t> time_index;
    size_t time_counter = 0;

    for(const auto& ts : timestamps)
    {
        std::string hour = ts.substr(0,13);

        if(time_index.find(hour) == time_index.end())
            time_index[hour] = time_counter++;
    }

    std::cout << "Building OMP simple cube: "
              << time_counter << " × "
              << lat_bins << " × "
              << lon_bins << " using "
              << num_threads << " threads\n";

    SimpleCube<float> cube(time_counter,lat_bins,lon_bins);
    SimpleCube<int> count(time_counter,lat_bins,lon_bins);

    cube.fill(0.0f);
    count.fill(0);

    // ----------------------------
    // Prepare bin_data
    // ----------------------------

    std::vector<BinData> bin_data;
    bin_data.reserve(lat.size());

    double inv_res = 1.0 / resolution;

    for(size_t i = 0; i < lat.size(); i++)
    {
        const float lat_val = lat[i];
        const float lon_val = lon[i];
        const float v       = nsr[i];

        std::string_view hour(timestamps[i].data(),13);

        auto it = time_index.find(std::string(hour));
        size_t t = it->second;

        size_t lat_idx = (lat_val - lat_min) * inv_res;
        size_t lon_idx = (lon_val - lon_min) * inv_res;

        if(lat_idx < lat_bins && lon_idx < lon_bins && v > -9000)
        {
            bin_data.push_back({t,lat_idx,lon_idx,v});
        }
    }

    // ----------------------------
    // Parallel binning
    // ----------------------------

#pragma omp parallel for schedule(static)
    for(size_t i=0;i<bin_data.size();i++)
    {
        const auto& bd = bin_data[i];

#pragma omp atomic
        cube.at(bd.t,bd.lat_idx,bd.lon_idx) += bd.value;

#pragma omp atomic
        count.at(bd.t,bd.lat_idx,bd.lon_idx) += 1;
    }

    // ----------------------------
    // Normalization
    // ----------------------------

#pragma omp parallel for collapse(3) schedule(static)
    for(size_t t=0;t<time_counter;t++)
    for(size_t lat=0;lat<lat_bins;lat++)
    for(size_t lon=0;lon<lon_bins;lon++)
    {
        int c = count.at(t,lat,lon);

        if(c > 0)
            cube.at(t,lat,lon) /= c;
    }

    return cube;
}
