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

    // ----------------------------
    // Prepare bin_data with time info
    // ----------------------------

    double inv_res = 1.0 / resolution;

    struct IndexedBin {
        size_t t;
        size_t lat_idx;
        size_t lon_idx;
        float value;
    };

    std::vector<IndexedBin> bin_data;
    bin_data.reserve(lat.size());

    for(size_t i = 0; i < lat.size(); i++)
    {
        const float lat_val = lat[i];
        const float lon_val = lon[i];
        const float v       = nsr[i];

        std::string_view hour(timestamps[i].data(),13);
        auto it = time_index.find(std::string(hour));
        size_t t = it->second;

        size_t lat_idx = static_cast<size_t>((lat_val - lat_min) * inv_res);
        size_t lon_idx = static_cast<size_t>((lon_val - lon_min) * inv_res);

        if(lat_idx < lat_bins && lon_idx < lon_bins && v > -9000)
        {
            bin_data.push_back({t, lat_idx, lon_idx, v});
        }
    }

    // ----------------------------
    // Partition by time dimension - key optimization
    // Each thread gets exclusive time range, no atomics needed
    // ----------------------------

    SimpleCube<float> cube(time_counter, lat_bins, lon_bins);
    cube.fill(0.0f);

    SimpleCube<int> count(time_counter, lat_bins, lon_bins);
    count.fill(0);

    // Compute time ranges for each thread (contiguous T chunks for cache locality)
    std::vector<std::pair<size_t, size_t>> thread_time_ranges(num_threads);
    size_t t_per_thread = (time_counter + num_threads - 1) / num_threads;
    
    for(unsigned int tid = 0; tid < num_threads; tid++)
    {
        size_t t_start = tid * t_per_thread;
        size_t t_end   = std::min(t_start + t_per_thread, time_counter);
        thread_time_ranges[tid] = {t_start, t_end};
    }

    // ----------------------------
    // Parallel binning - time-partitioned (no atomics!)
    // ----------------------------

#pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        auto [t_start, t_end] = thread_time_ranges[tid];

        // Thread-local accumulators for assigned time range
        std::vector<std::vector<std::vector<float>>> local_sum(t_end - t_start);
        std::vector<std::vector<std::vector<int>>> local_cnt(t_end - t_start);

        for(size_t t_off = 0; t_off < t_end - t_start; t_off++)
        {
            local_sum[t_off].resize(lat_bins);
            local_cnt[t_off].resize(lat_bins);
            for(size_t lat = 0; lat < lat_bins; lat++)
            {
                local_sum[t_off][lat].resize(lon_bins, 0.0f);
                local_cnt[t_off][lat].resize(lon_bins, 0);
            }
        }

        // Build reverse index: which bins belong to this thread's time range?
        // Scan all data, accumulate into local buffers if in our time range
        for(const auto& bd : bin_data)
        {
            if(bd.t >= t_start && bd.t < t_end)
            {
                size_t t_off = bd.t - t_start;
                local_sum[t_off][bd.lat_idx][bd.lon_idx] += bd.value;
                local_cnt[t_off][bd.lat_idx][bd.lon_idx] += 1;
            }
        }

        // Merge local results into global cube (no contention - exclusive time ranges)
        for(size_t t_off = 0; t_off < t_end - t_start; t_off++)
        {
            size_t t = t_start + t_off;
            for(size_t lat = 0; lat < lat_bins; lat++)
            {
                for(size_t lon = 0; lon < lon_bins; lon++)
                {
                    int c = local_cnt[t_off][lat][lon];
                    if(c > 0)
                    {
                        cube.at(t, lat, lon) = local_sum[t_off][lat][lon] / c;
                        count.at(t, lat, lon) = c;
                    }
                }
            }
        }
    }

    return cube;
}
