#include "parallel_simple_cube_builder.h"
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <mutex>


SimpleCube<float>
ParallelSimpleCubeBuilder::build(
    const std::vector<float>& lat,
    const std::vector<float>& lon,
    const std::vector<float>& nsr,
    const std::vector<std::string>& timestamps,
    unsigned int num_threads)
{
    // Auto-detect threads
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    // ---- Hardcoded defaults ----
    const double lat_min = 5.0;
    const double lat_max = 40.0;
    const double lon_min = 65.0;
    const double lon_max = 100.0;
    const double resolution = 0.25;

    size_t lat_bins = std::ceil((lat_max - lat_min) / resolution);
    size_t lon_bins = std::ceil((lon_max - lon_min) / resolution);

    // ---- Build hourly time index ----
    std::unordered_map<std::string, size_t> time_index;
    size_t time_counter = 0;

    for (const auto& ts : timestamps) {
        // std::string hour = ts.substr(0, 13);
        // if (time_index.find(hour) == time_index.end()) {
        //     time_index[hour] = time_counter++;
        // }

        std::string_view hour(ts.data(), 13);

        auto it = time_index.find(std::string(hour));
        if (it == time_index.end())
        {
            time_index.emplace(std::string(hour), time_counter++);
        }
    }

    std::cout << "Building parallel simple cube: "
              << time_counter << " × " << lat_bins << " × " << lon_bins
              << " using " << num_threads << " threads\n";

    SimpleCube<float> cube(time_counter, lat_bins, lon_bins);
    SimpleCube<int> count(time_counter, lat_bins, lon_bins);
    cube.fill(0.0f);
    count.fill(0);

    // ---- Prepare binning data ----
    std::vector<BinData> bin_data;
    bin_data.reserve(lat.size());

    for (size_t i = 0; i < lat.size(); ++i) {
        const std::string hour = timestamps[i].substr(0, 13);
        size_t t = time_index[hour];
        size_t lat_idx = (lat[i] - lat_min) / resolution;
        size_t lon_idx = (lon[i] - lon_min) / resolution;

        if (lat_idx < lat_bins && lon_idx < lon_bins) {
            float v = nsr[i];
            if (v > -9000) {
                bin_data.push_back({t, lat_idx, lon_idx, nsr[i]});
            }
        }
    }

    // ---- Parallel binning ----
    // Use mutex for thread-safe updates
    std::vector<std::mutex> time_mutexes(time_counter);

    auto bin_worker_mutex = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const auto& bd = bin_data[i];
            std::lock_guard<std::mutex> lock(time_mutexes[bd.t]);
            cube.at(bd.t, bd.lat_idx, bd.lon_idx) += bd.value;
            count.at(bd.t, bd.lat_idx, bd.lon_idx) += 1;
        }
    };

    std::vector<std::thread> threads;
    size_t data_per_thread = (bin_data.size() + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * data_per_thread;
        size_t end = std::min(start + data_per_thread, bin_data.size());
        if (start >= bin_data.size()) break;
        threads.emplace_back(bin_worker_mutex, start, end);
    }

    for (auto& th : threads) {
        th.join();
    }

    // ---- Parallel normalization ----
    std::vector<std::thread> norm_threads;
    size_t time_per_thread = (time_counter + num_threads - 1) / num_threads;

    auto normalize_worker = [&](size_t t_start, size_t t_end) {
        for (size_t t = t_start; t < t_end; ++t) {
            for (size_t la = 0; la < lat_bins; ++la) {
                for (size_t lo = 0; lo < lon_bins; ++lo) {
                    if (count.at(t, la, lo) > 0) {
                        cube.at(t, la, lo) /= count.at(t, la, lo);
                    }
                }
            }
        }
    };

    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t t_start = t * time_per_thread;
        size_t t_end = std::min(t_start + time_per_thread, time_counter);
        if (t_start >= time_counter) break;
        norm_threads.emplace_back(normalize_worker, t_start, t_end);
    }

    for (auto& th : norm_threads) {
        th.join();
    }

    return cube;
}
