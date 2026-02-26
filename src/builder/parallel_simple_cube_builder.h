#pragma once

#include "../cube/simple_cube.h"
#include <vector>
#include <string>
#include <thread>
#include <atomic>


class ParallelSimpleCubeBuilder {
public:
    static SimpleCube<float> build(
        const std::vector<float>& lat,
        const std::vector<float>& lon,
        const std::vector<float>& nsr,
        const std::vector<std::string>& timestamps,
        unsigned int num_threads = 0  // 0 = auto-detect
    );

private:
    struct BinData {
        size_t t;
        size_t lat_idx;
        size_t lon_idx;
        float value;
    };

    // Thread worker for binning
    static void bin_worker(
        const std::vector<BinData>& input_data,
        size_t start,
        size_t end,
        SimpleCube<float>& cube,
        SimpleCube<int>& count,
        size_t lat_bins,
        size_t lon_bins
    );

    // Thread worker for normalization
    static void normalize_worker(
        SimpleCube<float>& cube,
        const SimpleCube<int>& count,
        size_t t_start,
        size_t t_end,
        size_t lat_bins,
        size_t lon_bins
    );
};
