#pragma once
#include "../cube/simple_cube.h"
#include <cstddef>
#include <vector>
#include <thread>

namespace parallel_olap {

    // Slice: parallel version - each thread handles a portion of lat rows
    template<typename Dtype>
    SimpleCube<Dtype> slice_time(const SimpleCube<Dtype>& cube, size_t t, unsigned int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }

        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        size_t lat_per_thread = (LAT + num_threads - 1) / num_threads;

        auto worker = [&](size_t lat_start, size_t lat_end) {
            for (size_t lat = lat_start; lat < lat_end; ++lat) {
                for (size_t lon = 0; lon < LON; ++lon) {
                    result.at(0, lat, lon) = cube.at(t, lat, lon);
                }
            }
        };

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            size_t start = i * lat_per_thread;
            size_t end = std::min(start + lat_per_thread, LAT);
            if (start >= LAT) break;
            threads.emplace_back(worker, start, end);
        }

        for (auto& th : threads) {
            th.join();
        }

        return result;
    }

    // Dice: parallel version - each thread handles a portion of time slices
    template<typename Dtype>
    SimpleCube<Dtype> dice(const SimpleCube<Dtype>& cube,
                           size_t t_start, size_t t_end,
                           size_t lat_start, size_t lat_end,
                           size_t lon_start, size_t lon_end,
                           unsigned int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }

        size_t newT = t_end - t_start;
        size_t newLAT = lat_end - lat_start;
        size_t newLON = lon_end - lon_start;

        SimpleCube<Dtype> result(newT, newLAT, newLON);

        size_t time_per_thread = (newT + num_threads - 1) / num_threads;

        auto worker = [&](size_t time_offset_start, size_t time_offset_end) {
            for (size_t t = time_offset_start; t < time_offset_end; ++t) {
                for (size_t lat = 0; lat < newLAT; ++lat) {
                    for (size_t lon = 0; lon < newLON; ++lon) {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat + lat_start, lon + lon_start);
                    }
                }
            }
        };

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            size_t start = i * time_per_thread;
            size_t end = std::min(start + time_per_thread, newT);
            if (start >= newT) break;
            threads.emplace_back(worker, start, end);
        }

        for (auto& th : threads) {
            th.join();
        }

        return result;
    }

    // Dice by time range only
    template<typename Dtype>
    SimpleCube<Dtype> dice_time(const SimpleCube<Dtype>& cube,
                                size_t t_start, size_t t_end,
                                unsigned int num_threads = 0) {
        return dice(cube, t_start, t_end,
                    0, cube.lat_dim(),
                    0, cube.lon_dim(),
                    num_threads);
    }

    // Dice by region only
    template<typename Dtype>
    SimpleCube<Dtype> dice_region(const SimpleCube<Dtype>& cube,
                                  size_t lat_start, size_t lat_end,
                                  size_t lon_start, size_t lon_end,
                                  unsigned int num_threads = 0) {
        return dice(cube, 0, cube.time_dim(),
                    lat_start, lat_end,
                    lon_start, lon_end,
                    num_threads);
    }

    // Rollup: time mean - parallel over lat dimension
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean(const SimpleCube<Dtype>& cube, unsigned int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }

        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        size_t lat_per_thread = (LAT + num_threads - 1) / num_threads;

        auto worker = [&](size_t lat_start, size_t lat_end) {
            for (size_t lat = lat_start; lat < lat_end; ++lat) {
                for (size_t lon = 0; lon < LON; ++lon) {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t) {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                }
            }
        };

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            size_t start = i * lat_per_thread;
            size_t end = std::min(start + lat_per_thread, LAT);
            if (start >= LAT) break;
            threads.emplace_back(worker, start, end);
        }

        for (auto& th : threads) {
            th.join();
        }

        return result;
    }

    // Rollup: time sum - parallel over lat dimension
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_sum(const SimpleCube<Dtype>& cube, unsigned int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }

        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        size_t lat_per_thread = (LAT + num_threads - 1) / num_threads;

        auto worker = [&](size_t lat_start, size_t lat_end) {
            for (size_t lat = lat_start; lat < lat_end; ++lat) {
                for (size_t lon = 0; lon < LON; ++lon) {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t) {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum;
                }
            }
        };

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            size_t start = i * lat_per_thread;
            size_t end = std::min(start + lat_per_thread, LAT);
            if (start >= LAT) break;
            threads.emplace_back(worker, start, end);
        }

        for (auto& th : threads) {
            th.join();
        }

        return result;
    }

    // Global mean - parallel over time dimension
    template<typename Dtype>
    Dtype global_mean(const SimpleCube<Dtype>& cube, unsigned int num_threads = 0) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;
        }

        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        std::vector<Dtype> partial_sums(num_threads, 0);

        size_t time_per_thread = (T_dim + num_threads - 1) / num_threads;

        auto worker = [&](size_t t_start, size_t t_end, size_t thread_id) {
            Dtype sum = 0;
            for (size_t t = t_start; t < t_end; ++t) {
                for (size_t lat = 0; lat < LAT; ++lat) {
                    for (size_t lon = 0; lon < LON; ++lon) {
                        sum += cube.at(t, lat, lon);
                    }
                }
            }
            partial_sums[thread_id] = sum;
        };

        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            size_t start = i * time_per_thread;
            size_t end = std::min(start + time_per_thread, T_dim);
            if (start >= T_dim) break;
            threads.emplace_back(worker, start, end, i);
        }

        for (auto& th : threads) {
            th.join();
        }

        Dtype total_sum = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            total_sum += partial_sums[i];
        }

        return total_sum / static_cast<Dtype>(T_dim * LAT * LON);
    }

} // namespace parallel_olap
