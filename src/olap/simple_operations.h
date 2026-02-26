#pragma once
#include "../cube/simple_cube.h"
#include <cstddef>
#include <vector>

namespace simple_olap {

    // Slice: returns a SimpleCube with single time dimension (1 x LAT x LON)
    template<typename Dtype>
    SimpleCube<Dtype> slice_time(const SimpleCube<Dtype>& cube, size_t t) {
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        for (size_t lat = 0; lat < LAT; ++lat) {
            for (size_t lon = 0; lon < LON; ++lon) {
                result.at(0, lat, lon) = cube.at(t, lat, lon);
            }
        }

        return result;
    }

    // Dice: returns a sub-cube within specified ranges
    template<typename Dtype>
    SimpleCube<Dtype> dice(const SimpleCube<Dtype>& cube,
                           size_t t_start, size_t t_end,
                           size_t lat_start, size_t lat_end,
                           size_t lon_start, size_t lon_end) {
        size_t newT = t_end - t_start;
        size_t newLAT = lat_end - lat_start;
        size_t newLON = lon_end - lon_start;

        SimpleCube<Dtype> result(newT, newLAT, newLON);

        for (size_t t = 0; t < newT; ++t) {
            for (size_t lat = 0; lat < newLAT; ++lat) {
                for (size_t lon = 0; lon < newLON; ++lon) {
                    result.at(t, lat, lon) = cube.at(t + t_start, lat + lat_start, lon + lon_start);
                }
            }
        }

        return result;
    }

    // Dice by time range only
    template<typename Dtype>
    SimpleCube<Dtype> dice_time(const SimpleCube<Dtype>& cube,
                                size_t t_start, size_t t_end) {
        return dice(cube,
                    t_start, t_end,
                    0, cube.lat_dim(),
                    0, cube.lon_dim());
    }

    // Dice by region only
    template<typename Dtype>
    SimpleCube<Dtype> dice_region(const SimpleCube<Dtype>& cube,
                                  size_t lat_start, size_t lat_end,
                                  size_t lon_start, size_t lon_end) {
        return dice(cube,
                    0, cube.time_dim(),
                    lat_start, lat_end,
                    lon_start, lon_end);
    }

    // Rollup: time mean (collapses time dimension to 1)
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean(const SimpleCube<Dtype>& cube) {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        for (size_t lat = 0; lat < LAT; ++lat) {
            for (size_t lon = 0; lon < LON; ++lon) {
                Dtype sum = 0;

                for (size_t t = 0; t < T_dim; ++t) {
                    sum += cube.at(t, lat, lon);
                }

                result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
            }
        }
        return result;
    }

    // Rollup: time sum (collapses time dimension to 1)
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_sum(const SimpleCube<Dtype>& cube) {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        for (size_t lat = 0; lat < LAT; ++lat) {
            for (size_t lon = 0; lon < LON; ++lon) {
                Dtype sum = 0;

                for (size_t t = 0; t < T_dim; ++t) {
                    sum += cube.at(t, lat, lon);
                }

                result.at(0, lat, lon) = sum;
            }
        }

        return result;
    }

    // Global mean across all dimensions
    template<typename Dtype>
    Dtype global_mean(const SimpleCube<Dtype>& cube) {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        Dtype sum = 0;

        for (size_t t = 0; t < T_dim; ++t) {
            for (size_t lat = 0; lat < LAT; ++lat) {
                for (size_t lon = 0; lon < LON; ++lon) {
                    sum += cube.at(t, lat, lon);
                }
            }
        }

        return sum / static_cast<Dtype>(T_dim * LAT * LON);
    }

    // Region mean within specified bounds
    template<typename Dtype>
    Dtype region_mean(const SimpleCube<Dtype>& cube,
                      size_t t_start, size_t t_end,
                      size_t lat_start, size_t lat_end,
                      size_t lon_start, size_t lon_end) {
        Dtype sum = 0;
        size_t count = 0;

        for (size_t t = t_start; t < t_end; ++t) {
            for (size_t lat = lat_start; lat < lat_end; ++lat) {
                for (size_t lon = lon_start; lon < lon_end; ++lon) {
                    sum += cube.at(t, lat, lon);
                    ++count;
                }
            }
        }

        if (count == 0) return 0;

        return sum / static_cast<Dtype>(count);
    }

} // namespace simple_olap
