#pragma once
#include "../cube/simple_cube.h"
#include <omp.h>
#include <algorithm>
#include <cstddef>

namespace omp_olap {

    // ========================================================================
    // SLICE OPERATIONS
    // ========================================================================

    template<typename Dtype>
    SimpleCube<Dtype> slice_time_rowwise(
        const SimpleCube<Dtype>& cube,
        size_t t,
        int chunk_size = 0)
    {
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();
        SimpleCube<Dtype> result(1, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                    result.at(0, lat, lon) = cube.at(t, lat, lon);
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                    result.at(0, lat, lon) = cube.at(t, lat, lon);
            }
        }

        return result;
    }

    template<typename Dtype>
    SimpleCube<Dtype> slice_time_colwise(const SimpleCube<Dtype>& cube,
                                         size_t t, int chunk_size = 0) {
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        if(chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size)
            for (size_t lon = 0; lon < LON; ++lon) {
                for (size_t lat = 0; lat < LAT; ++lat) {
                    result.at(0, lat, lon) = cube.at(t, lat, lon);
                }
            }
        }
        else{
            #pragma omp parallel for schedule(static)
            for (size_t lon = 0; lon < LON; ++lon) {
                for (size_t lat = 0; lat < LAT; ++lat) {
                    result.at(0, lat, lon) = cube.at(t, lat, lon);
                }
            }
        }
        return result;
    }

    template<typename Dtype>
    SimpleCube<Dtype> slice_time_tiled(const SimpleCube<Dtype>& cube, size_t t, size_t TILE = 32){
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        #pragma omp parallel for collapse(2) schedule(static)
        for(size_t lat_block = 0; lat_block < LAT; lat_block += TILE){
            for(size_t lon_block = 0; lon_block < LON; lon_block += TILE){
                for(size_t lat = lat_block; lat < std::min(lat_block + TILE, LAT); lat++){
                    for(size_t lon = lon_block; lon < std::min(lon_block + TILE, LON); lon++){
                        result.at(0, lat, lon) = cube.at(t, lat, lon);
                    }
                }
            }
        }

        return result;
    }

    // ========================================================================
    // DICE OPERATIONS - Multiple parallelization strategies
    // ========================================================================

    // Single loop parallelization (outer loop only - time dimension)
    template<typename Dtype>
    SimpleCube<Dtype> dice_time_1loop(const SimpleCube<Dtype>& cube,
                                      size_t t_start, size_t t_end,
                                      int chunk_size = 0)
    {
        size_t newT = t_end - t_start;
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(newT, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }

        return result;
    }

    // Two loops with collapse(2) - time and lat dimensions
    template<typename Dtype>
    SimpleCube<Dtype> dice_time_2loop(const SimpleCube<Dtype>& cube,
                                      size_t t_start, size_t t_end,
                                      int chunk_size = 0)
    {
        size_t newT = t_end - t_start;
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(newT, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for collapse(2) schedule(static, chunk_size)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }

        return result;
    }

    // Three loops with collapse(3) - all dimensions parallelized
    template<typename Dtype>
    SimpleCube<Dtype> dice_time_3loop(const SimpleCube<Dtype>& cube,
                                      size_t t_start, size_t t_end,
                                      int chunk_size = 0)
    {
        size_t newT = t_end - t_start;
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(newT, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for collapse(3) schedule(static, chunk_size)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse(3) schedule(static)
            for (size_t t = 0; t < newT; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                    }
                }
            }
        }

        return result;
    }

    // Tiled version (2D tiling on LAT/LON for each time slice)
    template<typename Dtype>
    SimpleCube<Dtype> dice_time_tiled(const SimpleCube<Dtype>& cube,
                                      size_t t_start, size_t t_end,
                                      size_t TILE = 32)
    {
        size_t newT = t_end - t_start;
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(newT, LAT, LON);

        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < newT; ++t)
        {
            for (size_t lat_block = 0; lat_block < LAT; lat_block += TILE)
            {
                for (size_t lon_block = 0; lon_block < LON; lon_block += TILE)
                {
                    for (size_t lat = lat_block; lat < std::min(lat_block + TILE, LAT); lat++)
                    {
                        for (size_t lon = lon_block; lon < std::min(lon_block + TILE, LON); lon++)
                        {
                            result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                        }
                    }
                }
            }
        }

        return result;
    }

    // Cubed version (3D tiling across T, LAT, LON)
    template<typename Dtype>
    SimpleCube<Dtype> dice_time_cubed(const SimpleCube<Dtype>& cube,
                                      size_t t_start, size_t t_end,
                                      size_t CUBE = 16)
    {
        size_t newT = t_end - t_start;
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(newT, LAT, LON);

        for (size_t t_block = 0; t_block < newT; t_block += CUBE)
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t lat_block = 0; lat_block < LAT; lat_block += CUBE)
            {
                for (size_t lon_block = 0; lon_block < LON; lon_block += CUBE)
                {
                    for (size_t t = t_block; t < std::min(t_block + CUBE, newT); t++)
                    {
                        for (size_t lat = lat_block; lat < std::min(lat_block + CUBE, LAT); lat++)
                        {
                            for (size_t lon = lon_block; lon < std::min(lon_block + CUBE, LON); lon++)
                            {
                                result.at(t, lat, lon) = cube.at(t + t_start, lat, lon);
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    // Alias for backward compatibility
    template<typename Dtype>
    SimpleCube<Dtype> dice_time(const SimpleCube<Dtype>& cube,
                                size_t t_start, size_t t_end,
                                int chunk_size = 0)
    {
        return dice_time_1loop(cube, t_start, t_end, chunk_size);
    }

    // ========================================================================
    // ROLLUP OPERATIONS - Multiple parallelization strategies
    // ========================================================================

    // Single loop parallelization (outer loop - LAT dimension)
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean_1loop(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                }
            }
        }

        return result;
    }

    // Two loops with collapse(2) - LAT and LON dimensions
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean_2loop(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for collapse(2) schedule(static, chunk_size)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                }
            }
        }

        return result;
    }

    // Tiled version (2D tiling on LAT/LON)
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean_tiled(const SimpleCube<Dtype>& cube, size_t TILE = 32)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lat_block = 0; lat_block < LAT; lat_block += TILE)
        {
            for (size_t lon_block = 0; lon_block < LON; lon_block += TILE)
            {
                for (size_t lat = lat_block; lat < std::min(lat_block + TILE, LAT); lat++)
                {
                    for (size_t lon = lon_block; lon < std::min(lon_block + TILE, LON); lon++)
                    {
                        Dtype sum = 0;
                        for (size_t t = 0; t < T_dim; ++t)
                        {
                            sum += cube.at(t, lat, lon);
                        }
                        result.at(0, lat, lon) = sum / static_cast<Dtype>(T_dim);
                    }
                }
            }
        }

        return result;
    }

    // Cubed version (3D blocking - process time blocks with reduction)
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean_cubed(const SimpleCube<Dtype>& cube, size_t CUBE = 16)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        // Initialize result to zero
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lat = 0; lat < LAT; ++lat)
        {
            for (size_t lon = 0; lon < LON; ++lon)
            {
                result.at(0, lat, lon) = 0;
            }
        }

        // Process time in blocks
        for (size_t t_block = 0; t_block < T_dim; t_block += CUBE)
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype partial_sum = 0;
                    for (size_t t = t_block; t < std::min(t_block + CUBE, T_dim); ++t)
                    {
                        partial_sum += cube.at(t, lat, lon);
                    }
                    #pragma omp atomic
                    result.at(0, lat, lon) += partial_sum;
                }
            }
        }

        // Divide by T_dim
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t lat = 0; lat < LAT; ++lat)
        {
            for (size_t lon = 0; lon < LON; ++lon)
            {
                result.at(0, lat, lon) /= static_cast<Dtype>(T_dim);
            }
        }

        return result;
    }

    // Alias for backward compatibility
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_mean(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        return rollup_time_mean_1loop(cube, chunk_size);
    }

    // Rollup sum variants
    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_sum_1loop(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum;
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum;
                }
            }
        }

        return result;
    }

    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_sum_2loop(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        SimpleCube<Dtype> result(1, LAT, LON);

        if (chunk_size > 0)
        {
            #pragma omp parallel for collapse(2) schedule(static, chunk_size)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum;
                }
            }
        }
        else
        {
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    Dtype sum = 0;
                    for (size_t t = 0; t < T_dim; ++t)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                    result.at(0, lat, lon) = sum;
                }
            }
        }

        return result;
    }

    template<typename Dtype>
    SimpleCube<Dtype> rollup_time_sum(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        return rollup_time_sum_1loop(cube, chunk_size);
    }

    // ========================================================================
    // GLOBAL OPERATIONS
    // ========================================================================

    template<typename Dtype>
    Dtype global_mean(const SimpleCube<Dtype>& cube, int chunk_size = 0)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();
        size_t total_elements = T_dim * LAT * LON;

        Dtype total_sum = 0;

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size) reduction(+:total_sum)
            for (size_t t = 0; t < T_dim; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        total_sum += cube.at(t, lat, lon);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static) reduction(+:total_sum)
            for (size_t t = 0; t < T_dim; ++t)
            {
                for (size_t lat = 0; lat < LAT; ++lat)
                {
                    for (size_t lon = 0; lon < LON; ++lon)
                    {
                        total_sum += cube.at(t, lat, lon);
                    }
                }
            }
        }

        return total_sum / static_cast<Dtype>(total_elements);
    }

    // ========================================================================
    // REGION MEAN OPERATIONS
    // ========================================================================

    template<typename Dtype>
    Dtype region_mean(const SimpleCube<Dtype>& cube,
                      size_t t_start, size_t t_end,
                      size_t lat_start, size_t lat_end,
                      size_t lon_start, size_t lon_end,
                      int chunk_size = 0)
    {
        size_t region_size = (t_end - t_start) * (lat_end - lat_start) * (lon_end - lon_start);
        Dtype sum = 0;

        if (chunk_size > 0)
        {
            #pragma omp parallel for schedule(static, chunk_size) reduction(+:sum)
            for (size_t t = t_start; t < t_end; ++t)
            {
                for (size_t lat = lat_start; lat < lat_end; ++lat)
                {
                    for (size_t lon = lon_start; lon < lon_end; ++lon)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static) reduction(+:sum)
            for (size_t t = t_start; t < t_end; ++t)
            {
                for (size_t lat = lat_start; lat < lat_end; ++lat)
                {
                    for (size_t lon = lon_start; lon < lon_end; ++lon)
                    {
                        sum += cube.at(t, lat, lon);
                    }
                }
            }
        }

        return sum / static_cast<Dtype>(region_size);
    }

} // namespace omp_olap
