#pragma once
#include "../cube/simple_cube.h"
#include <omp.h>
#include <cstddef>

namespace omp_olap {

//////////////////////////////////////////////////////////////
// SLICE
//////////////////////////////////////////////////////////////

template<typename Dtype>
SimpleCube<Dtype> slice_time(const SimpleCube<Dtype>& cube, size_t t)
{
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    SimpleCube<Dtype> result(1, LAT, LON);

#pragma omp parallel for schedule(static)
    for (size_t lat = 0; lat < LAT; ++lat)
    {
    #pragma omp simd
        for (size_t lon = 0; lon < LON; ++lon)
        {
            result.at(0, lat, lon) = cube.at(t, lat, lon);
        }
    }

    return result;
}

//////////////////////////////////////////////////////////////
// DICE
//////////////////////////////////////////////////////////////

template<typename Dtype>
SimpleCube<Dtype> dice(const SimpleCube<Dtype>& cube,
                       size_t t_start, size_t t_end,
                       size_t lat_start, size_t lat_end,
                       size_t lon_start, size_t lon_end)
{
    size_t newT   = t_end - t_start;
    size_t newLAT = lat_end - lat_start;
    size_t newLON = lon_end - lon_start;

    SimpleCube<Dtype> result(newT, newLAT, newLON);

#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < newT; ++t)
    {
        for (size_t lat = 0; lat < newLAT; ++lat)
        {
        #pragma omp simd
            for (size_t lon = 0; lon < newLON; ++lon)
            {
                result.at(t, lat, lon) =
                    cube.at(t + t_start,
                            lat + lat_start,
                            lon + lon_start);
            }
        }
    }

    return result;
}

//////////////////////////////////////////////////////////////
// DICE TIME
//////////////////////////////////////////////////////////////

template<typename Dtype>
SimpleCube<Dtype> dice_time(const SimpleCube<Dtype>& cube,
                            size_t t_start, size_t t_end)
{
    return dice(cube,
                t_start, t_end,
                0, cube.lat_dim(),
                0, cube.lon_dim());
}

//////////////////////////////////////////////////////////////
// DICE REGION
//////////////////////////////////////////////////////////////

template<typename Dtype>
SimpleCube<Dtype> dice_region(const SimpleCube<Dtype>& cube,
                              size_t lat_start, size_t lat_end,
                              size_t lon_start, size_t lon_end)
{
    return dice(cube,
                0, cube.time_dim(),
                lat_start, lat_end,
                lon_start, lon_end);
}

//////////////////////////////////////////////////////////////
// ROLLUP TIME MEAN
//////////////////////////////////////////////////////////////

// template<typename Dtype>
// SimpleCube<Dtype> rollup_time_mean(const SimpleCube<Dtype>& cube)
// {
//     size_t T   = cube.time_dim();
//     size_t LAT = cube.lat_dim();
//     size_t LON = cube.lon_dim();

//     SimpleCube<Dtype> result(1, LAT, LON);

// #pragma omp parallel for collapse(2) schedule(static,10)
//     for (size_t lat = 0; lat < LAT; ++lat)
//     {
//         for (size_t lon = 0; lon < LON; ++lon)
//         {
//             Dtype sum = 0;

//             for (size_t t = 0; t < T; ++t)
//             {
//                 sum += cube.at(t, lat, lon);
//             }

//             result.at(0, lat, lon) = sum / static_cast<Dtype>(T);
//         }
//     }

//     return result;
// }

template<typename Dtype>
SimpleCube<Dtype> rollup_time_mean(const SimpleCube<Dtype>& cube)
{
    size_t T   = cube.time_dim();
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    SimpleCube<Dtype> result(1, LAT, LON);

#pragma omp parallel
{
    // initialize
#pragma omp for collapse(2) schedule(static)
    for (size_t lat = 0; lat < LAT; ++lat)
        for (size_t lon = 0; lon < LON; ++lon)
            result.at(0, lat, lon) = 0;

    // accumulate
    for (size_t t = 0; t < T; ++t)
    {
#pragma omp for schedule(static)
        for (size_t lat = 0; lat < LAT; ++lat)
        {
#pragma omp simd
            for (size_t lon = 0; lon < LON; ++lon)
            {
                result.at(0, lat, lon) += cube.at(t, lat, lon);
            }
        }
    }

    // normalize
#pragma omp for collapse(2) schedule(static)
    for (size_t lat = 0; lat < LAT; ++lat)
        for (size_t lon = 0; lon < LON; ++lon)
            result.at(0, lat, lon) /= static_cast<Dtype>(T);
}

    return result;
}



//////////////////////////////////////////////////////////////
// ROLLUP TIME SUM
//////////////////////////////////////////////////////////////

template<typename Dtype>
SimpleCube<Dtype> rollup_time_sum(const SimpleCube<Dtype>& cube)
{
    size_t T   = cube.time_dim();
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    SimpleCube<Dtype> result(1, LAT, LON);

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lat = 0; lat < LAT; ++lat)
    {
        for (size_t lon = 0; lon < LON; ++lon)
        {
            Dtype sum = 0;

            for (size_t t = 0; t < T; ++t)
            {
                sum += cube.at(t, lat, lon);
            }

            result.at(0, lat, lon) = sum;
        }
    }

    return result;
}

//////////////////////////////////////////////////////////////
// GLOBAL MEAN
//////////////////////////////////////////////////////////////

template<typename Dtype>
Dtype global_mean(const SimpleCube<Dtype>& cube)
{
    size_t T   = cube.time_dim();
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    Dtype sum = 0;

#pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t t = 0; t < T; ++t)
    {
        for (size_t lat = 0; lat < LAT; ++lat)
        {
            #pragma omp simd reduction(+:sum)
            for (size_t lon = 0; lon < LON; ++lon)
            {
                sum += cube.at(t, lat, lon);
            }
        }
    }

    return sum / static_cast<Dtype>(T * LAT * LON);
}


//////////////////////////////////////////////////////////////
// REGION MEAN
//////////////////////////////////////////////////////////////

template<typename Dtype>
Dtype region_mean(const SimpleCube<Dtype>& cube,
                  size_t t_start, size_t t_end,
                  size_t lat_start, size_t lat_end,
                  size_t lon_start, size_t lon_end)
{
    Dtype sum = 0;

#pragma omp parallel for collapse(3) reduction(+:sum) schedule(static)
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

    size_t count =
        (t_end - t_start) *
        (lat_end - lat_start) *
        (lon_end - lon_start);

    return sum / static_cast<Dtype>(count);
}

} // namespace omp_olap
