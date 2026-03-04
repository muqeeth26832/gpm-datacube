#pragma once

#include "../cube/simple_cube.h"
#include "../olap/simple_operations.h"
#include "../olap/parallel_operations.h"
#include "../olap/omp_operations.h"

#include <omp.h>
#include <utility>

namespace benchmark {

struct SliceComparisonResult {
    double seq_time;
    double std_thread_time;
    double omp_row_time;
    double omp_col_time;
    double omp_tile_time;
};

//
// Generic average timer
//
template <typename F>
double measure_avg(F&& func, int trials = 5)
{
    // Warm-up run (very important)
    func();

    double total = 0.0;

    for (int i = 0; i < trials; ++i)
    {
        double start = omp_get_wtime();
        func();
        double end = omp_get_wtime();
        total += (end - start);
    }

    return total / trials;
}

//
// Compare all slice implementations
//
template<typename Dtype>
SliceComparisonResult compare_slice(
    const SimpleCube<Dtype>& cube,
    size_t t,
    int trials = 5)
{
    SliceComparisonResult result{};

    // --- Sequential ---
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::slice_time(cube, t);
        volatile auto guard = tmp.lat_dim(); // prevent optimization
        (void)guard;
    }, trials);

    // --- std::thread version ---
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::slice_time(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // --- OpenMP Row-wise ---
    result.omp_row_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_rowwise(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // --- OpenMP Column-wise ---
    result.omp_col_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_colwise(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // --- OpenMP Tiled ---
    result.omp_tile_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_tiled(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    return result;
}

} // namespace benchmark
