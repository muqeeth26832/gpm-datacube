#pragma once

#include "../cube/simple_cube.h"
#include "../olap/simple_operations.h"
#include "../olap/parallel_operations.h"
#include "../olap/omp_operations.h"

#include <omp.h>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace benchmark {

// ============================================================================
// Utility: Measure average execution time
// ============================================================================
template <typename F>
double measure_avg(F&& func, int trials = 5)
{
    // Warm-up run
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

// ============================================================================
// SLICE BENCHMARK RESULTS
// ============================================================================
struct SliceBenchmarkResult {
    std::string name;
    double time;
    double speedup_vs_sequential;
};

struct SliceFullResult {
    double seq_time;
    double std_thread_time;
    double omp_default_time;
    double omp_chunk_time;
    double omp_tile_time;
    
    // Speedups
    double speedup_std;
    double speedup_omp_default;
    double speedup_omp_chunk;
    double speedup_omp_tile;
};

template<typename Dtype>
SliceFullResult benchmark_slice(
    const SimpleCube<Dtype>& cube,
    size_t t,
    int chunk_size = 64,
    int tile_size = 32,
    int trials = 5)
{
    SliceFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::slice_time(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::slice_time(cube, t);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_rowwise(cube, t, 0);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_rowwise(cube, t, chunk_size);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP tiled
    result.omp_tile_time = measure_avg([&]{
        auto tmp = omp_olap::slice_time_tiled(cube, t, tile_size);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;
    result.speedup_omp_tile = result.seq_time / result.omp_tile_time;

    return result;
}

// ============================================================================
// DICE BENCHMARK RESULTS
// ============================================================================
struct DiceFullResult {
    double seq_time;
    double std_thread_time;
    double omp_default_time;
    double omp_chunk_time;
    
    double speedup_std;
    double speedup_omp_default;
    double speedup_omp_chunk;
};

template<typename Dtype>
DiceFullResult benchmark_dice_time(
    const SimpleCube<Dtype>& cube,
    size_t t_start, size_t t_end,
    int chunk_size = 16,
    int trials = 5)
{
    DiceFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::dice_time(cube, t_start, t_end);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::dice_time(cube, t_start, t_end);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        auto tmp = omp_olap::dice_time(cube, t_start, t_end, 0);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        auto tmp = omp_olap::dice_time(cube, t_start, t_end, chunk_size);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;

    return result;
}

template<typename Dtype>
DiceFullResult benchmark_dice_region(
    const SimpleCube<Dtype>& cube,
    size_t lat_start, size_t lat_end,
    size_t lon_start, size_t lon_end,
    int chunk_size = 64,
    int trials = 5)
{
    DiceFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::dice_region(cube, lat_start, lat_end, lon_start, lon_end);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::dice_region(cube, lat_start, lat_end, lon_start, lon_end);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        auto tmp = omp_olap::dice_region(cube, lat_start, lat_end, lon_start, lon_end, 0);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        auto tmp = omp_olap::dice_region(cube, lat_start, lat_end, lon_start, lon_end, chunk_size);
        volatile auto guard = tmp.time_dim();
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;

    return result;
}

// ============================================================================
// ROLLUP BENCHMARK RESULTS
// ============================================================================
struct RollupFullResult {
    double seq_time;
    double std_thread_time;
    double omp_default_time;
    double omp_chunk_time;
    
    double speedup_std;
    double speedup_omp_default;
    double speedup_omp_chunk;
};

template<typename Dtype>
RollupFullResult benchmark_rollup_time_mean(
    const SimpleCube<Dtype>& cube,
    int chunk_size = 64,
    int trials = 5)
{
    RollupFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::rollup_time_mean(cube);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::rollup_time_mean(cube);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        auto tmp = omp_olap::rollup_time_mean(cube, 0);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        auto tmp = omp_olap::rollup_time_mean(cube, chunk_size);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;

    return result;
}

template<typename Dtype>
RollupFullResult benchmark_rollup_time_sum(
    const SimpleCube<Dtype>& cube,
    int chunk_size = 64,
    int trials = 5)
{
    RollupFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        auto tmp = simple_olap::rollup_time_sum(cube);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        auto tmp = parallel_olap::rollup_time_sum(cube);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        auto tmp = omp_olap::rollup_time_sum(cube, 0);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        auto tmp = omp_olap::rollup_time_sum(cube, chunk_size);
        volatile auto guard = tmp.lat_dim();
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;

    return result;
}

// ============================================================================
// GLOBAL MEAN BENCHMARK RESULTS
// ============================================================================
struct GlobalMeanFullResult {
    double seq_time;
    double std_thread_time;
    double omp_default_time;
    double omp_chunk_time;
    
    double speedup_std;
    double speedup_omp_default;
    double speedup_omp_chunk;
};

template<typename Dtype>
GlobalMeanFullResult benchmark_global_mean(
    const SimpleCube<Dtype>& cube,
    int chunk_size = 8,
    int trials = 5)
{
    GlobalMeanFullResult result{};

    // Sequential
    result.seq_time = measure_avg([&]{
        volatile auto guard = simple_olap::global_mean(cube);
        (void)guard;
    }, trials);

    // std::thread
    result.std_thread_time = measure_avg([&]{
        volatile auto guard = parallel_olap::global_mean(cube);
        (void)guard;
    }, trials);

    // OpenMP default
    result.omp_default_time = measure_avg([&]{
        volatile auto guard = omp_olap::global_mean(cube, 0);
        (void)guard;
    }, trials);

    // OpenMP with chunk size
    result.omp_chunk_time = measure_avg([&]{
        volatile auto guard = omp_olap::global_mean(cube, chunk_size);
        (void)guard;
    }, trials);

    // Calculate speedups
    result.speedup_std = result.seq_time / result.std_thread_time;
    result.speedup_omp_default = result.seq_time / result.omp_default_time;
    result.speedup_omp_chunk = result.seq_time / result.omp_chunk_time;

    return result;
}

// ============================================================================
// PRINT UTILITIES
// ============================================================================
inline void print_slice_result(const SliceFullResult& r) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Sequential:      " << r.seq_time << "s\n";
    std::cout << "  std::thread:     " << r.std_thread_time << "s (speedup: " << r.speedup_std << "x)\n";
    std::cout << "  OMP default:     " << r.omp_default_time << "s (speedup: " << r.speedup_omp_default << "x)\n";
    std::cout << "  OMP chunk:       " << r.omp_chunk_time << "s (speedup: " << r.speedup_omp_chunk << "x)\n";
    std::cout << "  OMP tiled:       " << r.omp_tile_time << "s (speedup: " << r.speedup_omp_tile << "x)\n";
}

inline void print_dice_result(const DiceFullResult& r) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Sequential:      " << r.seq_time << "s\n";
    std::cout << "  std::thread:     " << r.std_thread_time << "s (speedup: " << r.speedup_std << "x)\n";
    std::cout << "  OMP default:     " << r.omp_default_time << "s (speedup: " << r.speedup_omp_default << "x)\n";
    std::cout << "  OMP chunk:       " << r.omp_chunk_time << "s (speedup: " << r.speedup_omp_chunk << "x)\n";
}

inline void print_rollup_result(const RollupFullResult& r) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Sequential:      " << r.seq_time << "s\n";
    std::cout << "  std::thread:     " << r.std_thread_time << "s (speedup: " << r.speedup_std << "x)\n";
    std::cout << "  OMP default:     " << r.omp_default_time << "s (speedup: " << r.speedup_omp_default << "x)\n";
    std::cout << "  OMP chunk:       " << r.omp_chunk_time << "s (speedup: " << r.speedup_omp_chunk << "x)\n";
}

inline void print_global_mean_result(const GlobalMeanFullResult& r) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Sequential:      " << r.seq_time << "s\n";
    std::cout << "  std::thread:     " << r.std_thread_time << "s (speedup: " << r.speedup_std << "x)\n";
    std::cout << "  OMP default:     " << r.omp_default_time << "s (speedup: " << r.speedup_omp_default << "x)\n";
    std::cout << "  OMP chunk:       " << r.omp_chunk_time << "s (speedup: " << r.speedup_omp_chunk << "x)\n";
}

} // namespace benchmark
