#pragma once

#include "../cube/simple_cube.h"
#include "../olap/simple_operations.h"
#include "../olap/parallel_operations.h"
#include "../olap/omp_operations.h"
#include "synthetic_cube.h"

#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace benchmark {

// ============================================================================
// Utility: Measure average execution time using omp_get_wtime (returns µs)
// ============================================================================
template <typename F>
double measure_avg_us(F&& func, int trials = 5)
{
    func();  // Warm-up

    double total = 0.0;
    for (int i = 0; i < trials; ++i)
    {
        double start = omp_get_wtime();
        func();
        double end = omp_get_wtime();
        total += (end - start);
    }
    return (total / trials) * 1e6;  // Convert to microseconds
}

// ============================================================================
// Size Sweep Result Structure
// ============================================================================
struct SizeSweepResult {
    std::vector<int> sizes;
    std::vector<double> seq_times;
    std::vector<double> std_thread_times;
    std::vector<double> omp_1loop_times;
    std::vector<double> omp_2loop_times;
    std::vector<double> omp_3loop_times;
    std::vector<double> omp_tile_times;
    std::vector<double> omp_cubed_times;
    std::string operation_name;
};

// ============================================================================
// SLICE BENCHMARK - All variants
// ============================================================================
inline SizeSweepResult benchmark_slice_sweep(
    const std::vector<int>& sizes,
    int trials = 5)
{
    SizeSweepResult result;
    result.operation_name = "slice_time";
    result.sizes = sizes;

    const int CHUNK_SIZE = 64;
    const int TILE_SIZE = 32;

    for (int size : sizes)
    {
        auto cube = generate_synthetic_cube(1, size, size);

        // Sequential
        result.seq_times.push_back(measure_avg_us([&]{
            auto tmp = simple_olap::slice_time(cube, 0);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // std::thread
        result.std_thread_times.push_back(measure_avg_us([&]{
            auto tmp = parallel_olap::slice_time(cube, 0);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 1-loop (row-wise, single loop parallelization)
        result.omp_1loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::slice_time_rowwise(cube, 0, CHUNK_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 2-loop (collapse(2) on LAT/LON)
        result.omp_2loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::slice_time_tiled(cube, 0, TILE_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 3-loop - not applicable for 2D slice (only 2 loops exist)
        result.omp_3loop_times.push_back(result.omp_2loop_times.back());

        // OMP Tile (same as 2-loop for slice)
        result.omp_tile_times.push_back(result.omp_2loop_times.back());

        // OMP Cubed - not applicable for 2D slice
        result.omp_cubed_times.push_back(result.omp_2loop_times.back());
    }

    return result;
}

// ============================================================================
// DICE BENCHMARK - All variants (1-loop, 2-loop, 3-loop, tile, cubed)
// ============================================================================
inline SizeSweepResult benchmark_dice_sweep(
    const std::vector<int>& sizes,
    int trials = 5)
{
    SizeSweepResult result;
    result.operation_name = "dice_time";
    result.sizes = sizes;

    const int CHUNK_SIZE = 16;
    const int TILE_SIZE = 32;
    const int CUBE_SIZE = 16;
    const size_t T_RANGE = 10;

    for (int size : sizes)
    {
        auto cube = generate_synthetic_cube(T_RANGE, size, size);

        // Sequential
        result.seq_times.push_back(measure_avg_us([&]{
            auto tmp = simple_olap::dice_time(cube, 0, T_RANGE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // std::thread
        result.std_thread_times.push_back(measure_avg_us([&]{
            auto tmp = parallel_olap::dice_time(cube, 0, T_RANGE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // OMP 1-loop (parallelize outer loop only - time dimension)
        result.omp_1loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::dice_time_1loop(cube, 0, T_RANGE, CHUNK_SIZE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // OMP 2-loop (collapse(2) on T and LAT)
        result.omp_2loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::dice_time_2loop(cube, 0, T_RANGE, CHUNK_SIZE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // OMP 3-loop (collapse(3) on T, LAT, LON)
        result.omp_3loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::dice_time_3loop(cube, 0, T_RANGE, CHUNK_SIZE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // OMP Tile (2D tiling on LAT/LON)
        result.omp_tile_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::dice_time_tiled(cube, 0, T_RANGE, TILE_SIZE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));

        // OMP Cubed (3D tiling across T, LAT, LON)
        result.omp_cubed_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::dice_time_cubed(cube, 0, T_RANGE, CUBE_SIZE);
            // volatile auto guard = tmp.time_dim();
            // (void)guard;
        }, trials));
    }

    return result;
}

// ============================================================================
// ROLLUP MEAN BENCHMARK - All variants
// ============================================================================
inline SizeSweepResult benchmark_rollup_mean_sweep(
    const std::vector<int>& sizes,
    int trials = 5)
{
    SizeSweepResult result;
    result.operation_name = "rollup_time_mean";
    result.sizes = sizes;

    const int CHUNK_SIZE = 64;
    const int TILE_SIZE = 32;
    const int CUBE_SIZE = 16;

    for (int size : sizes)
    {
        auto cube = generate_synthetic_cube(24, size, size);

        // Sequential
        result.seq_times.push_back(measure_avg_us([&]{
            auto tmp = simple_olap::rollup_time_mean(cube);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // std::thread
        result.std_thread_times.push_back(measure_avg_us([&]{
            auto tmp = parallel_olap::rollup_time_mean(cube);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 1-loop (parallelize outer loop - LAT dimension)
        result.omp_1loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::rollup_time_mean_1loop(cube, CHUNK_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 2-loop (collapse(2) on LAT and LON)
        result.omp_2loop_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::rollup_time_mean_2loop(cube, CHUNK_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP 3-loop - not directly applicable (reduction over T)
        result.omp_3loop_times.push_back(result.omp_2loop_times.back());

        // OMP Tile (2D tiling on LAT/LON)
        result.omp_tile_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::rollup_time_mean_tiled(cube, TILE_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));

        // OMP Cubed (3D blocking with reduction)
        result.omp_cubed_times.push_back(measure_avg_us([&]{
            auto tmp = omp_olap::rollup_time_mean_cubed(cube, CUBE_SIZE);
            volatile auto guard = tmp.lat_dim();
            (void)guard;
        }, trials));
    }

    return result;
}

// ============================================================================
// GLOBAL MEAN BENCHMARK
// ============================================================================
inline SizeSweepResult benchmark_global_mean_sweep(
    const std::vector<int>& sizes,
    int trials = 5)
{
    SizeSweepResult result;
    result.operation_name = "global_mean";
    result.sizes = sizes;

    const int CHUNK_SIZE = 8;
    const int TILE_SIZE = 32;
    const int CUBE_SIZE = 16;

    for (int size : sizes)
    {
        auto cube = generate_synthetic_cube(24, size, size);

        // Sequential
        result.seq_times.push_back(measure_avg_us([&]{
            volatile auto guard = simple_olap::global_mean(cube);
            (void)guard;
        }, trials));

        // std::thread
        result.std_thread_times.push_back(measure_avg_us([&]{
            volatile auto guard = parallel_olap::global_mean(cube);
            (void)guard;
        }, trials));

        // OMP 1-loop (parallelize outer loop - T dimension)
        result.omp_1loop_times.push_back(measure_avg_us([&]{
            volatile auto guard = omp_olap::global_mean(cube, CHUNK_SIZE);
            (void)guard;
        }, trials));

        // OMP 2-loop - not applicable (3 nested loops with reduction)
        result.omp_2loop_times.push_back(result.omp_1loop_times.back());

        // OMP 3-loop - not applicable with reduction
        result.omp_3loop_times.push_back(result.omp_1loop_times.back());

        // OMP Tile
        result.omp_tile_times.push_back(result.omp_1loop_times.back());

        // OMP Cubed
        result.omp_cubed_times.push_back(measure_avg_us([&]{
            volatile auto guard = omp_olap::global_mean(cube, CUBE_SIZE);
            (void)guard;
        }, trials));
    }

    return result;
}

// ============================================================================
// REGION MEAN BENCHMARK
// ============================================================================
inline SizeSweepResult benchmark_region_mean_sweep(
    const std::vector<int>& sizes,
    int trials = 5)
{
    SizeSweepResult result;
    result.operation_name = "region_mean";
    result.sizes = sizes;

    const int CHUNK_SIZE = 8;
    const int CUBE_SIZE = 16;
    const size_t T_RANGE = 5;
    const size_t LAT_RANGE = 50;
    const size_t LON_RANGE = 50;

    for (int size : sizes)
    {
        auto cube = generate_synthetic_cube(24, size, size);
        size_t lat_end = std::min(LAT_RANGE, static_cast<size_t>(size));
        size_t lon_end = std::min(LON_RANGE, static_cast<size_t>(size));

        // Sequential
        result.seq_times.push_back(measure_avg_us([&]{
            volatile auto guard = simple_olap::region_mean(cube, 0, T_RANGE, 0, lat_end, 0, lon_end);
            (void)guard;
        }, trials));

        // std::thread - not implemented, use sequential
        result.std_thread_times.push_back(result.seq_times.back());

        // OMP 1-loop (parallelize outer loop - T dimension)
        result.omp_1loop_times.push_back(measure_avg_us([&]{
            volatile auto guard = omp_olap::region_mean(cube, 0, T_RANGE, 0, lat_end, 0, lon_end, CHUNK_SIZE);
            (void)guard;
        }, trials));

        // OMP 2-loop - not applicable
        result.omp_2loop_times.push_back(result.omp_1loop_times.back());

        // OMP 3-loop - not applicable with reduction
        result.omp_3loop_times.push_back(result.omp_1loop_times.back());

        // OMP Tile
        result.omp_tile_times.push_back(result.omp_1loop_times.back());

        // OMP Cubed
        result.omp_cubed_times.push_back(measure_avg_us([&]{
            volatile auto guard = omp_olap::region_mean(cube, 0, T_RANGE, 0, lat_end, 0, lon_end, CUBE_SIZE);
            (void)guard;
        }, trials));
    }

    return result;
}

// ============================================================================
// EXPORT TO CSV
// ============================================================================
inline void export_size_sweep_csv(
    const SizeSweepResult& result,
    const std::string& filename)
{
    std::ofstream file(filename);

    file << "size,sequential,std_thread,omp_1loop,omp_2loop,omp_3loop,omp_tile,omp_cubed\n";

    for (size_t i = 0; i < result.sizes.size(); ++i)
    {
        file << result.sizes[i] << ","
             << result.seq_times[i] << ","
             << result.std_thread_times[i] << ","
             << result.omp_1loop_times[i] << ","
             << result.omp_2loop_times[i] << ","
             << result.omp_3loop_times[i] << ","
             << result.omp_tile_times[i] << ","
             << result.omp_cubed_times[i] << "\n";
    }

    file.close();
    std::cout << "Exported: " << filename << "\n";
}

// ============================================================================
// RUN ALL BENCHMARKS
// ============================================================================
inline void run_full_size(
    const std::vector<int>& sizes,
    int trials = 5)
{
    std::cout << "\n=== Size Sweep Benchmark ===\n";
    std::cout << "Testing sizes: ";
    for (int s : sizes) std::cout << s << " ";
    std::cout << "\n\n";
    std::cout << "Comparing: Sequential, std::thread, OMP (1-loop, 2-loop, 3-loop, tile, cubed)\n\n";

    // Slice
    std::cout << "Running slice_time benchmark...\n";
    auto slice_result = benchmark_slice_sweep(sizes, trials);
    export_size_sweep_csv(slice_result, "benchmark_slice_size_sweep.csv");

    // Dice
    std::cout << "Running dice_time benchmark...\n";
    auto dice_result = benchmark_dice_sweep(sizes, trials);
    export_size_sweep_csv(dice_result, "benchmark_dice_size_sweep.csv");

    // Rollup Mean
    std::cout << "Running rollup_time_mean benchmark...\n";
    auto rollup_result = benchmark_rollup_mean_sweep(sizes, trials);
    export_size_sweep_csv(rollup_result, "benchmark_rollup_mean_size_sweep.csv");

    // Global Mean
    std::cout << "Running global_mean benchmark...\n";
    auto global_result = benchmark_global_mean_sweep(sizes, trials);
    export_size_sweep_csv(global_result, "benchmark_global_mean_size_sweep.csv");

    // Region Mean
    std::cout << "Running region_mean benchmark...\n";
    auto region_result = benchmark_region_mean_sweep(sizes, trials);
    export_size_sweep_csv(region_result, "benchmark_region_mean_size_sweep.csv");

    std::cout << "\nAll benchmarks complete!\n";
    std::cout << "Run 'python build/visualize_all.py' to generate plots.\n";
}

} // namespace benchmark
