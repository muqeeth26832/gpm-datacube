#pragma once

#include "olap_benchmark.h"
#include "../cube/simple_cube.h"
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

namespace benchmark {

// ============================================================================
// COMPREHENSIVE BENCHMARK SUITE RESULTS
// ============================================================================

struct FullBenchmarkSuite {
    // Slice benchmarks with different parameters
    std::vector<double> slice_seq_times;
    std::vector<double> slice_std_times;
    std::vector<double> slice_omp_default_times;
    std::vector<double> slice_omp_chunk_times;
    std::vector<double> slice_omp_tile_times;
    
    // Dice benchmarks
    std::vector<double> dice_seq_times;
    std::vector<double> dice_std_times;
    std::vector<double> dice_omp_default_times;
    std::vector<double> dice_omp_chunk_times;
    
    // Rollup mean benchmarks
    std::vector<double> rollup_mean_seq_times;
    std::vector<double> rollup_mean_std_times;
    std::vector<double> rollup_mean_omp_default_times;
    std::vector<double> rollup_mean_omp_chunk_times;
    
    // Rollup sum benchmarks
    std::vector<double> rollup_sum_seq_times;
    std::vector<double> rollup_sum_std_times;
    std::vector<double> rollup_sum_omp_default_times;
    std::vector<double> rollup_sum_omp_chunk_times;
    
    // Global mean benchmarks
    std::vector<double> global_mean_seq_times;
    std::vector<double> global_mean_std_times;
    std::vector<double> global_mean_omp_default_times;
    std::vector<double> global_mean_omp_chunk_times;
    
    // Parameters used
    std::vector<int> chunk_sizes;           // For chunk size sweep
    std::vector<int> tile_sizes;            // For tile size sweep
    std::vector<int> thread_counts;         // For thread count sweep (if applicable)
    
    std::string cube_info;
};

// ============================================================================
// CHUNK SIZE SWEEP BENCHMARK
// ============================================================================
template<typename Dtype>
FullBenchmarkSuite benchmark_chunk_size_sweep(
    const SimpleCube<Dtype>& cube,
    const std::vector<int>& chunk_sizes,
    int tile_size = 32,
    int trials = 5)
{
    FullBenchmarkSuite suite;
    suite.chunk_sizes = chunk_sizes;
    suite.cube_info = "Cube: " + std::to_string(cube.time_dim()) + "x" + 
                      std::to_string(cube.lat_dim()) + "x" + std::to_string(cube.lon_dim());

    std::cout << "\n=== Chunk Size Sweep Benchmark ===\n";
    std::cout << suite.cube_info << "\n";
    std::cout << "Testing chunk sizes: ";
    for (int cs : chunk_sizes) std::cout << cs << " ";
    std::cout << "\n\n";

    for (int chunk_size : chunk_sizes)
    {
        std::cout << "Chunk size: " << chunk_size << "\n";

        // Slice benchmark
        auto slice_result = benchmark_slice(cube, 0, chunk_size, tile_size, trials);
        suite.slice_omp_chunk_times.push_back(slice_result.omp_chunk_time);
        
        // Dice benchmark
        size_t t_end = std::min(size_t(10), cube.time_dim());
        auto dice_result = benchmark_dice_time(cube, 0, t_end, chunk_size, trials);
        suite.dice_omp_chunk_times.push_back(dice_result.omp_chunk_time);
        
        // Rollup mean benchmark
        auto rollup_mean_result = benchmark_rollup_time_mean(cube, chunk_size, trials);
        suite.rollup_mean_omp_chunk_times.push_back(rollup_mean_result.omp_chunk_time);
        
        // Rollup sum benchmark
        auto rollup_sum_result = benchmark_rollup_time_sum(cube, chunk_size, trials);
        suite.rollup_sum_omp_chunk_times.push_back(rollup_sum_result.omp_chunk_time);
        
        // Global mean benchmark
        auto global_mean_result = benchmark_global_mean(cube, chunk_size, trials);
        suite.global_mean_omp_chunk_times.push_back(global_mean_result.omp_chunk_time);

        std::cout << "  Slice: " << std::fixed << std::setprecision(6) 
                  << slice_result.omp_chunk_time << "s\n";
        std::cout << "  Dice: " << dice_result.omp_chunk_time << "s\n";
        std::cout << "  Rollup Mean: " << rollup_mean_result.omp_chunk_time << "s\n";
        std::cout << "  Rollup Sum: " << rollup_sum_result.omp_chunk_time << "s\n";
        std::cout << "  Global Mean: " << global_mean_result.omp_chunk_time << "s\n\n";
    }

    // Run baseline benchmarks (seq, std, omp default) once
    std::cout << "Running baseline benchmarks...\n";
    
    // Slice baseline
    auto slice_base = benchmark_slice(cube, 0, 64, tile_size, trials);
    suite.slice_seq_times.push_back(slice_base.seq_time);
    suite.slice_std_times.push_back(slice_base.std_thread_time);
    suite.slice_omp_default_times.push_back(slice_base.omp_default_time);
    suite.slice_omp_tile_times.push_back(slice_base.omp_tile_time);
    
    // Dice baseline
    size_t t_end = std::min(size_t(10), cube.time_dim());
    auto dice_base = benchmark_dice_time(cube, 0, t_end, 64, trials);
    suite.dice_seq_times.push_back(dice_base.seq_time);
    suite.dice_std_times.push_back(dice_base.std_thread_time);
    suite.dice_omp_default_times.push_back(dice_base.omp_default_time);
    
    // Rollup mean baseline
    auto rollup_mean_base = benchmark_rollup_time_mean(cube, 64, trials);
    suite.rollup_mean_seq_times.push_back(rollup_mean_base.seq_time);
    suite.rollup_mean_std_times.push_back(rollup_mean_base.std_thread_time);
    suite.rollup_mean_omp_default_times.push_back(rollup_mean_base.omp_default_time);
    
    // Rollup sum baseline
    auto rollup_sum_base = benchmark_rollup_time_sum(cube, 64, trials);
    suite.rollup_sum_seq_times.push_back(rollup_sum_base.seq_time);
    suite.rollup_sum_std_times.push_back(rollup_sum_base.std_thread_time);
    suite.rollup_sum_omp_default_times.push_back(rollup_sum_base.omp_default_time);
    
    // Global mean baseline
    auto global_mean_base = benchmark_global_mean(cube, 64, trials);
    suite.global_mean_seq_times.push_back(global_mean_base.seq_time);
    suite.global_mean_std_times.push_back(global_mean_base.std_thread_time);
    suite.global_mean_omp_default_times.push_back(global_mean_base.omp_default_time);

    std::cout << "Baseline complete.\n";

    return suite;
}

// ============================================================================
// TILE SIZE SWEEP BENCHMARK
// ============================================================================
template<typename Dtype>
FullBenchmarkSuite benchmark_tile_size_sweep(
    const SimpleCube<Dtype>& cube,
    const std::vector<int>& tile_sizes,
    int trials = 5)
{
    FullBenchmarkSuite suite;
    suite.tile_sizes = tile_sizes;
    suite.cube_info = "Cube: " + std::to_string(cube.time_dim()) + "x" + 
                      std::to_string(cube.lat_dim()) + "x" + std::to_string(cube.lon_dim());

    std::cout << "\n=== Tile Size Sweep Benchmark ===\n";
    std::cout << suite.cube_info << "\n";
    std::cout << "Testing tile sizes: ";
    for (int ts : tile_sizes) std::cout << ts << " ";
    std::cout << "\n\n";

    for (int tile_size : tile_sizes)
    {
        std::cout << "Tile size: " << tile_size << "\n";

        // Slice benchmark with tiling
        auto slice_result = benchmark_slice(cube, 0, 64, tile_size, trials);
        suite.slice_omp_tile_times.push_back(slice_result.omp_tile_time);

        std::cout << "  Slice (tiled): " << std::fixed << std::setprecision(6) 
                  << slice_result.omp_tile_time << "s\n\n";
    }

    // Run baseline benchmarks once
    std::cout << "Running baseline benchmarks...\n";
    
    auto slice_base = benchmark_slice(cube, 0, 64, 32, trials);
    suite.slice_seq_times.push_back(slice_base.seq_time);
    suite.slice_std_times.push_back(slice_base.std_thread_time);
    suite.slice_omp_default_times.push_back(slice_base.omp_default_time);
    suite.slice_omp_chunk_times.push_back(slice_base.omp_chunk_time);

    std::cout << "Baseline complete.\n";

    return suite;
}

// ============================================================================
// FULL OPERATION COMPARISON BENCHMARK
// ============================================================================
template<typename Dtype>
FullBenchmarkSuite benchmark_all_operations(
    const SimpleCube<Dtype>& cube,
    int chunk_size = 64,
    int tile_size = 32,
    int trials = 5)
{
    FullBenchmarkSuite suite;
    suite.chunk_sizes = {chunk_size};
    suite.tile_sizes = {tile_size};
    suite.cube_info = "Cube: " + std::to_string(cube.time_dim()) + "x" + 
                      std::to_string(cube.lat_dim()) + "x" + std::to_string(cube.lon_dim());

    std::cout << "\n=== Full Operations Benchmark ===\n";
    std::cout << suite.cube_info << "\n";
    std::cout << "Chunk size: " << chunk_size << ", Tile size: " << tile_size << "\n\n";

    // Slice benchmark
    std::cout << "Slice (t=0):\n";
    auto slice_result = benchmark_slice(cube, 0, chunk_size, tile_size, trials);
    suite.slice_seq_times.push_back(slice_result.seq_time);
    suite.slice_std_times.push_back(slice_result.std_thread_time);
    suite.slice_omp_default_times.push_back(slice_result.omp_default_time);
    suite.slice_omp_chunk_times.push_back(slice_result.omp_chunk_time);
    suite.slice_omp_tile_times.push_back(slice_result.omp_tile_time);
    print_slice_result(slice_result);
    std::cout << "\n";

    // Dice benchmark
    size_t t_end = std::min(size_t(10), cube.time_dim());
    std::cout << "Dice (t=0.." << t_end << "):\n";
    auto dice_result = benchmark_dice_time(cube, 0, t_end, chunk_size, trials);
    suite.dice_seq_times.push_back(dice_result.seq_time);
    suite.dice_std_times.push_back(dice_result.std_thread_time);
    suite.dice_omp_default_times.push_back(dice_result.omp_default_time);
    suite.dice_omp_chunk_times.push_back(dice_result.omp_chunk_time);
    print_dice_result(dice_result);
    std::cout << "\n";

    // Rollup mean benchmark
    std::cout << "Rollup Time Mean:\n";
    auto rollup_mean_result = benchmark_rollup_time_mean(cube, chunk_size, trials);
    suite.rollup_mean_seq_times.push_back(rollup_mean_result.seq_time);
    suite.rollup_mean_std_times.push_back(rollup_mean_result.std_thread_time);
    suite.rollup_mean_omp_default_times.push_back(rollup_mean_result.omp_default_time);
    suite.rollup_mean_omp_chunk_times.push_back(rollup_mean_result.omp_chunk_time);
    print_rollup_result(rollup_mean_result);
    std::cout << "\n";

    // Rollup sum benchmark
    std::cout << "Rollup Time Sum:\n";
    auto rollup_sum_result = benchmark_rollup_time_sum(cube, chunk_size, trials);
    suite.rollup_sum_seq_times.push_back(rollup_sum_result.seq_time);
    suite.rollup_sum_std_times.push_back(rollup_sum_result.std_thread_time);
    suite.rollup_sum_omp_default_times.push_back(rollup_sum_result.omp_default_time);
    suite.rollup_sum_omp_chunk_times.push_back(rollup_sum_result.omp_chunk_time);
    print_rollup_result(rollup_sum_result);
    std::cout << "\n";

    // Global mean benchmark
    std::cout << "Global Mean:\n";
    auto global_mean_result = benchmark_global_mean(cube, chunk_size, trials);
    suite.global_mean_seq_times.push_back(global_mean_result.seq_time);
    suite.global_mean_std_times.push_back(global_mean_result.std_thread_time);
    suite.global_mean_omp_default_times.push_back(global_mean_result.omp_default_time);
    suite.global_mean_omp_chunk_times.push_back(global_mean_result.omp_chunk_time);
    print_global_mean_result(global_mean_result);
    std::cout << "\n";

    return suite;
}

// ============================================================================
// EXPORT TO CSV
// ============================================================================
void export_chunk_size_sweep_csv(const FullBenchmarkSuite& suite, 
                                  const std::string& filename = "benchmark_chunk_size_sweep.csv")
{
    std::ofstream file(filename);
    
    file << "chunk_size,slice_omp_chunk,dice_omp_chunk,rollup_mean_omp_chunk,rollup_sum_omp_chunk,global_mean_omp_chunk,";
    file << "slice_seq,slice_std,slice_omp_default,slice_omp_tile,";
    file << "dice_seq,dice_std,dice_omp_default,";
    file << "rollup_mean_seq,rollup_mean_std,rollup_mean_omp_default,";
    file << "rollup_sum_seq,rollup_sum_std,rollup_sum_omp_default,";
    file << "global_mean_seq,global_mean_std,global_mean_omp_default\n";

    // First, write the baseline values (same for all rows) - converted to microseconds
    std::string baseline = std::to_string(suite.slice_seq_times[0] * 1e6) + "," +
                           std::to_string(suite.slice_std_times[0] * 1e6) + "," +
                           std::to_string(suite.slice_omp_default_times[0] * 1e6) + "," +
                           std::to_string(suite.slice_omp_tile_times[0] * 1e6) + "," +
                           std::to_string(suite.dice_seq_times[0] * 1e6) + "," +
                           std::to_string(suite.dice_std_times[0] * 1e6) + "," +
                           std::to_string(suite.dice_omp_default_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_mean_seq_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_mean_std_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_mean_omp_default_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_sum_seq_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_sum_std_times[0] * 1e6) + "," +
                           std::to_string(suite.rollup_sum_omp_default_times[0] * 1e6) + "," +
                           std::to_string(suite.global_mean_seq_times[0] * 1e6) + "," +
                           std::to_string(suite.global_mean_std_times[0] * 1e6) + "," +
                           std::to_string(suite.global_mean_omp_default_times[0] * 1e6);

    for (size_t i = 0; i < suite.chunk_sizes.size(); ++i)
    {
        file << suite.chunk_sizes[i] << ",";
        file << suite.slice_omp_chunk_times[i] * 1e6 << ",";
        file << suite.dice_omp_chunk_times[i] * 1e6 << ",";
        file << suite.rollup_mean_omp_chunk_times[i] * 1e6 << ",";
        file << suite.rollup_sum_omp_chunk_times[i] * 1e6 << ",";
        file << suite.global_mean_omp_chunk_times[i] * 1e6 << ",";
        file << baseline << "\n";
    }

    file.close();
    std::cout << "Exported chunk size sweep to " << filename << "\n";
}

void export_tile_size_sweep_csv(const FullBenchmarkSuite& suite,
                                 const std::string& filename = "benchmark_tile_size_sweep.csv")
{
    std::ofstream file(filename);
    
    file << "tile_size,slice_omp_tile,slice_seq,slice_std,slice_omp_default,slice_omp_chunk\n";

    for (size_t i = 0; i < suite.tile_sizes.size(); ++i)
    {
        file << suite.tile_sizes[i] << ",";
        file << suite.slice_omp_tile_times[i] * 1e6 << ",";
        file << suite.slice_seq_times[0] * 1e6 << ",";
        file << suite.slice_std_times[0] * 1e6 << ",";
        file << suite.slice_omp_default_times[0] * 1e6 << ",";
        file << suite.slice_omp_chunk_times[0] * 1e6 << "\n";
    }

    file.close();
    std::cout << "Exported tile size sweep to " << filename << "\n";
}

void export_all_operations_csv(const FullBenchmarkSuite& suite,
                                const std::string& filename = "benchmark_all_operations.csv")
{
    std::ofstream file(filename);
    
    file << "operation,sequential,std_thread,omp_default,omp_chunk,omp_tile\n";
    
    // Helper lambda to write a row - times in microseconds
    auto write_row = [&file](const std::string& op,
                             const std::vector<double>& seq,
                             const std::vector<double>& std_t,
                             const std::vector<double>& omp_def,
                             const std::vector<double>& omp_ch,
                             const std::vector<double>& omp_tl)
    {
        file << op << ",";
        file << (seq.empty() ? 0 : seq[0] * 1e6) << ",";
        file << (std_t.empty() ? 0 : std_t[0] * 1e6) << ",";
        file << (omp_def.empty() ? 0 : omp_def[0] * 1e6) << ",";
        file << (omp_ch.empty() ? 0 : omp_ch[0] * 1e6) << ",";
        file << (omp_tl.empty() ? 0 : omp_tl[0] * 1e6) << "\n";
    };

    write_row("slice", suite.slice_seq_times, suite.slice_std_times,
              suite.slice_omp_default_times, suite.slice_omp_chunk_times, suite.slice_omp_tile_times);

    write_row("dice", suite.dice_seq_times, suite.dice_std_times,
              suite.dice_omp_default_times, suite.dice_omp_chunk_times, suite.dice_omp_default_times);

    write_row("rollup_mean", suite.rollup_mean_seq_times, suite.rollup_mean_std_times,
              suite.rollup_mean_omp_default_times, suite.rollup_mean_omp_chunk_times, suite.rollup_mean_omp_default_times);

    write_row("rollup_sum", suite.rollup_sum_seq_times, suite.rollup_sum_std_times,
              suite.rollup_sum_omp_default_times, suite.rollup_sum_omp_chunk_times, suite.rollup_sum_omp_default_times);

    write_row("global_mean", suite.global_mean_seq_times, suite.global_mean_std_times,
              suite.global_mean_omp_default_times, suite.global_mean_omp_chunk_times, suite.global_mean_omp_default_times);

    file.close();
    std::cout << "Exported all operations to " << filename << "\n";
}

// ============================================================================
// EXPORT METADATA (for visualization context)
// ============================================================================
void export_benchmark_metadata(const FullBenchmarkSuite& suite,
                                const std::string& filename = "benchmark_metadata.csv")
{
    std::ofstream file(filename);
    
    file << "# Benchmark Metadata - Constant Parameters\n";
    file << "# This file contains information about fixed parameters during benchmark runs\n";
    file << "#\n";
    file << "key,value\n";
    file << "cube_dimensions," << suite.cube_info << "\n";
    
    // If chunk_sizes has data, we ran chunk sweep
    if (!suite.chunk_sizes.empty())
    {
        file << "chunk_sizes_tested,\"";
        for (size_t i = 0; i < suite.chunk_sizes.size(); ++i)
        {
            file << suite.chunk_sizes[i];
            if (i < suite.chunk_sizes.size() - 1) file << ",";
        }
        file << "\"\n";
        file << "tile_size_fixed,32\n";
    }
    
    // If tile_sizes has data, we ran tile sweep
    if (!suite.tile_sizes.empty())
    {
        file << "tile_sizes_tested,\"";
        for (size_t i = 0; i < suite.tile_sizes.size(); ++i)
        {
            file << suite.tile_sizes[i];
            if (i < suite.tile_sizes.size() - 1) file << ",";
        }
        file << "\"\n";
        file << "chunk_size_fixed,64\n";
    }
    
    file << "trials_per_measurement,5\n";
    file << "time_unit,microseconds\n";
    
    file.close();
    std::cout << "Exported benchmark metadata to " << filename << "\n";
}

} // namespace benchmark
