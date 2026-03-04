#pragma once
#include "../cube/simple_cube.h"
#include "../olap/omp_operations.h"
#include "../olap/simple_operations.h"
#include "synthetic_cube.h"
#include <omp.h>
#include <fstream>
#include <vector>

namespace benchmark {

struct SweepConfig {
    std::vector<int> threads;
    std::vector<int> chunk_sizes;
    std::vector<int> tile_sizes;
    std::vector<int> sizes; // LAT=LON
    int trials = 5;
};

inline void run_full_slice_sweep(
    const SweepConfig& config,
    const std::string& output_file)
{
    std::ofstream file(output_file);

    file << "size,threads,mode,param,time_us\n";

    for (int size : config.sizes)
    {
        auto cube = generate_synthetic_cube(1, size, size);

        for (int threads : config.threads)
        {
            omp_set_dynamic(0);
            omp_set_num_threads(threads);

            // Sequential baseline
            double seq_time = 0;
            for (int i = 0; i < config.trials; ++i)
            {
                double start = omp_get_wtime();
                simple_olap::slice_time(cube, 0);
                seq_time += omp_get_wtime() - start;
            }
            seq_time = (seq_time / config.trials) * 1e6;

            file << size << "," << threads
                 << ",sequential,0," << seq_time << "\n";

            // Row sweep
            for (int chunk : config.chunk_sizes)
            {
                double total = 0;
                for (int i = 0; i < config.trials; ++i)
                {
                    double start = omp_get_wtime();
                    omp_olap::slice_time_rowwise(cube, 0, chunk);
                    total += omp_get_wtime() - start;
                }

                file << size << "," << threads
                     << ",row," << chunk << ","
                     << (total / config.trials) * 1e6 << "\n";
            }

            // Col sweep
            for (int chunk : config.chunk_sizes)
            {
                double total = 0;
                for (int i = 0; i < config.trials; ++i)
                {
                    double start = omp_get_wtime();
                    omp_olap::slice_time_colwise(cube, 0, chunk);
                    total += omp_get_wtime() - start;
                }

                file << size << "," << threads
                     << ",col," << chunk << ","
                     << (total / config.trials) * 1e6 << "\n";
            }

            // Tile sweep
            for (int tile : config.tile_sizes)
            {
                double total = 0;
                for (int i = 0; i < config.trials; ++i)
                {
                    double start = omp_get_wtime();
                    omp_olap::slice_time_tiled(cube, 0, tile);
                    total += omp_get_wtime() - start;
                }

                file << size << "," << threads
                     << ",tile," << tile << ","
                     << (total / config.trials) * 1e6 << "\n";
            }
        }
    }

    file.close();
}

}
