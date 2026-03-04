#pragma once

#include <iostream>
#include <algorithm>
#include <vector>

#include "../utils/timer.h"

#include "../builder/simple_cube_builder.h"
#include "../builder/parallel_simple_cube_builder.h"
#include "../builder/omp_sc_builder.h"

#include "../olap/simple_operations.h"
#include "../olap/parallel_operations.h"
#include "../olap/omp_operations.h"


namespace benchmark {

inline void run_benchmark_generic(size_t T, size_t LAT, size_t LON)
{
    std::cout << "\n=== Synthetic Cube Benchmark ===\n";
    std::cout << "Cube size: "
              << T << " × "
              << LAT << " × "
              << LON << "\n\n";

    Timer timer;

    const int RUNS = 10;

    //--------------------------------------------------
    // BUILD
    //--------------------------------------------------

    std::cout << "Building cubes...\n";

    for(int i = 0; i < RUNS; i++)
    {
        auto t0 = Timer::now();
        auto seq_cube = SimpleCube<float>(T,LAT,LON);
        auto t1 = Timer::now();
        timer.record("build_simple", Timer::elapsed(t0,t1));

        auto thread_cube = SimpleCube<float>(T,LAT,LON);
        auto t2 = Timer::now();
        timer.record("build_thread", Timer::elapsed(t1,t2));

        auto omp_cube = SimpleCube<float>(T,LAT,LON);
        auto t3 = Timer::now();
        timer.record("build_omp", Timer::elapsed(t2,t3));
    }

    //--------------------------------------------------
    // Build cubes once for operations
    //--------------------------------------------------

    auto seq_cube = SimpleCube<float>(T,LAT,LON);
    auto thread_cube = SimpleCube<float>(T,LAT,LON);
    auto omp_cube = SimpleCube<float>(T,LAT,LON);

    //--------------------------------------------------
    // WARMUP
    //--------------------------------------------------

    simple_olap::slice_time(seq_cube,0);
    parallel_olap::slice_time(thread_cube,0);
    omp_olap::slice_time(omp_cube,0);

    size_t slice_t = 0;
    size_t t_start = 0;
    size_t t_end = std::min((size_t)10, seq_cube.time_dim());

    //--------------------------------------------------
    // RUN BENCHMARKS
    //--------------------------------------------------

    for(int i = 0; i < RUNS; i++)
    {
        //--------------------------------------------------
        // SLICE
        //--------------------------------------------------

        auto s0 = Timer::now();
        simple_olap::slice_time(seq_cube,slice_t);
        auto s1 = Timer::now();
        timer.record("slice_simple", Timer::elapsed(s0,s1));

        parallel_olap::slice_time(thread_cube,slice_t);
        auto s2 = Timer::now();
        timer.record("slice_thread", Timer::elapsed(s1,s2));

        omp_olap::slice_time(omp_cube,slice_t);
        auto s3 = Timer::now();
        timer.record("slice_omp", Timer::elapsed(s2,s3));


        //--------------------------------------------------
        // DICE
        //--------------------------------------------------

        auto d0 = Timer::now();
        simple_olap::dice_time(seq_cube,t_start,t_end);
        auto d1 = Timer::now();
        timer.record("dice_simple", Timer::elapsed(d0,d1));

        parallel_olap::dice_time(thread_cube,t_start,t_end);
        auto d2 = Timer::now();
        timer.record("dice_thread", Timer::elapsed(d1,d2));

        omp_olap::dice_time(omp_cube,t_start,t_end);
        auto d3 = Timer::now();
        timer.record("dice_omp", Timer::elapsed(d2,d3));


        //--------------------------------------------------
        // ROLLUP
        //--------------------------------------------------

        auto r0 = Timer::now();
        simple_olap::rollup_time_mean(seq_cube);
        auto r1 = Timer::now();
        timer.record("rollup_simple", Timer::elapsed(r0,r1));

        parallel_olap::rollup_time_mean(thread_cube);
        auto r2 = Timer::now();
        timer.record("rollup_thread", Timer::elapsed(r1,r2));

        omp_olap::rollup_time_mean(omp_cube);
        auto r3 = Timer::now();
        timer.record("rollup_omp", Timer::elapsed(r2,r3));


        //--------------------------------------------------
        // GLOBAL
        //--------------------------------------------------

        auto g0 = Timer::now();
        simple_olap::global_mean(seq_cube);
        auto g1 = Timer::now();
        timer.record("global_simple", Timer::elapsed(g0,g1));

        parallel_olap::global_mean(thread_cube);
        auto g2 = Timer::now();
        timer.record("global_thread", Timer::elapsed(g1,g2));

        omp_olap::global_mean(omp_cube);
        auto g3 = Timer::now();
        timer.record("global_omp", Timer::elapsed(g2,g3));
    }

    //--------------------------------------------------
    // EXPORT
    //--------------------------------------------------

    timer.export_csv("benchmark_raw.csv");
    timer.export_summary_csv("benchmark_summary.csv");

    std::cout << "\nBenchmark finished.\n";
    std::cout << "Runs per operation: " << RUNS << "\n";
}
}
