#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "builder/default_cube_builder.h"
#include "builder/simple_cube_builder.h"
#include "cube/datacube.h"
#include "cube/simple_cube.h"
#include "loader/zarr_loader.h"
#include "olap/operations.h"
#include "olap/simple_operations.h"
#include "utils/timer.h"

#include <chrono>
#include <fstream>

void export_slice_csv(
    const std::vector<float>& slice,
    size_t lat_bins,
    size_t lon_bins,
    const std::string& filename)
{
    std::ofstream file(filename);

    for (size_t lat = 0; lat < lat_bins; ++lat)
    {
        for (size_t lon = 0; lon < lon_bins; ++lon)
        {
            file << slice[lat * lon_bins + lon];

            if (lon != lon_bins - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

void export_simple_cube_csv(
    const SimpleCube<float>& cube,
    size_t t,
    const std::string& filename)
{
    std::ofstream file(filename);

    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    for (size_t lat = 0; lat < LAT; ++lat)
    {
        for (size_t lon = 0; lon < LON; ++lon)
        {
            file << cube.at(t, lat, lon);

            if (lon != LON - 1)
                file << ",";
        }
        file << "\n";
    }

    file.close();
}

void run_datacube(Datacube<float>& cube)
{
    std::string cmd;

    std::cout << "\n=== OLAP Console (Datacube) ===\n";
    std::cout << "Commands:\n";
    std::cout << "1  global_mean\n";
    std::cout << "2  rollup_time_sum\n";
    std::cout << "3  rollup_time_mean\n";
    std::cout << "4  slice_time <t>\n";
    std::cout << "5  dice_time <t_start> <t_end>\n";
    std::cout << "6  exit\n\n";

    while (true)
    {
        std::cout << ">>> ";
        std::getline(std::cin, cmd);

        if (cmd == "exit" || cmd == "6")
            break;

        else if (cmd == "global_mean" || cmd == "1")
        {
            float val = olap::global_mean(cube);
            std::cout << "Global Mean: " << val << "\n";
        }
        else if (cmd == "rollup_time_sum" || cmd == "2")
        {
            auto rolled = olap::rollup_time_sum(cube);
            std::cout << "Rolled up (sum) over time.\n";
            std::cout << "Result dims: "
                      << rolled.time_dim() << " × "
                      << rolled.lat_dim() << " × "
                      << rolled.lon_dim() << "\n";
        }
        else if (cmd == "rollup_time_mean" || cmd == "3")
        {
            auto rolled = olap::rollup_time_mean(cube);
            std::cout << "Rolled up (mean) over time.\n";
            std::cout << "Result dims: "
                      << rolled.time_dim() << " × "
                      << rolled.lat_dim() << " × "
                      << rolled.lon_dim() << "\n";

            auto slice = olap::slice_time(rolled, 0);

            std::ofstream file("mean_rainfall.csv");

            size_t LAT = rolled.lat_dim();
            size_t LON = rolled.lon_dim();

            for (size_t lat = 0; lat < LAT; ++lat)
            {
                for (size_t lon = 0; lon < LON; ++lon)
                {
                    file << slice[lat * LON + lon];

                    if (lon != LON - 1)
                        file << ",";
                }
                file << "\n";
            }

            file.close();

            std::cout << "Exported to mean_rainfall.csv\n";
        }
        else if (cmd.rfind("slice_time", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t;
            iss >> temp >> t;

            if (t >= cube.time_dim())
            {
                std::cout << "Invalid time index\n";
                continue;
            }

            auto slice = olap::slice_time(cube, t);

            std::cout << "Slice at time " << t << "\n";
            std::cout << "Lat × Lon grid:\n";

            size_t LAT = cube.lat_dim();
            size_t LON = cube.lon_dim();

            for (size_t i = 0; i < LAT; ++i)
            {
                for (size_t j = 0; j < LON; ++j)
                {
                    std::cout << std::fixed
                              << std::setprecision(2)
                              << slice[i * LON + j] << " ";
                }
                std::cout << "\n";
            }
        }
        else if (cmd.rfind("dice_time", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t_start, t_end;
            iss >> temp >> t_start >> t_end;

            if (t_end > cube.time_dim() || t_start >= t_end)
            {
                std::cout << "Invalid time range\n";
                continue;
            }

            auto diced = olap::dice_time(cube, t_start, t_end);

            std::cout << "Diced cube dims: "
                      << diced.time_dim() << " × "
                      << diced.lat_dim() << " × "
                      << diced.lon_dim() << "\n";
        }
        else
        {
            std::cout << "Unknown command\n";
        }
    }
}

void run_simplecube(SimpleCube<float>& cube, Timer& timer)
{
    std::string cmd;

    std::cout << "\n=== OLAP Console (SimpleCube) ===\n";
    std::cout << "Cube Dimensions: "
              << cube.time_dim() << " (time) × "
              << cube.lat_dim() << " (lat) × "
              << cube.lon_dim() << " (lon)\n";
    std::cout << "Geographic Coverage: Lat [5°N-40°N], Lon [65°E-100°E]\n";
    std::cout << "Resolution: 0.25°\n\n";

    std::cout << "Commands:\n";
    std::cout << "1  global_mean\n";
    std::cout << "2  rollup_time_sum\n";
    std::cout << "3  rollup_time_mean\n";
    std::cout << "4  slice_time <t>           (example: slice_time 0)\n";
    std::cout << "5  dice_time <t1> <t2>      (example: dice_time 0 10)\n";
    std::cout << "6  dice_region <lat1> <lat2> <lon1> <lon2>  (example: dice_region 0 50 0 50)\n";
    std::cout << "7  region_mean <t1> <t2> <lat1> <lat2> <lon1> <lon2>  (example: region_mean 0 5 0 20 0 20)\n";
    std::cout << "8  export_slice <t>         (example: export_slice 0)\n";
    std::cout << "9  export_timing_summary\n";
    std::cout << "10 info                     (show cube stats)\n";
    std::cout << "11 exit\n\n";

    while (true)
    {
        std::cout << ">>> ";
        std::getline(std::cin, cmd);

        if (cmd == "exit" || cmd == "10")
            break;

        else if (cmd == "global_mean" || cmd == "1")
        {
            auto t0 = Timer::now();
            float val = simple_olap::global_mean(cube);
            auto t1 = Timer::now();
            timer.record("global_mean", Timer::elapsed(t0, t1));
            std::cout << "Global Mean: " << val << "\n";
        }
        else if (cmd == "rollup_time_sum" || cmd == "2")
        {
            auto t0 = Timer::now();
            auto rolled = simple_olap::rollup_time_sum(cube);
            auto t1 = Timer::now();
            timer.record("rollup_time_sum", Timer::elapsed(t0, t1));
            std::cout << "Rolled up (sum) over time.\n";
            std::cout << "Result dims: "
                      << rolled.time_dim() << " × "
                      << rolled.lat_dim() << " × "
                      << rolled.lon_dim() << "\n";
        }
        else if (cmd == "rollup_time_mean" || cmd == "3")
        {
            auto t0 = Timer::now();
            auto rolled = simple_olap::rollup_time_mean(cube);
            auto t1 = Timer::now();
            timer.record("rollup_time_mean", Timer::elapsed(t0, t1));
            std::cout << "Rolled up (mean) over time.\n";
            std::cout << "Result dims: "
                      << rolled.time_dim() << " × "
                      << rolled.lat_dim() << " × "
                      << rolled.lon_dim() << "\n";

            export_simple_cube_csv(rolled, 0, "mean_rainfall_simple.csv");
            std::cout << "Exported to mean_rainfall_simple.csv\n";
        }
        else if (cmd.rfind("slice_time", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t;
            iss >> temp >> t;

            if (t >= cube.time_dim())
            {
                std::cout << "Invalid time index\n";
                continue;
            }

            auto t0 = Timer::now();
            auto slice = simple_olap::slice_time(cube, t);
            auto t1 = Timer::now();
            timer.record("slice_time", Timer::elapsed(t0, t1));

            std::cout << "Slice at time " << t << "\n";
            std::cout << "Lat × Lon grid:\n";

            size_t LAT = cube.lat_dim();
            size_t LON = cube.lon_dim();

            for (size_t i = 0; i < LAT; ++i)
            {
                for (size_t j = 0; j < LON; ++j)
                {
                    std::cout << std::fixed
                              << std::setprecision(2)
                              << slice.at(0, i, j) << " ";
                }
                std::cout << "\n";
            }
        }
        else if (cmd.rfind("dice_time", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t_start, t_end;
            iss >> temp >> t_start >> t_end;

            if (t_end > cube.time_dim() || t_start >= t_end)
            {
                std::cout << "Invalid time range\n";
                continue;
            }

            auto t0 = Timer::now();
            auto diced = simple_olap::dice_time(cube, t_start, t_end);
            auto t1 = Timer::now();
            timer.record("dice_time", Timer::elapsed(t0, t1));

            std::cout << "Diced cube dims: "
                      << diced.time_dim() << " × "
                      << diced.lat_dim() << " × "
                      << diced.lon_dim() << "\n";
        }
        else if (cmd.rfind("dice_region", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t lat_start, lat_end, lon_start, lon_end;
            iss >> temp >> lat_start >> lat_end >> lon_start >> lon_end;

            if (lat_end > cube.lat_dim() || lon_end > cube.lon_dim() ||
                lat_start >= lat_end || lon_start >= lon_end)
            {
                std::cout << "Invalid region\n";
                continue;
            }

            auto t0 = Timer::now();
            auto diced = simple_olap::dice_region(cube, lat_start, lat_end, lon_start, lon_end);
            auto t1 = Timer::now();
            timer.record("dice_region", Timer::elapsed(t0, t1));

            std::cout << "Diced cube dims: "
                      << diced.time_dim() << " × "
                      << diced.lat_dim() << " × "
                      << diced.lon_dim() << "\n";
        }
        else if (cmd.rfind("region_mean", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t_start, t_end, lat_start, lat_end, lon_start, lon_end;
            iss >> temp >> t_start >> t_end >> lat_start >> lat_end >> lon_start >> lon_end;

            if (t_end > cube.time_dim() || lat_end > cube.lat_dim() || lon_end > cube.lon_dim() ||
                t_start >= t_end || lat_start >= lat_end || lon_start >= lon_end)
            {
                std::cout << "Invalid region\n";
                continue;
            }

            auto t0 = Timer::now();
            float val = simple_olap::region_mean(cube, t_start, t_end, lat_start, lat_end, lon_start, lon_end);
            auto t1 = Timer::now();
            timer.record("region_mean", Timer::elapsed(t0, t1));

            std::cout << "Region Mean: " << val << "\n";
        }
        else if (cmd.rfind("export_slice", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string temp;
            size_t t;
            iss >> temp >> t;

            if (t >= cube.time_dim())
            {
                std::cout << "Invalid time index\n";
                continue;
            }

            auto t0 = Timer::now();
            auto slice = simple_olap::slice_time(cube, t);
            export_simple_cube_csv(slice, 0, "slice_simple.csv");
            auto t1 = Timer::now();
            timer.record("slice_time", Timer::elapsed(t0, t1));

            std::cout << "Exported slice to slice_simple.csv\n";
        }
        else if (cmd == "export_timing_summary" || cmd == "9")
        {
            timer.export_summary_csv("timing_summary.csv");
            std::cout << "Timing summary exported to timing_summary.csv\n";
        }
        else if (cmd == "info" || cmd == "10")
        {
            std::cout << "\n=== Cube Statistics ===\n";
            std::cout << "Time dimension:  " << cube.time_dim() << " bins\n";
            std::cout << "Latitude dimension: " << cube.lat_dim() << " bins\n";
            std::cout << "Longitude dimension: " << cube.lon_dim() << " bins\n";
            std::cout << "Total cells: " << cube.time_dim() * cube.lat_dim() * cube.lon_dim() << "\n\n";

            std::cout << "Valid Query Ranges:\n";
            std::cout << "  Time index (t):     0 to " << (cube.time_dim() - 1) << "\n";
            std::cout << "  Latitude index:     0 to " << (cube.lat_dim() - 1)
                      << " (maps to 5°N - 40°N)\n";
            std::cout << "  Longitude index:    0 to " << (cube.lon_dim() - 1)
                      << " (maps to 65°E - 100°E)\n\n";

            std::cout << "Geographic Resolution: 0.25°\n";
            std::cout << "Latitude per bin:  (40 - 5) / " << cube.lat_dim() << " = 0.25°\n";
            std::cout << "Longitude per bin: (100 - 65) / " << cube.lon_dim() << " = 0.25°\n\n";

            std::cout << "Example Queries:\n";
            std::cout << "  slice_time 0                  - Get first time slice\n";
            std::cout << "  dice_time 0 24                - Get first 24 time bins\n";
            std::cout << "  dice_region 0 40 0 40         - Get NW corner region\n";
            std::cout << "  region_mean 0 10 0 20 0 20    - Mean of sub-region\n";
            std::cout << "\n";
        }
        else if (cmd == "exit" || cmd == "11")
            break;
        else
        {
            std::cout << "Unknown command\n";
        }
    }
}

int main() {
    std::string path = "/media/muqeeth26832/KALI LINUX/GPM_DPR_India_2024.zarr/2D/";

    std::cout << "=== GPM Datacube Project ===\n";
    std::cout << "Select cube type:\n";
    std::cout << "1. Datacube (flat vector storage)\n";
    std::cout << "2. SimpleCube (3D vector storage)\n";
    std::cout << "Choice: ";

    std::string choice;
    std::getline(std::cin, choice);

    Timer timer;

    if (choice == "1")
    {
        // Datacube version
        auto t0 = std::chrono::high_resolution_clock::now();

        auto nsr_data = ZarrLoader::load_float_array(path + "nsr");
        auto t1 = std::chrono::high_resolution_clock::now();

        auto lat_data = ZarrLoader::load_float_array(path + "lat");
        auto t2 = std::chrono::high_resolution_clock::now();

        auto lon_data = ZarrLoader::load_float_array(path + "lon");
        auto t3 = std::chrono::high_resolution_clock::now();

        auto timestamp_data = ZarrLoader::load_string_array(path + "timestamps");
        auto t4 = std::chrono::high_resolution_clock::now();

        auto dur = [](auto start, auto end) {
            return std::chrono::duration<double>(end - start).count();
        };

        std::cout << "\n=== Loading Time Analytics ===\n";
        std::cout << "NSR load time: " << dur(t0, t1) << " sec\n";
        std::cout << "LAT load time: " << dur(t1, t2) << " sec\n";
        std::cout << "LON load time: " << dur(t2, t3) << " sec\n";
        std::cout << "Timestamp load time: " << dur(t3, t4) << " sec\n";
        std::cout << "Total load time: " << dur(t0, t4) << " sec\n";


        std::cout << "\nBuilding default cube...\n";
        auto tcube0 = std::chrono::high_resolution_clock::now();
        auto cube = DefaultCubeBuilder::build(
            lat_data,
            lon_data,
            nsr_data,
            timestamp_data);
        auto tcube1 = std::chrono::high_resolution_clock::now();
        std::cout << "Total build time: " << dur(tcube0, tcube1) << " sec\n";
        std::cout << "Cube ready.\n";

        run_datacube(cube);
    }
    else
    {
        // SimpleCube version
        auto t0 = std::chrono::high_resolution_clock::now();

        auto nsr_data = ZarrLoader::load_float_array(path + "nsr");
        auto t1 = std::chrono::high_resolution_clock::now();
        timer.record("load_nsr", std::chrono::duration<double>(t1 - t0).count());

        auto lat_data = ZarrLoader::load_float_array(path + "lat");
        auto t2 = std::chrono::high_resolution_clock::now();
        timer.record("load_lat", std::chrono::duration<double>(t2 - t1).count());

        auto lon_data = ZarrLoader::load_float_array(path + "lon");
        auto t3 = std::chrono::high_resolution_clock::now();
        timer.record("load_lon", std::chrono::duration<double>(t3 - t2).count());

        auto timestamp_data = ZarrLoader::load_string_array(path + "timestamps");
        auto t4 = std::chrono::high_resolution_clock::now();
        timer.record("load_timestamps", std::chrono::duration<double>(t4 - t3).count());

        auto dur = [](auto start, auto end) {
            return std::chrono::duration<double>(end - start).count();
        };

        std::cout << "\n=== Loading Time Analytics ===\n";
        std::cout << "NSR load time: " << dur(t0, t1) << " sec\n";
        std::cout << "LAT load time: " << dur(t1, t2) << " sec\n";
        std::cout << "LON load time: " << dur(t2, t3) << " sec\n";
        std::cout << "Timestamp load time: " << dur(t3, t4) << " sec\n";
        std::cout << "Total load time: " << dur(t0, t4) << " sec\n";

        std::cout << "\nBuilding simple cube...\n";
        auto tcube0 = std::chrono::high_resolution_clock::now();
        auto cube = SimpleCubeBuilder::build(
            lat_data,
            lon_data,
            nsr_data,
            timestamp_data);
        auto tcube1 = std::chrono::high_resolution_clock::now();
        double build_time = dur(tcube0, tcube1);
        timer.record("build_cube", build_time);
        std::cout << "Total build time: " << build_time << " sec\n";
        std::cout << "SimpleCube ready.\n";

        run_simplecube(cube, timer);

        // Export timing data on exit
        timer.export_csv("timing_raw.csv");
        timer.export_summary_csv("timing_summary.csv");
        std::cout << "\nTiming data exported to timing_raw.csv and timing_summary.csv\n";
    }

    return 0;
}
