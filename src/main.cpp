#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "builder/default_cube_builder.h"
#include "cube/datacube.h"
#include "loader/zarr_loader.h"
#include "olap/operations.h"

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


void run(Datacube<float>& cube)
{
    std::string cmd;

    std::cout << "\n=== OLAP Console ===\n";
    std::cout << "Commands:\n";
    std::cout << "1  global_mean\n";
    std::cout << "2  rollup_time_sum\n";
    std::cout << "3  rollup_time_mean\n";
    std::cout << "4  slice_time <t>\n";
    std::cout<< "slice_export\n";
    std::cout << "5  exit\n\n";

    while (true)
    {
        std::cout << ">>> ";
        std::getline(std::cin, cmd);

        if (cmd == "exit" || cmd == "5")
            break;

        else if (cmd == "global_mean" || cmd=="1")
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
        else if (cmd == "rollup_time_mean" || cmd=="3")
        {
            auto rolled = olap::rollup_time_mean(cube);
            std::cout << "Rolled up (mean) over time.\n";
            std::cout << "Result dims: "
                      << rolled.time_dim() << " × "
                      << rolled.lat_dim() << " × "
                      << rolled.lon_dim() << "\n";

            auto slice = olap::slice_time(rolled, 0);

            // Export
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

            break;  // exit CLI loop

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
        else if (cmd.rfind("slice_export", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string tmp;
            size_t t;
            iss >> tmp >> t;

            if (t >= cube.time_dim())
            {
                std::cout << "Invalid time index\n";
                continue;
            }

            auto slice = olap::slice_time(cube, t);

            export_slice_csv(
                slice,
                cube.lat_dim(),
                cube.lon_dim(),
                "slice.csv");

            std::cout << "Exported slice to slice.csv\n";
        }
        else
        {
            std::cout << "Unknown command\n";
        }
    }
}


int main() {
    std::string path = "/media/muqeeth26832/KALI LINUX/GPM_DPR_India_2024.zarr/2D/";


    auto t0 = std::chrono::high_resolution_clock::now();

    auto nsr_data = ZarrLoader::load_float_array(path+"nsr");
    auto t1 = std::chrono::high_resolution_clock::now();

    auto lat_data = ZarrLoader::load_float_array(path+"lat");
    auto t2 = std::chrono::high_resolution_clock::now();

    auto lon_data = ZarrLoader::load_float_array(path+"lon");
    auto t3 = std::chrono::high_resolution_clock::now();

    auto timestamp_data = ZarrLoader::load_string_array(path+"timestamps");
    auto t4 = std::chrono::high_resolution_clock::now();

    auto dur = [](auto start, auto end) {
        return std::chrono::duration<double>(end - start).count();
    };

    std::cout << "\n=== Loading Time Analytics ===\n";
    std::cout << "NSR load time: " << dur(t0,t1) << " sec\n";
    std::cout << "LAT load time: " << dur(t1,t2) << " sec\n";
    std::cout << "LON load time: " << dur(t2,t3) << " sec\n";
    std::cout << "Timestamp load time: " << dur(t3,t4) << " sec\n";
    std::cout << "Total load time: " << dur(t0,t4) << " sec\n";


    std::cout << "Building default cube...\n";
    auto tcube0 = std::chrono::high_resolution_clock::now();
    auto cube =
        DefaultCubeBuilder::build(
            lat_data,
            lon_data,
            nsr_data,
            timestamp_data);
    auto tcube1 = std::chrono::high_resolution_clock::now();
    std::cout << "Total build time: " << dur(tcube0,tcube1) << " sec\n";
    std::cout << "Cube ready.\n";

    run(cube);

    return 0;
}
