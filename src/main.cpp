#include <cstddef>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "cube/datacube.h"
#include "loader/zarr_loader.h"
#include "olap/operations.h"

#include <chrono>

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


    std::cout << "\n=== Dataset Size ===\n";
    std::cout << "Total observations: " << nsr_data.size() << "\n";
    std::cout << "Memory (raw floats only): "
              << nsr_data.size()*sizeof(float)/1024.0/1024.0
              << " MB\n";

    double min_lat = 1e9, max_lat = -1e9;
    double min_lon = 1e9, max_lon = -1e9;

    for (size_t i=0;i<lat_data.size();++i)
    {
        min_lat = std::min(min_lat,(double)lat_data[i]);
        max_lat = std::max(max_lat,(double)lat_data[i]);
        min_lon = std::min(min_lon,(double)lon_data[i]);
        max_lon = std::max(max_lon,(double)lon_data[i]);
    }

    std::cout << "\n=== Spatial Range ===\n";
    std::cout << "Latitude: " << min_lat << " to " << max_lat << "\n";
    std::cout << "Longitude: " << min_lon << " to " << max_lon << "\n";

    std::unordered_set<std::string> unique_hour;
    for (auto& ts : timestamp_data)
        unique_hour.insert(ts.substr(0,13));

    std::cout << "Unique hourly bins: "
              << unique_hour.size() << "\n";

    size_t non_zero = 0;
    for (float v : nsr_data)
        if (v > 0) non_zero++;

    std::cout << "\n=== Sparsity ===\n";
    std::cout << "Non-zero ratio: "
            << (double)non_zero/nsr_data.size()
            << "\n";
    return 0;
}
