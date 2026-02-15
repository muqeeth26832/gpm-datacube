#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "cube/datacube.h"
#include "olap/operations.h"

void print_2d_cube(const Datacube<double>& cube)
{
    size_t LAT = cube.lat_dim();
    size_t LON = cube.lon_dim();

    std::cout << "\nRollup Result (Time Aggregated):\n\n";

    for (size_t lat = 0; lat < LAT; ++lat) {
        for (size_t lon = 0; lon < LON; ++lon) {
            std::cout << std::setw(8)
                      << cube.at(0, lat, lon)
                      << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}

int main() {

    std::ifstream file("../data/sample.csv");
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV\n";
        return 1;
    }

    size_t T, LAT, LON;
    file >> T >> LAT >> LON;

    Datacube<double> cube(T, LAT, LON);

    size_t t, lat, lon;
    double value;

    while (file >> t >> lat >> lon >> value) {
        cube.at(t, lat, lon) = value;
    }

    // Perform rollup over time


    auto mean_cube = olap::rollup_time_mean(cube);
    print_2d_cube(cube);
    print_2d_cube(mean_cube);

    std::cout << "Rollup Mean at (0,0): "
              << mean_cube.at(0,0,0) << "\n";




    return 0;
}
