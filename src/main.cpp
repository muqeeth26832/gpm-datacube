#include <cstddef>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "cube/datacube.h"
#include "loader/zarr_loader.h"
#include "olap/operations.h"


int main() {

    std::string path = "/media/muqeeth26832/KALI LINUX/GPM_DPR_India_2024.zarr/2D/";

    auto nsr_data = ZarrLoader::load_float_array(path+"nsr");
    auto lat_data = ZarrLoader::load_float_array(path+"lat");
    auto lon_data = ZarrLoader::load_float_array(path+"lon");

    double min_lat = std::numeric_limits<double>::max();
    double max_lat = -min_lat;
    double min_lon = std::numeric_limits<double>::max();
    double max_lon = -min_lon;

    std::unordered_set<float> unique_lat;
    std::unordered_set<float> unique_lon;


    for(size_t i=0;i<lat_data.size();++i)
    {
        float lat = lat_data[i];
        float lon = lon_data[i];

        min_lat = std::min(min_lat,(double)lat);
        max_lat = std::max(max_lat, (double)lat);

        min_lon = std::min(min_lon, (double)lon);
        max_lon = std::max(max_lon, (double)lon);

        unique_lat.insert(lat);
        unique_lon.insert(lon);
    }

    std::cout << "Lat range: " << min_lat << " to " << max_lat << "\n";
    std::cout << "Lon range: " << min_lon << " to " << max_lon << "\n";
    std::cout << "Unique lat count: " << unique_lat.size() << "\n";
    std::cout << "Unique lon count: " << unique_lon.size() << "\n";


    auto timestamp_data = ZarrLoader::load_string_array(path+"timestamps");
    std::unordered_set<std::string> unique_time;

    for(int i=0;i<5;i++)
        std::cout<<timestamp_data[i]<<"\n";

    for (auto& ts : timestamp_data)
        unique_time.insert(ts);

    std::cout << "Unique timestamps: "
              << unique_time.size() << "\n";

    return 0;
}
