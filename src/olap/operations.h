#pragma once
#include "../cube/datacube.h"
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace olap {
    template<typename Dtype>
    std::vector<Dtype> slice_time(const Datacube<Dtype>&cube, size_t t){
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        std::vector<Dtype> result(LAT*LON);

        for(size_t lat =0;lat<LAT;lat++){
            for(size_t lon = 0;lon<LON;lon++){
                result[lat*LON + lon] = cube.at(t, lat, lon);
            }
        }

        return result;
    }

    template<typename Dtype>
    Datacube<Dtype> dice(const Datacube<Dtype>& cube,
        size_t t_start,size_t t_end,
        size_t lat_start,size_t lat_end,
    size_t lon_start,size_t lon_end)
    {
        size_t newT = t_end-t_start;
        size_t newLAT = lat_end - lat_start;
        size_t newLON = lon_end - lon_start;

        Datacube<Dtype> result(newT,newLAT,newLON);

        for(size_t t=0;t<newT;++t)
            for(size_t lat=0;lat<newLAT;++lat)
                for(size_t lon=0;lon<newLON;++lon)
                    result.at(t,lat,lon)=cube.at(t+t_start,lat+lat_start,lon+lon_start);

        return result;
    }

    template<typename Dtype>
    Datacube<Dtype> rollup_time_mean(const Datacube<Dtype>& cube){
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        Datacube<Dtype> result(1, LAT, LON);

        for(size_t lat=0;lat<LAT;++lat){
            for(size_t lon=0;lon<LON;++lon){
                Dtype sum =0;

                for(size_t t=0;t<T_dim;++t){
                    sum+= cube.at(t,lat,lon); // no cache locality on time
                }

                result.at(0,lat,lon)=sum/static_cast<double>(T_dim);
            }
        }
        return result;
    }

    template<typename Dtype>
    Datacube<Dtype> rollup_time_sum(const Datacube<Dtype>& cube)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        Datacube<Dtype> result(1, LAT, LON);

        for (size_t lat = 0; lat < LAT; ++lat) {
            for (size_t lon = 0; lon < LON; ++lon) {

                Dtype sum = 0;

                for (size_t t = 0; t < T_dim; ++t)
                    sum += cube.at(t, lat, lon);

                result.at(0, lat, lon) = sum;
            }
        }

        return result;
    }

    template<typename Dtype>
    Dtype global_mean(const Datacube<Dtype>& cube)
    {
        size_t T_dim = cube.time_dim();
        size_t LAT = cube.lat_dim();
        size_t LON = cube.lon_dim();

        Dtype sum = 0;

        for (size_t t = 0; t < T_dim; ++t)
            for (size_t lat = 0; lat < LAT; ++lat)
                for (size_t lon = 0; lon < LON; ++lon)
                    sum += cube.at(t, lat, lon);

        return sum / static_cast<double>(T_dim * LAT * LON);
    }

} // namespace olap
