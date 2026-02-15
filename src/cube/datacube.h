#pragma once
#include <cstddef>
#include <scoped_allocator>
#include <vector>
#include <stdexcept>
#include <algorithm>

template<typename Dtype>
class Datacube {
private:
    std::size_t T_dim,LAT_dim,LON_dim;
    std::vector<Dtype> data;

    size_t index(size_t t,size_t lat,size_t lon) const {
        if(t>=T_dim || lat >= LAT_dim || lon >= LON_dim)
            throw std::out_of_range("Index out of bounds");

        return t*(LAT_dim * LON_dim) + lat*LON_dim + lon;
    }

public:
    Datacube(size_t T,size_t LAT, size_t LON)
    : T_dim(T),LAT_dim(LAT),LON_dim(LON),data(T*LAT*LON) {}

    Dtype& at(size_t t,size_t lat,size_t lon) {
        return data[index(t,lat,lon)];
    }

    const Dtype& at(size_t t, size_t lat, size_t lon) const {
            return data[index(t, lat, lon)];
    }

    size_t time_dim() const { return T_dim; }
    size_t lat_dim() const { return LAT_dim; }
    size_t lon_dim() const { return LON_dim; }

    void fill(const Dtype& value) {
        std::fill(data.begin(), data.end(), value);
    }
};
