#pragma once
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>

template<typename Dtype>
class SimpleCube {
private:
    std::size_t T_dim, LAT_dim, LON_dim;
    std::vector<std::vector<std::vector<Dtype>>> data;

public:
    SimpleCube(size_t T, size_t LAT, size_t LON)
        : T_dim(T), LAT_dim(LAT), LON_dim(LON)
    {
        data.resize(T);
        for (size_t t = 0; t < T; ++t) {
            data[t].resize(LAT);
            for (size_t lat = 0; lat < LAT; ++lat) {
                data[t][lat].resize(LON, Dtype{});
            }
        }
    }

    Dtype& at(size_t t, size_t lat, size_t lon) {
        if (t >= T_dim || lat >= LAT_dim || lon >= LON_dim)
            throw std::out_of_range("Index out of bounds");
        return data[t][lat][lon];
    }

    const Dtype& at(size_t t, size_t lat, size_t lon) const {
        if (t >= T_dim || lat >= LAT_dim || lon >= LON_dim)
            throw std::out_of_range("Index out of bounds");
        return data[t][lat][lon];
    }

    size_t time_dim() const { return T_dim; }
    size_t lat_dim() const { return LAT_dim; }
    size_t lon_dim() const { return LON_dim; }

    void fill(const Dtype& value) {
        for (size_t t = 0; t < T_dim; ++t) {
            for (size_t lat = 0; lat < LAT_dim; ++lat) {
                std::fill(data[t][lat].begin(), data[t][lat].end(), value);
            }
        }
    }
};
