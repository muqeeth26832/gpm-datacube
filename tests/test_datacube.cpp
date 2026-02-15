#include <iostream>
#include <cassert>

#include "../src/cube/datacube.h"
#include "../src/olap/operations.h"

void test_basic_indexing() {
    Datacube<int> cube(2,2,2);

    cube.at(0,0,0) = 1;
    cube.at(1,1,1) = 8;

    assert(cube.at(0,0,0) == 1);
    assert(cube.at(1,1,1) == 8);

    std::cout << "✓ test_basic_indexing passed\n";
}

void test_rollup_mean() {
    Datacube<double> cube(2,2,2);

    // Fill known values
    cube.at(0,0,0)=1;
    cube.at(0,0,1)=2;
    cube.at(0,1,0)=3;
    cube.at(0,1,1)=4;

    cube.at(1,0,0)=5;
    cube.at(1,0,1)=6;
    cube.at(1,1,0)=7;
    cube.at(1,1,1)=8;

    auto mean_cube = olap::rollup_time_mean(cube);

    assert(mean_cube.at(0,0,0) == 3);
    assert(mean_cube.at(0,0,1) == 4);
    assert(mean_cube.at(0,1,0) == 5);
    assert(mean_cube.at(0,1,1) == 6);

    std::cout << "✓ test_rollup_mean passed\n";
}

void test_global_mean() {
    Datacube<double> cube(2,2,2);

    int value = 1;
    for(size_t t=0;t<2;++t)
        for(size_t lat=0;lat<2;++lat)
            for(size_t lon=0;lon<2;++lon)
                cube.at(t,lat,lon)=value++;

    double gm = olap::global_mean(cube);

    assert(gm == 4.5);

    std::cout << "✓ test_global_mean passed\n";
}

void test_dice() {
    Datacube<int> cube(3,3,3);

    int value = 0;
    for(size_t t=0;t<3;++t)
        for(size_t lat=0;lat<3;++lat)
            for(size_t lon=0;lon<3;++lon)
                cube.at(t,lat,lon)=value++;

    auto sub = olap::dice(cube,
                          1,3,
                          1,3,
                          1,3);

    assert(sub.time_dim()==2);
    assert(sub.lat_dim()==2);
    assert(sub.lon_dim()==2);

    std::cout << "✓ test_dice passed\n";
}

int main() {

    test_basic_indexing();
    test_rollup_mean();
    test_global_mean();
    test_dice();

    std::cout << "\nAll tests passed.\n";

    return 0;
}
