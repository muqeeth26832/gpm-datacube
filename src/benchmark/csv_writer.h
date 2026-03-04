#pragma once
#include <fstream>
#include <string>

namespace benchmark {

class CSVWriter
{
    std::ofstream file;

public:

    CSVWriter(const std::string& path)
    {
        file.open(path);
    }

    void write_header()
    {
        file << "method,operation,T,LAT,LON,time_ms\n";
    }

    void write_row(const std::string& method,
                   const std::string& op,
                   size_t T,
                   size_t LAT,
                   size_t LON,
                   double time)
    {
        file << method << ","
             << op << ","
             << T << ","
             << LAT << ","
             << LON << ","
             << time << "\n";
    }

};

}
