#pragma once
#include <cstdint>
#include <string>
#include <vector>

class ZarrLoader {
public:
    // Load float32 array (<f4)
    static std::vector<float>
    load_float_array(const std::string& folder_path);

    // Load fixed-width string array (|SXX)
    static std::vector<std::string>
    load_string_array(const std::string& folder_path);
};
