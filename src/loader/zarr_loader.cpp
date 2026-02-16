#include "zarr_loader.h"

#include <atomic>
#include <cstddef>
#include <fstream>
#include <vector>
#include <cstring>
#include <zlib.h>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


#include "zarr_loader.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <zlib.h>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

std::vector<float>
ZarrLoader::load_float_array(const std::string& folder_path)
{
    std::ifstream meta_file(folder_path + "/.zarray");
    if (!meta_file)
        throw std::runtime_error("Failed to open .zarray");

    json meta;
    meta_file >> meta;

    size_t total_size = meta["shape"][0];
    size_t chunk_size = meta["chunks"][0];

    size_t num_chunks =
        (total_size + chunk_size - 1) / chunk_size;

    std::vector<float> result(total_size);

    size_t offset = 0;

    for (size_t chunk_index = 0;
         chunk_index < num_chunks;
         ++chunk_index)
    {
        std::string chunk_path =
            folder_path + "/" +
            std::to_string(chunk_index);

        std::ifstream chunk_file(chunk_path,
                                 std::ios::binary);

        size_t remaining = total_size - offset;
        size_t logical_elements =
            std::min(chunk_size, remaining);

        // 🔥 Always allocate FULL chunk size for decompression
        size_t full_chunk_bytes =
            chunk_size * sizeof(float);

        if (!chunk_file)
        {
            // Missing chunk → fill_value
            std::memset(
                result.data() + offset,
                0,
                logical_elements * sizeof(float));

            offset += logical_elements;
            continue;
        }

        std::vector<unsigned char> compressed(
            (std::istreambuf_iterator<char>(chunk_file)),
            std::istreambuf_iterator<char>());

        if (compressed.empty())
        {
            std::memset(
                result.data() + offset,
                0,
                logical_elements * sizeof(float));

            offset += logical_elements;
            continue;
        }

        std::vector<unsigned char> decompressed(full_chunk_bytes);

        uLongf dest_len = full_chunk_bytes;

        int res = uncompress(
            decompressed.data(),
            &dest_len,
            compressed.data(),
            compressed.size());

        if (res != Z_OK)
            throw std::runtime_error(
                "Zlib decompression failed at chunk "
                + std::to_string(chunk_index));

        // Copy only logical elements
        std::memcpy(
            result.data() + offset,
            decompressed.data(),
            logical_elements * sizeof(float));

        offset += logical_elements;
    }

    return result;
}


std::vector<std::string>
ZarrLoader::load_string_array(
    const std::string& folder_path)
{
    std::ifstream meta_file(folder_path + "/.zarray");
    if (!meta_file)
        throw std::runtime_error("Failed to open .zarray");

    json meta;
    meta_file >> meta;

    size_t total_size = meta["shape"][0];
    size_t chunk_size = meta["chunks"][0];

    std::string dtype = meta["dtype"];
    size_t element_size = std::stoul(dtype.substr(2));

    size_t num_chunks =
        (total_size + chunk_size - 1) / chunk_size;

    std::vector<std::string> result(total_size);

    size_t offset = 0;

    for (size_t chunk_index = 0;
         chunk_index < num_chunks;
         ++chunk_index)
    {
        std::string chunk_path =
            folder_path + "/" +
            std::to_string(chunk_index);

        std::ifstream chunk_file(chunk_path,
                                 std::ios::binary);

        size_t remaining = total_size - offset;
        size_t logical_elements =
            std::min(chunk_size, remaining);

        size_t full_chunk_bytes =
            chunk_size * element_size;

        if (!chunk_file)
        {
            for (size_t i = 0; i < logical_elements; ++i)
                result[offset + i] = "";

            offset += logical_elements;
            continue;
        }

        std::vector<uint8_t> compressed(
            (std::istreambuf_iterator<char>(chunk_file)),
            std::istreambuf_iterator<char>());

        if (compressed.empty())
        {
            for (size_t i = 0; i < logical_elements; ++i)
                result[offset + i] = "";

            offset += logical_elements;
            continue;
        }

        std::vector<uint8_t> decompressed(full_chunk_bytes);

        uLongf dest_len = full_chunk_bytes;

        int res = uncompress(
            decompressed.data(),
            &dest_len,
            compressed.data(),
            compressed.size());

        if (res != Z_OK)
            throw std::runtime_error(
                "Zlib decompression failed at chunk "
                + std::to_string(chunk_index));

        for (size_t i = 0; i < logical_elements; ++i)
        {
            const char* ptr =
                reinterpret_cast<const char*>(
                    decompressed.data() + i * element_size);

            std::string s(ptr, element_size);

            size_t null_pos = s.find('\0');
            if (null_pos != std::string::npos)
                s.resize(null_pos);

            result[offset + i] = s;
        }

        offset += logical_elements;
    }

    return result;
}
