#pragma once
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    struct TimingRecord {
        std::string operation;
        double time_seconds;
    };

private:
    std::vector<TimingRecord> records;

public:
    // Start timing
    static TimePoint now() {
        return Clock::now();
    }

    // Measure duration between two time points
    static double elapsed(TimePoint start, TimePoint end) {
        return std::chrono::duration<double>(end - start).count();
    }

    // Record a timing measurement
    void record(const std::string& operation, double time_seconds) {
        records.push_back({operation, time_seconds});
    }

    // Calculate average time for a specific operation
    double average(const std::string& operation) const {
        double sum = 0.0;
        size_t count = 0;

        for (const auto& rec : records) {
            if (rec.operation == operation) {
                sum += rec.time_seconds;
                ++count;
            }
        }

        return count > 0 ? sum / static_cast<double>(count) : 0.0;
    }

    // Get count of recordings for an operation
    size_t count(const std::string& operation) const {
        size_t c = 0;
        for (const auto& rec : records) {
            if (rec.operation == operation) ++c;
        }
        return c;
    }

    // Export all records to CSV
    void export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        file << "operation,time_seconds\n";

        for (const auto& rec : records) {
            file << rec.operation << "," 
                 << std::fixed << std::setprecision(6) 
                 << rec.time_seconds << "\n";
        }

        file.close();
    }

    // Export summary (averages) to CSV
    void export_summary_csv(const std::string& filename) const {
        // Collect unique operations
        std::vector<std::string> ops;
        for (const auto& rec : records) {
            bool found = false;
            for (const auto& op : ops) {
                if (op == rec.operation) {
                    found = true;
                    break;
                }
            }
            if (!found) ops.push_back(rec.operation);
        }

        std::ofstream file(filename);
        file << "operation,avg_time_seconds,count\n";

        for (const auto& op : ops) {
            file << op << ","
                 << std::fixed << std::setprecision(6)
                 << average(op) << ","
                 << count(op) << "\n";
        }

        file.close();
    }

    // Clear all records
    void clear() {
        records.clear();
    }
};

// Helper macro for easy timing
#define TIME_OPERATION(timer, op_name, code) \
    do { \
        auto _start = Timer::now(); \
        code; \
        auto _end = Timer::now(); \
        (timer).record(op_name, Timer::elapsed(_start, _end)); \
    } while(0)
