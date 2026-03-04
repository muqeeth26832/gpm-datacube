#!/usr/bin/env python3
"""
Benchmark Visualization Tool

Compares:
    - Datacube (flat vector)
    - SimpleCube (sequential)
    - SimpleCube (std::thread)
    - SimpleCube (OpenMP)

Produces:
    1. Runtime comparison chart
    2. Speedup comparison chart
    3. Combined dashboard

Designed for systems benchmarking / OLAP evaluation.
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Load CSV
# ------------------------------------------------------------

def load_data(filename: str):

    ops = []
    datacube = []
    simple = []
    thread = []
    omp = []

    s_simple = []
    s_thread = []
    s_omp = []

    with open(filename) as f:

        reader = csv.DictReader(f)

        for r in reader:

            ops.append(r["operation"])

            datacube.append(float(r["datacube_time"]))
            simple.append(float(r["simple_time"]))
            thread.append(float(r["thread_time"]))
            omp.append(float(r["omp_time"]))

            s_simple.append(float(r["speedup_simple"]))
            s_thread.append(float(r["speedup_thread"]))
            s_omp.append(float(r["speedup_omp"]))

    return {
        "ops": ops,
        "datacube": np.array(datacube),
        "simple": np.array(simple),
        "thread": np.array(thread),
        "omp": np.array(omp),
        "s_simple": np.array(s_simple),
        "s_thread": np.array(s_thread),
        "s_omp": np.array(s_omp),
    }


# ------------------------------------------------------------
# Runtime Plot
# ------------------------------------------------------------

def plot_runtime(data, outfile):

    ops = data["ops"]
    x = np.arange(len(ops))
    w = 0.2

    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(x-1.5*w, data["datacube"], w, label="Datacube", color="#e74c3c")
    ax.bar(x-0.5*w, data["simple"], w, label="SimpleCube (seq)", color="#3498db")
    ax.bar(x+0.5*w, data["thread"], w, label="SimpleCube (thread)", color="#2ecc71")
    ax.bar(x+1.5*w, data["omp"], w, label="SimpleCube (OpenMP)", color="#9b59b6")

    ax.set_yscale("log")  # important for benchmarks

    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Execution Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=15)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved runtime plot → {outfile}")


# ------------------------------------------------------------
# Speedup Plot
# ------------------------------------------------------------

def plot_speedup(data, outfile):

    ops = data["ops"]
    x = np.arange(len(ops))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12,6))

    ax.bar(x-w, data["s_simple"], w, label="SimpleCube", color="#3498db")
    ax.bar(x, data["s_thread"], w, label="Thread", color="#2ecc71")
    ax.bar(x+w, data["s_omp"], w, label="OpenMP", color="#9b59b6")

    ax.axhline(1.0, linestyle="--", color="red", linewidth=2)

    ax.set_ylabel("Speedup vs Datacube")
    ax.set_title("Speedup Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=15)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"Saved speedup plot → {outfile}")


# ------------------------------------------------------------
# Combined Dashboard
# ------------------------------------------------------------

def plot_dashboard(data, outfile):

    ops = data["ops"]
    x = np.arange(len(ops))
    w = 0.2

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))

    # runtime
    ax1.bar(x-1.5*w,data["datacube"],w,label="Datacube")
    ax1.bar(x-0.5*w,data["simple"],w,label="Simple")
    ax1.bar(x+0.5*w,data["thread"],w,label="Thread")
    ax1.bar(x+1.5*w,data["omp"],w,label="OpenMP")

    ax1.set_yscale("log")
    ax1.set_title("Runtime Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ops,rotation=15)
    ax1.set_ylabel("Seconds")
    ax1.grid(True,axis="y",linestyle="--",alpha=0.4)
    ax1.legend()

    # speedup
    w2 = 0.25
    ax2.bar(x-w2,data["s_simple"],w2,label="Simple")
    ax2.bar(x,data["s_thread"],w2,label="Thread")
    ax2.bar(x+w2,data["s_omp"],w2,label="OpenMP")

    ax2.axhline(1,color="red",linestyle="--")

    ax2.set_title("Speedup vs Datacube")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ops,rotation=15)
    ax2.set_ylabel("Speedup")
    ax2.grid(True,axis="y",linestyle="--",alpha=0.4)
    ax2.legend()

    plt.suptitle("OLAP Datacube Benchmark")

    plt.tight_layout()
    plt.savefig(outfile,dpi=200)

    print(f"Saved dashboard → {outfile}")


# ------------------------------------------------------------
# Text Summary
# ------------------------------------------------------------

def print_summary(data):

    print("\nBenchmark Summary\n")
    print(f"{'Operation':<20}{'Datacube':<12}{'Seq':<12}{'Thread':<12}{'OMP':<12}")

    for i,op in enumerate(data["ops"]):

        print(
            f"{op:<20}"
            f"{data['datacube'][i]:<12.6f}"
            f"{data['simple'][i]:<12.6f}"
            f"{data['thread'][i]:<12.6f}"
            f"{data['omp'][i]:<12.6f}"
        )

    print()

    print("Average Speedups")

    print(f"Simple  : {np.mean(data['s_simple']):.2f}x")
    print(f"Thread  : {np.mean(data['s_thread']):.2f}x")
    print(f"OpenMP  : {np.mean(data['s_omp']):.2f}x")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    file = "benchmark_results.csv"

    if len(sys.argv) > 1:
        file = sys.argv[1]

    if not Path(file).exists():
        print("Benchmark file not found.")
        print("Run: ./gpmcube → option 3")
        return

    data = load_data(file)

    print_summary(data)

    plot_runtime(data,"vb/runtime_comparison.png")
    plot_speedup(data,"vb/peedup_comparison.png")
    plot_dashboard(data,"vb/benchmark_dashboard.png")

    print("\nAll plots generated.\n")


if __name__ == "__main__":
    main()
