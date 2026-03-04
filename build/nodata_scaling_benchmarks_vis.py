import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

EXEC = "./gpmcube"
LAT = 140
LON = 140

T_VALUES = [500, 1000, 2000, 3000, 4000, 5000]

OUTPUT_DIR = "nodata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_results = []

print("Running scaling benchmarks...\n")

for T in T_VALUES:

    print(f"Running T = {T}")

    subprocess.run([EXEC, str(T), str(LAT), str(LON)])

    df = pd.read_csv("benchmark_summary.csv")

    df["T"] = T

    all_results.append(df)

# --------------------------------------------------
# Combine results
# --------------------------------------------------

data = pd.concat(all_results, ignore_index=True)

data.to_csv(f"{OUTPUT_DIR}/scaling_results.csv", index=False)

# --------------------------------------------------
# Split operation + implementation
# --------------------------------------------------

ops = []
impl = []

for name in data["operation"]:
    parts = name.split("_")
    impl.append(parts[-1])
    ops.append("_".join(parts[:-1]))

data["op"] = ops
data["impl"] = impl

# --------------------------------------------------
# Plot scaling for each operation
# --------------------------------------------------

operations = data["op"].unique()

for op in operations:

    subset = data[data["op"] == op]

    plt.figure(figsize=(8,5))

    for impl_name in ["simple","thread","omp"]:

        impl_data = subset[subset["impl"] == impl_name]

        plt.plot(
            impl_data["T"],
            impl_data["avg_time_seconds"],
            marker="o",
            label=impl_name
        )

    plt.title(f"{op} scaling vs time dimension")
    plt.xlabel("Time dimension (T)")
    plt.ylabel("Execution time (seconds)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{op}_scaling.png")
    plt.close()

# --------------------------------------------------
# Speedup plots (fixed)
# --------------------------------------------------

for op in operations:

    subset = data[data["op"] == op]

    pivot = subset.pivot(
        index="T",
        columns="impl",
        values="avg_time_seconds"
    ).sort_index()

    if {"simple","thread","omp"}.issubset(pivot.columns):

        thread_speedup = pivot["simple"] / pivot["thread"]
        omp_speedup = pivot["simple"] / pivot["omp"]

        plt.figure(figsize=(8,5))

        plt.plot(
            pivot.index,
            thread_speedup,
            marker="o",
            label="thread speedup"
        )

        plt.plot(
            pivot.index,
            omp_speedup,
            marker="o",
            label="omp speedup"
        )

        plt.title(f"{op} speedup vs time dimension")
        plt.xlabel("Time dimension (T)")
        plt.ylabel("Speedup")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(f"{OUTPUT_DIR}/{op}_speedup.png")
        plt.close()
