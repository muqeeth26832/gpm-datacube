import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# Setup
# --------------------------------------------------

CSV_FILE = "benchmark_summary.csv"
OUTPUT_DIR = "nodata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(CSV_FILE)

# Extract operation and implementation
ops = []
impl = []

for name in df["operation"]:
    parts = name.split("_")
    impl.append(parts[-1])
    ops.append("_".join(parts[:-1]))

df["op"] = ops
df["impl"] = impl

# Pivot for easier plotting
pivot = df.pivot(index="op", columns="impl", values="avg_time_seconds")

print("\nPivot table:")
print(pivot)

# --------------------------------------------------
# Time comparison plot
# --------------------------------------------------

plt.figure(figsize=(10,6))

pivot.plot(kind="bar")

plt.ylabel("Time (seconds)")
plt.title("Benchmark Execution Time Comparison")
plt.xticks(rotation=30)
plt.grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/execution_times.png")
plt.close()

# --------------------------------------------------
# Speedup plots
# --------------------------------------------------

speedup_thread = pivot["simple"] / pivot["thread"]
speedup_omp = pivot["simple"] / pivot["omp"]

speedup_df = pd.DataFrame({
    "thread_speedup": speedup_thread,
    "omp_speedup": speedup_omp
})

plt.figure(figsize=(10,6))

speedup_df.plot(kind="bar")

plt.ylabel("Speedup vs Sequential")
plt.title("Parallel Speedup")
plt.xticks(rotation=30)
plt.grid(True, axis="y")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/speedups.png")
plt.close()

# --------------------------------------------------
# Per-operation charts
# --------------------------------------------------

for op in pivot.index:
    plt.figure()

    pivot.loc[op].plot(kind="bar")

    plt.title(f"{op} execution time")
    plt.ylabel("seconds")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{op}_time.png")
    plt.close()

print("\nPlots saved to:", OUTPUT_DIR)
