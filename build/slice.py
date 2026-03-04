import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("slice_full_sweep.csv")

# THREAD SCALING (tile=32)
plt.figure(figsize=(8,6))
subset = df[(df["mode"]=="tile") & (df["param"]==32)]
sns.lineplot(data=subset, x="threads", y="time_us", hue="size")
plt.title("Thread Scaling (Tile=32)")
plt.ylabel("Time (µs)")
plt.grid()
plt.savefig("thread_scaling.png")
plt.show()

# TILE IMPACT
plt.figure(figsize=(8,6))
subset = df[(df["mode"]=="tile") & (df["threads"]==8)]
sns.lineplot(data=subset, x="param", y="time_us", hue="size")
plt.title("Tile Size Impact (8 Threads)")
plt.xlabel("Tile Size")
plt.grid()
plt.savefig("tile_impact.png")
plt.show()

# CHUNK IMPACT
plt.figure(figsize=(8,6))
subset = df[(df["mode"]=="row") & (df["threads"]==8)]
sns.lineplot(data=subset, x="param", y="time_us", hue="size")
plt.title("Row Chunk Impact (8 Threads)")
plt.xlabel("Chunk Size")
plt.grid()
plt.savefig("chunk_impact.png")
plt.show()

# PROBLEM SIZE SCALING
plt.figure(figsize=(8,6))
subset = df[(df["mode"]=="sequential")]
sns.lineplot(data=subset, x="size", y="time_us")
plt.title("Sequential Scaling vs Size")
plt.grid()
plt.savefig("size_scaling.png")
plt.show()
