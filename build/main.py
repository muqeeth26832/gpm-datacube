import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------- Usage ----------
# python visualize_slice.py mean_rainfall.csv

# if len(sys.argv) < 2:
#     print("Usage: python visualize_slice.py <csv_file>")
#     sys.exit(1)

# filename = sys.argv[1]
filename= "mean_rainfall.csv"
# ---------- Load Data ----------
data = np.loadtxt(filename, delimiter=",")

# ---------- Geographic Parameters ----------
lat_min = 5.0
lat_max = 40.0
lon_min = 65.0
lon_max = 100.0
resolution = 0.25

lat_bins, lon_bins = data.shape

latitudes = np.linspace(lat_min, lat_max, lat_bins)
longitudes = np.linspace(lon_min, lon_max, lon_bins)

lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# ---------- Mask Zero Values (optional) ----------
masked = np.ma.masked_where(data <= 0, data)

# ---------- Plot ----------
plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, alpha=0.1)
ax.add_feature(cfeature.OCEAN, alpha=0.1)

# Use rainfall-style colormap
mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    masked,
    cmap="turbo",  # blue → green → yellow → red
    shading="auto",
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(mesh)
cbar.set_label("Mean Rainfall (mm/hr)")

plt.title(f"Rainfall Map: {filename}")
plt.show()
