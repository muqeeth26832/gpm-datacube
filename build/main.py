import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def visualize_csv(filename: str, output: str | None = None) -> None:
    """
    Visualize a rainfall CSV file as a geographic map.
    
    Args:
        filename: Path to CSV file with rainfall data
        output: If provided, save figure to this path instead of showing
    """
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
    plt.figure(figsize=(12, 9))
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

    cbar = plt.colorbar(mesh, shrink=0.6)
    cbar.set_label("Mean Rainfall (mm/hr)", fontsize=12)

    plt.title(f"Mean Rainfall Map\n(from {filename})", fontsize=14)
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    # Default file
    filename = "mean_rainfall_simple.csv"
    output_file = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    try:
        visualize_csv(filename, output_file)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Usage: python main.py [csv_file] [output_image]")
        print("Example: python main.py mean_rainfall_simple.csv rainfall_map.png")
        sys.exit(1)
