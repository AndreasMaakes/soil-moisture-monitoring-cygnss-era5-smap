import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import cartopy.io.img_tiles as cimgt

# Optional: Use satellite imagery if internet access is available
class ESRIImagery(cimgt.GoogleTiles):
    def __init__(self):
        super().__init__()
        self.tile_size = 256
        self.max_zoom = 19
        self.min_zoom = 1

    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# ISMN coordinates
latitudes = [7.74029, 6.66462, 8.86232, 10.91320, 8.24704, 6.23025, 6.22984, 9.40083, 9.88575, 10.42496,
             11.06886, 10.67041, 6.34518, 7.78199, 6.69099, 8.47606, 10.88337, 7.41301, 9.49418, 10.50105, 7.75559]
longitudes = [-0.97168, -3.09725, 0.05191, -0.81012, -2.25272, -0.34655, -0.34698, -1.00191, -2.46211, -2.55448,
              -0.11478, -2.46477, -2.82625, -0.21264, -1.51909, -0.02853, -1.07111, -2.47607, -2.43478, -1.96855, -2.10156]

# CYGNSS bounding box
min_lon, max_lon = -3.2, 0.2
min_lat, max_lat = 6, 11.2

# Create map
tiler = ESRIImagery()
fig = plt.figure(figsize=(10, 6.7))  # Consistent figsize
ax = plt.axes(projection=tiler.crs)
ax.set_extent([-6, 3.7, 3.9, 12], crs=ccrs.PlateCarree())
ax.set_aspect('auto', adjustable='box')
ax.add_image(tiler, 8)
ax.add_feature(cfeature.BORDERS, edgecolor='#cfd4c0', linewidth=1.25)

# Plot stations
ax.scatter(longitudes, latitudes, color='red', s=60, transform=ccrs.PlateCarree(), label='ISMN Stations')

# CYGNSS rectangle
rect = mpatches.Rectangle(
    (min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
    linewidth=7.5, edgecolor="blue", facecolor='none',
    transform=ccrs.PlateCarree(), label='CYGNSS Retrieval Area'
)
ax.add_patch(rect)

# Grid and labels
gl = ax.gridlines(draw_labels=True, linewidth=0)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}

plt.text(0.5, -0.1, "Longitude", fontsize=26, ha='center', transform=ax.transAxes)
plt.text(-0.16, 0.5, "Latitude", fontsize=26, va='center', rotation='vertical', transform=ax.transAxes)

plt.title("ISMN Soil Moisture Stations in Ghana", fontsize=36, pad=20)
plt.legend(fontsize=20)
plt.show()
