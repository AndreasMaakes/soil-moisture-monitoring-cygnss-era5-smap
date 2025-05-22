import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.patches as mpatches

# Optional: Satellite tiles
class ESRIImagery(cimgt.GoogleTiles):
    def __init__(self):
        super().__init__()
        self.tile_size = 256
        self.max_zoom = 19
        self.min_zoom = 1

    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

# Coordinates
lats = [-35.32395, -34.65478, -35.10975, -35.00535, -35.38978, -34.72835,
        -34.71943, -35.12493, -34.84262, -35.22750, -35.27202, -35.06960,
        -34.62888, -35.09025, -35.39392, -34.84697, -34.85183, -34.96777]
lons = [147.53480, 146.11028, 145.93553, 146.30988, 147.45720, 146.29317,
        146.02003, 147.49740, 145.86692, 147.48500, 147.42902, 146.16893,
        145.84895, 146.30648, 147.56618, 146.41398, 146.11530, 146.01632]

# CYGNSS bounds
min_lon, max_lon = 145.8, 147.6
min_lat, max_lat = -35.4, -34.6

# Map
tiler = ESRIImagery()
fig = plt.figure(figsize=(10, 6.7))  # Same figsize as Ghana
ax = plt.axes(projection=tiler.crs)
ax.set_extent([143, 152, -39, -33], crs=ccrs.PlateCarree())
ax.set_aspect('auto', adjustable='box')
ax.add_image(tiler, 8)
ax.add_feature(cfeature.BORDERS, edgecolor='#cfd4c0', linewidth=1.25)

# Stations
ax.scatter(lons, lats, color='red', s=60, transform=ccrs.PlateCarree(), label="ISMN Stations")

# CYGNSS box
rect = mpatches.Rectangle(
    (min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
    linewidth=7.5, edgecolor='blue', facecolor='none',
    transform=ccrs.PlateCarree(), label="CYGNSS Retrieval Area"
)
ax.add_patch(rect)

# Grid/labels
gl = ax.gridlines(draw_labels=True, linewidth=0)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 28}
gl.ylabel_style = {'size': 28}

plt.text(0.5, -0.1, "Longitude", fontsize=26, ha='center', transform=ax.transAxes)
plt.text(-0.14, 0.5, "Latitude", fontsize=26, va='center', rotation='vertical', transform=ax.transAxes)

plt.title("ISMN Soil Moisture Stations in Australia", fontsize=36, pad=20)
plt.legend(fontsize=20)
plt.show()
