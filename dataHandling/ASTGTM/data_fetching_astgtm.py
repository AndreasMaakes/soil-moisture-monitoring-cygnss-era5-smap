import os
import requests
import numpy as np
import shutil
import zipfile
import rioxarray as rxr
import xarray as xr
from tqdm import tqdm

# Set your token here
BEARER_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImFuZHJlYXNtYWFrZXMiLCJleHAiOjE3NTI5NTg5NzAsImlhdCI6MTc0Nzc3NDk3MCwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.RT99p1psqgvnnl2ZQbh1eGHUdol7p0jQCNvbqMUFpuy2FGyYUAwQeGGhKcItTmtejohVshZopYjRKeYC2NEBkumpey0bpe8a9tJze5pcX-ZkjObBJWfgqRh7jJToGcWLGL-eJFVM-_eHcW8ouxm9ccuQYPKNVDTMxZ99gL9mypBixDueoTJIzdZJ9MwdCsqsTb9SeGeFN2AYe44E3p08kgw876ttmD6QQ7d8aERHuhSz38kaWa5RyDRLLkIlLncCsdCoOb88HOihmti6qW63U4beLa-MXwS_t9yV2zbuBo84pg45IIjKFpO3QKb3nX51_QqsCW4q_yhf3jWQ5aaKnA"  # Replace with your real token

# Date folder on LP DAAC
LPDAAC_DATE_FOLDER = "2000.03.01"
BASE_URL = f"https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/{LPDAAC_DATE_FOLDER}/"

def download_file_with_token(url, local_path):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def data_fetching_astgtm(min_lat, max_lat, min_lon, max_lon, name):
    lats = np.arange(int(min_lat), int(max_lat) + 1)
    lons = np.arange(int(min_lon), int(max_lon) + 1)
    local_dir = "downloaded_ASTGTM"
    os.makedirs(local_dir, exist_ok=True)
    outdir = "data/ASTGTM"
    os.makedirs(outdir, exist_ok=True)
    temp_files = []

    total_iterations = len(lats) * len(lons)

    with tqdm(total=total_iterations, desc="Downloading DEM Tiles", unit="file") as pbar:
        for lat in lats:
            for lon in lons:
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                zip_filename = f"ASTGTMV003_{lat_str}{lon_str}.zip"
                url = BASE_URL + zip_filename
                zip_path = os.path.join(local_dir, zip_filename)

                try:
                    print(f"\nDownloading {url}")
                    download_file_with_token(url, zip_path)

                    # Extract .tif file
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(local_dir)

                    tif_files = [f for f in os.listdir(local_dir)
                                 if f.endswith(".tif") and lat_str in f and lon_str in f]
                    if not tif_files:
                        raise FileNotFoundError(f"No .tif file found in {zip_path}")
                    tif_path = os.path.join(local_dir, tif_files[0])

                    # Open GeoTIFF using rioxarray
                    ds = rxr.open_rasterio(tif_path).squeeze("band", drop=True)

                    # Downsample to approx. 9 km
                    downsample_factor = 30
                    downsampled_ds = ds.coarsen(y=downsample_factor, x=downsample_factor, boundary="trim").mean()

                    print("Before coarsen:", ds.shape)
                    print("After coarsen:", downsampled_ds.shape)

                    temp_file = os.path.join(outdir, f"{lat_str}{lon_str}_temp.nc")
                    downsampled_ds.to_netcdf(temp_file)
                    temp_files.append(temp_file)

                    print(f"Saved downsampled tile: {temp_file}")

                except Exception as e:
                    print(f"Failed to download or process {url}: {e}")

                pbar.update(1)

    if temp_files:
        datasets = [xr.open_dataset(f) for f in temp_files]
        combined_ds = xr.combine_by_coords(datasets, join="outer")

        print("Final lat range:", combined_ds.y.min().values, combined_ds.y.max().values)
        print("Final lon range:", combined_ds.x.min().values, combined_ds.x.max().values)

        outfile = os.path.join(outdir, f"{name}_downsampled.nc")
        combined_ds.to_netcdf(outfile)
        print(f"\nSaved final merged dataset to {outfile}")

        for f in temp_files:
            os.remove(f)

    shutil.rmtree(local_dir, ignore_errors=True)

# Example usage
#data_fetching_astgtm("-34", "-30", "18", "23", "South_Africa_sharper")
data_fetching_astgtm("24", "29", "65", "75", "Pakistan")
