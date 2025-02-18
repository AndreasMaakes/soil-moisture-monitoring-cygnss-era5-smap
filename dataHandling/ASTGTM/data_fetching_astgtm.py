import os
import requests
import xarray as xr
from tqdm import tqdm
import numpy as np
import shutil  # Import shutil to remove directories

username = "andreasmaakes"
password = "Terrengmodell69!"

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    with requests.get(url, stream=True, auth=(username, password)) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_path

def data_fetching_astgtm(min_lat, max_lat, min_lon, max_lon, name):
    # Generate lat/lon ranges
    lats = np.arange(int(min_lat), int(max_lat) + 1)
    lons = np.arange(int(min_lon), int(max_lon) + 1)
    base_url = "https://opendap.cr.usgs.gov/opendap/hyrax/ASTGTM_NC.003/"
    local_dir = "downloaded_ASTGTM"
    os.makedirs(local_dir, exist_ok=True)
    datasets = []
    total_iterations = len(lats) * len(lons)
    
    with tqdm(total=total_iterations, desc="Downloading DEM Tiles", unit="file") as pbar:
        for lat in lats:
            for lon in lons:
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                filename = f"ASTGTMV003_{lat_str}{lon_str}_dem.ncml.dap.nc4"
                url = base_url + filename
                local_path = os.path.join(local_dir, filename)
                try:
                    print(f"\nDownloading {url}")
                    download_file(url, local_path)
                    ds = xr.open_dataset(local_path)
                    datasets.append(ds)
                    print(f"Successfully downloaded and opened {url}")
                except Exception as e:
                    print(f"Failed to download or open {url}: {e}")
                pbar.update(1)
    
    if datasets:
        combined_ds = xr.concat(datasets, dim="lat")

        # Select only numerical variables to prevent dtype issues
        numerical_vars = {var: combined_ds[var] for var in combined_ds.data_vars if np.issubdtype(combined_ds[var].dtype, np.number)}
        numerical_ds = xr.Dataset(numerical_vars, coords=combined_ds.coords)

        # Downsample using a coarsening factor of 3 (from ~30m to ~90m resolution)
        downsampled_ds = numerical_ds.coarsen(lat=3, lon=3, boundary="trim").mean()

        outdir = "data/ASTGM"
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"{name}.nc")

        downsampled_ds.to_netcdf(outfile)
        print(f"\nSaved downsampled dataset to {outfile}")
        
        # Delete downloaded files and remove the directory after merging
        try:
            shutil.rmtree(local_dir)
            print(f"Deleted folder: {local_dir}")
        except Exception as e:
            print(f"Error deleting folder {local_dir}: {e}")
    else:
        print("\nNo valid datasets fetched.")

if __name__ == '__main__':
    data_fetching_astgtm("62", "64", "9", "11", "Trondheim")
