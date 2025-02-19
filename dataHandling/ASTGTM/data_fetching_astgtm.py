import os
import requests
import xarray as xr
from tqdm import tqdm
import numpy as np
import shutil

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
    lats = np.arange(int(min_lat), int(max_lat) + 1)
    lons = np.arange(int(min_lon), int(max_lon) + 1)
    base_url = "https://opendap.cr.usgs.gov/opendap/hyrax/ASTGTM_NC.003/"
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
                filename = f"ASTGTMV003_{lat_str}{lon_str}_dem.ncml.dap.nc4"
                print("Here comes the filename: ", filename)
                url = base_url + filename
                local_path = os.path.join(local_dir, filename)

                try:
                    print(f"\nDownloading {url}")
                    download_file(url, local_path)
                    ds = xr.open_dataset(local_path)

                    # Select only numerical variables
                    numerical_vars = {var: ds[var] for var in ds.data_vars if np.issubdtype(ds[var].dtype, np.number)}
                    numerical_ds = xr.Dataset(numerical_vars, coords=ds.coords)

                    # Downsample before saving
                    downsampled_ds = numerical_ds.coarsen(lat=5, lon=5, boundary="trim").mean()

                    print("Before coarsen:", numerical_ds.dims)
                    print("After coarsen:", downsampled_ds.dims)

                    temp_file = os.path.join(outdir, f"{lat_str}{lon_str}_temp.nc")
                    downsampled_ds.to_netcdf(temp_file)
                    temp_files.append(temp_file)

                    print(f"Saved downsampled tile: {temp_file}")
                except Exception as e:
                    print(f"Failed to download or process {url}: {e}")

                pbar.update(1)

    if temp_files:
        # Open datasets lazily and concatenate
        datasets = [xr.open_dataset(f) for f in temp_files]
        combined_ds = xr.combine_by_coords(datasets, join="outer")

        print("Final lat range:", combined_ds.lat.min().values, combined_ds.lat.max().values)
        print("Final lon range:", combined_ds.lon.min().values, combined_ds.lon.max().values)


        outfile = os.path.join(outdir, f"{name}_downsampled.nc")
        combined_ds.to_netcdf(outfile)
        print(f"\nSaved final merged dataset to {outfile}")

        # Clean up temporary files
        for f in temp_files:
            os.remove(f)
        
    # Remove downloaded files
    shutil.rmtree(local_dir, ignore_errors=True)

if __name__ == '__main__':
    data_fetching_astgtm("24", "32", "70", "80", "India")
