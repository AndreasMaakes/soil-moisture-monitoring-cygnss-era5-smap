import numpy as np
import xarray as xr
from pydap.client import open_url
from pydap.cas.urs import setup_session
from tqdm import tqdm

# Credentials (replace with secure handling)
username = "andreasmaakes"
password = "Terrengmodell69!"

def data_fetching_astgtm(min_lat, max_lat, min_lon, max_lon, name):

    # Generate lat/lon ranges (assuming north and east hemispheres)
    lats = np.arange(int(min_lat), int(max_lat) + 1)
    lons = np.arange(int(min_lon), int(max_lon) + 1)

    base_url = "https://opendap.cr.usgs.gov/opendap/hyrax/ASTGTM_NC.003/"

    datasets = []

    session = setup_session(None, None)  # Authenticate once
    
    
    total_iterations = len(lats) * len(lons)

    with tqdm(total=total_iterations, desc="Progress: ", unit="files", colour = "green") as pbar:
        for lat in lats:
            for lon in lons:
                lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
                lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
                filename = f"ASTGTMV003_{lat_str}{lon_str}_dem.ncml.dap.nc4"
                url = base_url + filename
                print(url)
                try:
                    dataset = open_url(url, session=session)

                    # Extract required variables
                    lon_data = np.array(dataset["lon"][:])
                    lat_data = np.array(dataset["lat"][:])
                    dem_data = np.array(dataset["ASTER_GDEM_DEM"][:])

                    # Convert to Xarray Dataset
                    ds = xr.Dataset(
                        {
                            "ASTER_GDEM_DEM": (["lat", "lon"], dem_data)
                        },
                        coords={
                            "lat": lat_data,
                            "lon": lon_data
                        }
                    )

                    datasets.append(ds)

                    print(f"Successfully fetched {url}")

                except Exception as e:
                    print(f"Failed to access {url}: {e}")

        # Merge all datasets into a single dataset
        if datasets:
            combined_ds = xr.concat(datasets, dim="lat")
            basePath = "data/ASTGM/"
            combined_ds.to_netcdf(f'{basePath}/{name}.nc')
            print(f"Saved merged dataset to {basePath}/{name}.nc")
        else:
            print("No valid datasets fetched.")

# Example usage
data_fetching_astgtm("62", "64", "9", "11", "Trondheim")
