import cdsapi
import os
import xarray as xr
from create_dates_array import create_dates_array

def data_fetching_era5(timeSeries: bool,
                       startDate: str,
                       endDate: str,
                       min_lat, max_lat,
                       min_lon, max_lon,
                       name: str,
                       basePath: str = "data/ERA5"):
    """
    1) Downloads ERA5-Land soil moisture and land_sea_mask via two separate .nc downloads.
    2) Opens, interpolates the mask onto the soil-moisture grid, merges into one Dataset.
    3) Writes out a single NetCDF and removes the two intermediate files.
    """

    # --- 1) Dates & output paths ---
    year, month, days = create_dates_array(startDate, endDate, "era5")
    datestr = f"{year}{month}{days[0]}_{year}{month}{days[-1]}"

    if timeSeries:
        out_dir = os.path.join(basePath, "ERA5")
        prefix  = f"ERA5_{datestr}"
    else:
        out_dir = os.path.join(basePath, name)
        prefix  = f"ERA5_{name}_{datestr}"

    os.makedirs(out_dir, exist_ok=True)

    # --- 2) CDS client and time list ---
    client = cdsapi.Client()
    times  = [f"{h:02d}:00" for h in range(24)]
    area   = [max_lat, min_lon, min_lat, max_lon]

    # --- 3) Download each variable into its own file ---
    vars_to_fetch = {
        "volumetric_soil_water_layer_1": f"{prefix}_swvl1.nc",
        "land_sea_mask":                f"{prefix}_lsm.nc"
    }

    for var, fname in vars_to_fetch.items():
        req = {
            "variable":        [var],
            "year":            [year],
            "month":           [month],
            "day":             days,
            "time":            times,
            "data_format":     "netcdf",
            "download_format": "unarchived",
            "area":            area,
        }
        target_path = os.path.join(out_dir, fname)
        print(f"Downloading {var} → {target_path}")
        client.retrieve("reanalysis-era5-land", req).download(target_path)

    # --- 4) Open both files ---
    swvl1_path = os.path.join(out_dir, vars_to_fetch["volumetric_soil_water_layer_1"])
    lsm_path   = os.path.join(out_dir, vars_to_fetch["land_sea_mask"])

    ds_sw  = xr.open_dataset(swvl1_path, engine="netcdf4")
    ds_lsm = xr.open_dataset(lsm_path,   engine="netcdf4")

    # --- 5) Interpolate mask onto the soil-moisture grid ---
    ds_lsm2 = ds_lsm.interp(
        latitude  = ds_sw.latitude,
        longitude = ds_sw.longitude,
        method    = "nearest"
    )

    # --- 6) Merge and write final file ---
    merged_path = os.path.join(out_dir, f"{prefix}.nc")
    xr.merge([ds_sw, ds_lsm2]).to_netcdf(merged_path)
    print(f"Wrote merged file → {merged_path}")
    
    # -- - 6.5) Close datasets to free up memory ---
    ds_sw.close()
    ds_lsm.close()
    ds_lsm2.close() 

    # --- 7) Clean up intermediate files ---
    os.remove(swvl1_path)
    os.remove(lsm_path)
    print("Deleted intermediate files.")
