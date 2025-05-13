import xarray as xr

# Load both datasets
ds_cvh = xr.open_dataset("data/Vegetation/cvh.nc")
ds_cvl = xr.open_dataset("data/Vegetation/cvl.nc")

# Add the cvl variable to the cvh dataset
ds_merged = ds_cvh.assign(cvl=ds_cvl['cvl'])

# Save the merged dataset to a new file
ds_merged.to_netcdf("merged_cvh_cvl.nc")