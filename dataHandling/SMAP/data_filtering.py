#This function filters the dataframe geospatially and removes filler values
def data_filtering_SMAP(df, max_lat: float, min_lat: float, max_lon: float, min_lon: float):

    #Geospatial filter, requires the input of minimum and maximum latitudes and longitudes
    df_filtered_spatial = df[(df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) & (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)]

    #Filters out NaN values, which are set to -9999 
    df_filtered_nan_values = df_filtered_spatial[(df_filtered_spatial["soil_moisture"] != -9999)]
    
    return df_filtered_nan_values