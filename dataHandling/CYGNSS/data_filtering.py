'''
This function filters the data based on geographical location, quality flag and inicident angle 
quality flags 2, 4, 5, 8, 16, and 17. Also filtered is ddm_snr below a given threshold, and sp_rx_gain between two thresholds.
'''
def data_filtering(df, max_lat: float, min_lat: float, max_lon: float, min_lon: float, inc_angle: float, min_ddm_snr: float, min_sp_rx_gain: float, max_sp_rx_gain: float):
    
    print("Filtering data. Dataframe length before geospatial filtering: ", len(df))
    df_filtered_spatial = df[(df["sp_lon"] >= min_lon) & (df["sp_lon"] <= max_lon) & (df["sp_lat"] >= min_lat) & (df["sp_lat"] <= max_lat)]
    print("Dataframe length after geospatial filtering: ", len(df_filtered_spatial))
    
    print("Filtering data based on inclination angle")
    df_filtered_incidence = df_filtered_spatial[(df_filtered_spatial["sp_inc_angle"] <= inc_angle)]
    print("Dataframe length after inclination angle filtering: ", len(df_filtered_incidence))
    
    print("Filtering data based on ddm_snr")
    df_filtered_ddm_snr = df_filtered_incidence[(df_filtered_incidence["ddm_snr"] >= min_ddm_snr)]
    print("Dataframe length after ddm_snr filtering: ", len(df_filtered_ddm_snr))
    
    print("Filtering data based on sp_rx_gain")
    df_filtered_sp_rx_gain = df_filtered_ddm_snr[(df_filtered_ddm_snr["sp_rx_gain"] >= min_sp_rx_gain) & (df_filtered_ddm_snr["sp_rx_gain"] <= max_sp_rx_gain)]
    print("Dataframe length after sp_rx_gain filtering: ", len(df_filtered_sp_rx_gain))
    
    # Bitmasks to exclude rows with certain quality flags set
    bitmask_exclude = (
        0x00000002 |  # S-Band powered up (qf 2)
        0x00000008 |  # small SC attitude error (qf 4)
        0x00000010 |  # black body DDM (qf 5)
        0x00000080 |  # ddm_is_test_pattern (qf 8)
        0x00008000 |  # direct_signal_in_ddm (qf 15)
        0x00010000    # low_confdence_gps_eirp_estimate (qf 16)
        
    )
    
    # Use bitwise AND to filter out rows with these quality flag bits set
    print("Filtering data based on quality flags")
    df_filtered_qf = df_filtered_sp_rx_gain[(df_filtered_sp_rx_gain["quality_flags"] & bitmask_exclude) == 0]
    print("Dataframe length after quality flag filtering: ", len(df_filtered_qf))
    
    return df_filtered_qf


