from datetime import datetime, timedelta

'''
This function creates a time and date array to easier handle the dates in the data fetching function.
It takes in the start and end date as strings and the data source as a string, since the format of the three data sources are different.


# Example usage:
smap_dates = create_dates_array("20240704", "20240705", "smap")
era5_dates = create_dates_array("20240701", "20240703", "era5")
cygnss_dates = create_dates_array("20240701", "20240703", "cygnss")

print("SMAP Dates:", smap_dates)      # Returns ['2024-07-04', '2024-07-05']
print("ERA5 Dates:", era5_dates)      # Returns ('2024', '07', ['01', '02', '03'])
print("CYGNSS Dates:", cygnss_dates)  # Returns ['20240701', '20240702', '20240703']

'''

def create_dates_array(startDate: str, endDate: str, dataSource: str):
    # Convert string dates to datetime objects
    start = datetime.strptime(startDate, "%Y%m%d")
    end = datetime.strptime(endDate, "%Y%m%d")
    
    # Generate all dates in the range
    dates = []
    current_date = start

    while current_date <= end:
        if dataSource.lower() == "smap":
            # Format: "YYYY-MM-DD"
            dates.append(current_date.strftime("%Y-%m-%d"))
        elif dataSource.lower() == "era5":
            # Format: ("YYYY", "MM", ["DD", "DD", ...])
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            dates.append(current_date.strftime("%d"))  # Collect day separately
        elif dataSource.lower() == "cygnss":
            # Format: "YYYYMMDD"
            dates.append(current_date.strftime("%Y%m%d"))
        else:
            raise ValueError(f"Unsupported data source: {dataSource}")
        
        current_date += timedelta(days=1)

    # For ERA5, return a tuple (year, month, [days])
    if dataSource.lower() == "era5":
        return (start.strftime("%Y"), start.strftime("%m"), dates)  # (Year, Month, [Day List])
    
    return dates  # e.g., ["2024-07-04", "2024-07-05"] for SMAP or ["20240701", "20240703"] for CYGNSS

