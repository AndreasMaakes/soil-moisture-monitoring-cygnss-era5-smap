from datetime import datetime, timedelta

'''
This function creates a time and date array to easier handle the dates in the data fetching function
'''
def create_dates_array(startDate: str, endDate: str):
    # Convert string dates to datetime objects
    start = datetime.strptime(startDate, "%Y%m%d")
    end = datetime.strptime(endDate, "%Y%m%d")
    
    # Generate all dates in the range
    dates = []
    current_date = start
    while current_date <= end:
        # Append the date in the desired format
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    return dates