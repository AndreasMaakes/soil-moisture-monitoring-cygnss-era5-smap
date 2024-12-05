from import_data import importData
import matplotlib.pyplot as plt
import numpy as np

df_des = importData("Chad-test/Chad-20231212-20231212")
df_jan = importData("Chad-test/Chad-20240112-20240112")
df_feb = importData("Chad-test/Chad-20240212-20240212")
df_mar = importData("Chad-test/Chad-20240312-20240312")
df_apr = importData("Chad-test/Chad-20240412-20240412")
df_may = importData("Chad-test/Chad-20240512-20240512")
df_jun = importData("Chad-test/Chad-20240612-20240612")
df_jul = importData("Chad-test/Chad-20240712-20240712")
df_aug = importData("Chad-test/Chad-20240812-20240812")
df_sep = importData("Chad-test/Chad-20240912-20240912")
df_oct = importData("Chad-test/Chad-20241012-20241012")
df_nov = importData("Chad-test/Chad-20241112-20241112")

def analyseSR(df):
    srs = np.array([])
    
    for df in df:
        sr = np.array(df['sr'])
        srs = np.append(srs, sr)
    average = np.mean(srs)
    return average

average_des = analyseSR(df_des)
average_jan = analyseSR(df_jan)
average_feb = analyseSR(df_feb)
average_mar = analyseSR(df_mar)
average_apr = analyseSR(df_apr)
average_may = analyseSR(df_may)
average_jun = analyseSR(df_jun)
average_jul = analyseSR(df_jul)
average_aug = analyseSR(df_aug)
average_sep = analyseSR(df_sep)
average_oct = analyseSR(df_oct)
average_nov = analyseSR(df_nov)

months = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
averages = [average_des, average_jan, average_feb, average_mar, average_apr, average_may, average_jun, average_jul, average_aug, average_sep, average_oct, average_nov]

plt.plot(months, averages)
plt.xlabel('Month')
plt.ylabel('Average SR')
plt.title('Average SR per month')

plt.show()
