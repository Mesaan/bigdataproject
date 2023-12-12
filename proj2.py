import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#----------Import and adjust dataset----------

data = pd.read_csv('london_weather.csv')

#Convert date column to datetime
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
print(data.head(10))
#Check the date conversion went well
print(data['date'].min())
print(data['date'].max())

#Get yearly values
year_avg = data.groupby(data['date'].dt.year).mean()
print(year_avg.head(10))


#----------Visualize dataset----------

#Temperature
plt.figure(figsize=(8, 5))
plt.title('Evolution of temperature through the years in London')
plt.plot(year_avg.index, year_avg['max_temp'], label='Max Temperature')
plt.plot(year_avg.index, year_avg['mean_temp'], label='Mean Temperature')
plt.plot(year_avg.index, year_avg['min_temp'], label='Min Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()

#Rain - per year
plt.figure(figsize=(8, 5))
plt.bar(year_avg.index, year_avg['precipitation'], color='blue', alpha=0.7)
plt.title('Average rain per year in London')
plt.xlabel('Year')
plt.ylabel('Precipitation')
plt.show()

#Since figure shows us rainiest year was 2014 :
#Avg rain per month in 2014

#Get monthly values for comparison
data_2014 = data[(data['date'].dt.year == 2014)]
month_avg = data_2014.groupby(data['date'].dt.month).mean()

#Rain - per month (2014)
plt.figure(figsize=(8, 5))
plt.bar(month_avg.index, month_avg['precipitation'], color='blue', alpha=0.7)
plt.title('Average rain per month in 2014 in London')
plt.xlabel('Month')
plt.ylabel('Precipitation')
plt.show()

#Sun
plt.figure(figsize=(8, 5))
plt.bar(year_avg.index, year_avg['sunshine'])
plt.title('Sunshine length per year in London')
plt.xlabel('Year')
plt.ylabel('Sunshine length')
plt.show()

#Days of snow - per year

#Adding column for snow visualisation
snow_threshold = 0.0
data['snow_day'] =  np.where(data['snow_depth'] > snow_threshold, 1, 0)
snow_year_avg = data['snow_day'].groupby(data['date'].dt.year).sum()

#Snow
plt.figure(figsize=(8, 5))
plt.bar(year_avg.index, snow_year_avg)
plt.title('Number of days of snow per year in London')
plt.xlabel('Year')
plt.ylabel('Snow days')
plt.show()
