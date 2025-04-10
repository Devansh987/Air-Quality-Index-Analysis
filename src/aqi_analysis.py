import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest

df = pd.read_csv('citywiseAQI_index.csv')
df.head(10)
df.tail(10)
df.info()
df.describe()

# Data Processing and Cleaning Using Numpy and Pandas
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.head(30)


for col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zero values")

cols_to_fix = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].mean())


df['Month'] = df['Date'].dt.month
df['PM2.5'] = df.groupby('City')['PM2.5'].transform(lambda x: x.fillna(x.mean()))


pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']
for col in pollutants:   
    df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.mean()))
    df[col] = df[col].fillna(df[col].mean())


missing_summary = df.groupby('City')[pollutants].apply(lambda x: x.isna().mean())
full_missing = missing_summary[missing_summary == 1.0].dropna(how='all')
print("Cities with 100% missing for any pollutant:")
print(full_missing)


def compute_pm25_subindex(pm25):
    if pm25 <= 30:
        return (50 / 30) * pm25
    elif pm25 <= 60:
        return 50 + ((100 - 51) / (60 - 31)) * (pm25 - 31)
    elif pm25 <= 90:
        return 101 + ((200 - 101) / (90 - 61)) * (pm25 - 61)
    elif pm25 <= 120:
        return 201 + ((300 - 201) / (120 - 91)) * (pm25 - 91)
    elif pm25 <= 250:
        return 301 + ((400 - 301) / (250 - 121)) * (pm25 - 121)
    else:
        return 401 + ((500 - 401) / (430 - 251)) * (pm25 - 251)

def compute_pm10_subindex(pm10):
    if pm10 <= 50:
        return pm10
    elif pm10 <= 100:
        return 51 + ((100 - 51) / (100 - 51)) * (pm10 - 51)
    elif pm10 <= 250:
        return 101 + ((200 - 101) / (250 - 101)) * (pm10 - 101)
    elif pm10 <= 350:
        return 201 + ((300 - 201) / (350 - 251)) * (pm10 - 251)
    elif pm10 <= 430:
        return 301 + ((400 - 301) / (430 - 351)) * (pm10 - 351)
    else:
        return 401 + ((500 - 401) / (600 - 431)) * (pm10 - 431)
df['PM2.5_SubIndex'] = df['PM2.5'].apply(lambda x: compute_pm25_subindex(x) if pd.notnull(x) else np.nan)
df['PM10_SubIndex']  = df['PM10'].apply(lambda x: compute_pm10_subindex(x) if pd.notnull(x) else np.nan)
df['AQI_Recalc'] = df[['PM2.5_SubIndex', 'PM10_SubIndex']].max(axis=1)

df['AQI'] = df['AQI'].fillna(df['AQI_Recalc'])
df.drop(columns=['PM2.5_SubIndex', 'PM10_SubIndex', 'AQI_Recalc'], inplace=True)

def aqi_bucket(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Satisfactory'
    elif value <= 200:
        return 'Moderate'
    elif value <= 300:
        return 'Poor'
    elif value <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

df['AQI_Bucket'] = df['AQI'].apply(aqi_bucket)

#Data Visulization using Matplotlib and Seaborn
#Line Plot 
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df_sorted = df.sort_values(by='Date')
plt.figure(figsize=(14,6))
sns.lineplot(data=df_sorted, x='Date', y='AQI', color='orange')
plt.title('AQI Over Time')
plt.xlabel('Date')
plt.ylabel('Air Quality Index')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Bar Plot
city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False)
city_df = city_aqi.reset_index()
city_df.columns = ['City', 'Average_AQI']
plt.figure(figsize=(12,6))
sns.barplot(data=city_df, x='City', y='Average_AQI', hue='City', palette='viridis', dodge=False, legend=False)
plt.title('Average AQI by City')
plt.xticks(rotation=45)
plt.ylabel('Average AQI')
plt.tight_layout()
plt.show()

#HeatMap

df['Month'] = df['Date'].dt.month_name()
pivot_data = df.pivot_table(index='City', columns='Month', values='AQI', aggfunc='mean')
plt.figure(figsize=(14,8))
sns.heatmap(pivot_data, cmap='Spectral', annot=True, fmt='.1f', linewidths=0.5)
plt.title('Average Monthly AQI by City')
plt.ylabel('City')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#BoxPlot

plt.figure(figsize=(14,6))
sns.boxplot(data=df, x='City', y='AQI')
plt.title('AQI Distribution by City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Histrogram
plt.figure(figsize=(10,5))
sns.histplot(df['AQI'], bins=50, kde=True, color='teal')
plt.title('Distribution of AQI')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()

#Top 10 Cities With Worst Aqi using Plot
city_mean_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
city_mean_aqi.plot(kind='bar', color='coral')
plt.title('Average AQI for Top 10 Cities')
plt.ylabel('AQI')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

#Top 10 Cities With Best Aqi using Plot
valid_aqi = df.dropna(subset=['AQI'])
city_mean_aqi = valid_aqi.groupby('City')['AQI'].mean().sort_values()
cleanest_city = city_mean_aqi.idxmin()
lowest_aqi_value = city_mean_aqi.min()
print(f"🏙️ Cleanest city based on AQI is: **{cleanest_city}** with an average AQI of {lowest_aqi_value:.2f}")
plt.figure(figsize=(10, 5))
city_mean_aqi.head(10).plot(kind='bar', color='seagreen')
plt.title('Top 10 Cities with Cleanest Air (Lowest Avg AQI)')
plt.ylabel('Average AQI')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Stacked Area Chart For Pollutant Over Time
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
df_plot = df[['Date'] + pollutants].copy()
df_plot = df_plot.dropna(subset=pollutants)
df_plot = df_plot.groupby('Date').mean().reset_index()
plt.figure(figsize=(14, 6))
plt.stackplot(df_plot['Date'], [df_plot[p] for p in pollutants], labels=pollutants, alpha=0.8)
plt.legend(loc='upper left')
plt.title('Stacked Area Chart: Pollutant Levels Over Time')
plt.xlabel('Date')
plt.ylabel('Pollutant Concentration')
plt.tight_layout()
plt.show()

#Aqi Category Ditribution using Pie chart
bucket_counts = df['AQI_Bucket'].value_counts()
plt.figure(figsize=(8, 8))
colors = sns.color_palette('Set2', len(bucket_counts))
plt.pie(bucket_counts, labels=bucket_counts.index, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('AQI Category Distribution')
plt.axis('equal')
plt.show()

#Correlation HeatMap for Air Quality Parameters

plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Air Quality Parameters')
plt.show()

#BoxPlot for Outlier Detection in Pollutants
pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
plt.figure(figsize=(15, 6))
df[pollutants].boxplot()
plt.title('Boxplot for Outlier Detection in Pollutants')
plt.ylabel('Concentration')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Aqi vs Pollutants Graph
for pollutant in ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x=pollutant, y='AQI')
    plt.title(f'AQI vs {pollutant}')
    plt.grid(True)
    plt.show()

#Descriptive Analysis
aqi_stats = df['AQI'].describe()
print(aqi_stats)


#K-s Test
aqi_data = df['AQI'].dropna()
standardized_data = (aqi_data - aqi_data.mean()) / aqi_data.std()


stat, p_value = kstest(standardized_data, 'norm')
print(f'K-S Test Statistic: {stat:.4f}, p-value: {p_value:.4f}')


