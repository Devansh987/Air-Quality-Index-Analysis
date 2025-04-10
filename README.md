# 🌍 Air Quality Index (AQI) Analysis Project

This project involves an in-depth analysis of air quality data across various Indian cities using Python, Pandas, NumPy, Seaborn, and Matplotlib. It includes data cleaning, AQI calculation, visualization, statistical analysis, and outlier detection.

---

## 📁 Dataset

The dataset used is `citywiseAQI_index.csv`, which contains pollutant concentrations like PM2.5, PM10, NO2, CO, SO2, O3, etc., along with AQI values and timestamps for multiple cities.

---

## 📊 Key Features

- ✅ **Data Cleaning & Preprocessing**  
  - Replaced zero values with NaNs for pollutants  
  - Filled missing data with city-wise and overall means  
  - Handled missing dates and standardized date formats

- 📈 **AQI Recalculation**  
  - Computed AQI sub-indices for PM2.5 and PM10  
  - Recalculated missing AQI values  
  - Bucketed AQI values into standard categories

- 📉 **Visualizations**  
  - Line plot: AQI over time  
  - Bar chart: Average AQI per city  
  - Heatmap: Monthly AQI variation  
  - Boxplot: AQI distribution & pollutant outliers  
  - Histogram: AQI distribution  
  - Stacked area chart: Pollutants over time  
  - Pie chart: AQI category distribution  
  - Scatter plots: AQI vs pollutants  
  - Correlation heatmap

- 🧪 **Statistical Analysis**  
  - Descriptive statistics for AQI  
  - Kolmogorov–Smirnov test for normality

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scipy`

---

## 📁 Project Structure

```bash
.
├── citywiseAQI_index.csv
├── AQI_analysis.ipynb / AQI_analysis.py
└── README.md
