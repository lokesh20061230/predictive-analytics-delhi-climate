# Predictive Analytics Using Historical Data (Daily Delhi Climate)

A predictive modeling and time-series forecasting project built on historical daily weather data (`DailyDelhiClimateTest.csv`) to forecast **mean daily temperature (`meantemp`)**.

## Dataset Columns
- `date`: Observation date (`YYYY-MM-DD`)
- `meantemp`: Mean daily temperature (°C) — **Forecast Target**
- `humidity`: Daily humidity (%)
- `wind_speed`: Wind speed (km/h)
- `meanpressure`: Atmospheric pressure (hPa)

## Key Steps Implemented
1. **Data Cleaning & Preprocessing**
   - Parse and sort `date` chronologically, set daily frequency, and handle missing values/duplicates.
   - Detect and fix extreme pressure outliers (e.g., `meanpressure = 59.0` on `2017-01-01`) using IQR bounds and linear interpolation.
2. **Feature Engineering**
   - Lag features (`temp_lag_1`, `temp_lag_2`, `temp_lag_3`, `temp_lag_7`)
   - Rolling window statistics (`temp_roll_mean_7`, `temp_roll_std_7`)
   - Calendar & lagged weather features (`day_of_week`, `day_of_year`, `humidity_lag_1`, `wind_lag_1`, `pressure_lag_1`)
3. **Models Compared**
   - **Linear Regression** (baseline regression with lag & weather features)
   - **Random Forest Regressor** (non-linear ensemble regression)
   - **SARIMA (1,1,1)(1,0,1,7)** (statistical time-series forecasting model)
4. **Evaluation & Visualization**
   - Evaluated on a chronological 80/20 train-test split using **MAE**, **RMSE**, **MAPE (%)**, and **R²**.
   - Generates plots for historical trends, actual vs. predicted comparison, residual analysis, and a 14-day future temperature forecast with 95% confidence intervals.

## How to Run

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `predictive_analytics.ipynb`.
3. Click **Runtime -> Run all** and upload `DailyDelhiClimateTest.csv` when prompted.

### Option 2: Local Python
```bash
pip install -r requirements.txt
python predictive_model.py
```
