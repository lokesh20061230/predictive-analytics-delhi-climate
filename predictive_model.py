"""Predictive analytics pipeline for Daily Delhi Climate historical data."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def load_dataset(csv_path: str = "DailyDelhiClimateTest.csv") -> pd.DataFrame:
    """Load the dataset from common project paths."""
    candidates = [
        Path(csv_path),
        Path("data") / csv_path,
        Path(__file__).resolve().parent / csv_path,
        Path(__file__).resolve().parent / "data" / csv_path,
    ]
    for path in candidates:
        if path.exists():
            print(f"Loaded dataset: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find '{csv_path}'. Place it in the project folder or data/."
    )


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, remove duplicates, fix outliers, and fill daily gaps."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    df = df.sort_values("date").set_index("date").asfreq("D")

    numeric_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace extreme pressure sensor anomalies (e.g. 59.0 hPa) using IQR bounds
    q1 = df["meanpressure"].quantile(0.25)
    q3 = df["meanpressure"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    outlier_mask = (df["meanpressure"] < lower) | (df["meanpressure"] > upper)
    if outlier_mask.any():
        print(f"Replaced {int(outlier_mask.sum())} meanpressure outlier(s) via interpolation.")
        df.loc[outlier_mask, "meanpressure"] = np.nan

    df[numeric_cols] = df[numeric_cols].interpolate(method="time").bfill().ffill()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and calendar features without future leakage."""
    feat = df.copy()
    for lag in (1, 2, 3, 7):
        feat[f"temp_lag_{lag}"] = feat["meantemp"].shift(lag)
    feat["temp_roll_mean_7"] = feat["meantemp"].shift(1).rolling(7).mean()
    feat["temp_roll_std_7"] = feat["meantemp"].shift(1).rolling(7).std()
    feat["humidity_lag_1"] = feat["humidity"].shift(1)
    feat["wind_lag_1"] = feat["wind_speed"].shift(1)
    feat["pressure_lag_1"] = feat["meanpressure"].shift(1)
    feat["day_of_week"] = feat.index.dayofweek
    feat["day_of_year"] = feat.index.dayofyear
    return feat.dropna()


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, name: str) -> dict:
    """Compute MAE, RMSE, MAPE, and R2 metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    non_zero = y_true != 0
    mape = float(
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    )
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


def save_plots(
    df: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    y_test: pd.Series,
    preds: dict[str, np.ndarray],
    future_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate and save exploratory, evaluation, residual, and forecast plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Historical trends
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    cols = [
        ("meantemp", "Mean Temperature (°C)", "tab:red"),
        ("humidity", "Humidity (%)", "tab:blue"),
        ("wind_speed", "Wind Speed (km/h)", "tab:green"),
        ("meanpressure", "Mean Pressure (hPa)", "tab:purple"),
    ]
    for ax, (col, title, color) in zip(axes.flat, cols):
        ax.plot(df.index, df[col], color=color, linewidth=1.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Historical Climate Trends (Cleaned Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "01_historical_trends.png", dpi=150)
    plt.close()

    # 2. Actual vs Predicted
    plt.figure(figsize=(12, 5))
    plt.plot(test_index, y_test, label="Actual", color="black", linewidth=2.2)
    for model_name, y_hat in preds.items():
        plt.plot(test_index, y_hat, linestyle="--", linewidth=1.8, label=model_name)
    plt.title("Actual vs Predicted Mean Temperature (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "02_actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3. Residual distribution for best regression/time-series model
    best_name = min(
        preds.keys(), key=lambda m: np.sqrt(mean_squared_error(y_test, preds[m]))
    )
    residuals = y_test.values - preds[best_name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(test_index, residuals, marker="o", linestyle="-", color="tab:orange")
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set_title(f"Residuals Over Time ({best_name})")
    axes[0].tick_params(axis="x", rotation=30)
    sns.histplot(residuals, kde=True, ax=axes[1], color="tab:orange")
    axes[1].set_title(f"Residual Distribution ({best_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "03_residual_analysis.png", dpi=150)
    plt.close()

    # 4. Future forecast
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["meantemp"], label="Historical Temperature", color="tab:blue")
    plt.plot(
        future_df.index,
        future_df["forecast"],
        label="14-Day ARIMA Forecast",
        color="tab:red",
        linewidth=2,
    )
    plt.fill_between(
        future_df.index,
        future_df["lower_ci"],
        future_df["upper_ci"],
        color="tab:red",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.title("14-Day Future Temperature Forecast")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature (°C)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "04_future_forecast.png", dpi=150)
    plt.close()


def main() -> None:
    raw_df = load_dataset("DailyDelhiClimateTest.csv")
    clean_df = clean_and_preprocess(raw_df)

    feat_df = build_features(clean_df)
    feature_cols = [c for c in feat_df.columns if c not in clean_df.columns]

    split_idx = int(len(feat_df) * 0.8)
    train_df = feat_df.iloc[:split_idx]
    test_df = feat_df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["meantemp"]
    X_test, y_test = test_df[feature_cols], test_df["meantemp"]

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # Model 2: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Model 3: ARIMA / SARIMAX on the target series aligned to the same train/test split
    series_train = clean_df.loc[: train_df.index[-1], "meantemp"]
    sarima = SARIMAX(
        series_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_pred = sarima.forecast(steps=len(y_test)).values

    preds = {
        "Linear Regression": lr_pred,
        "Random Forest": rf_pred,
        "SARIMA (1,1,1)(1,0,1,7)": sarima_pred,
    }
    results = pd.DataFrame(
        [evaluate_predictions(y_test, p, name) for name, p in preds.items()]
    ).sort_values("RMSE")
    print("\nModel Accuracy Comparison:")
    print(results.to_string(index=False))

    # Refit SARIMA on full cleaned history to forecast next 14 days
    full_sarima = SARIMAX(
        clean_df["meantemp"],
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecast_res = full_sarima.get_forecast(steps=14)
    future_idx = pd.date_range(
        start=clean_df.index[-1] + pd.Timedelta(days=1), periods=14, freq="D"
    )
    conf_int = forecast_res.conf_int()
    future_df = pd.DataFrame(
        {
            "forecast": forecast_res.predicted_mean.values,
            "lower_ci": conf_int.iloc[:, 0].values,
            "upper_ci": conf_int.iloc[:, 1].values,
        },
        index=future_idx,
    )
    print("\nNext 14 Days Temperature Forecast:")
    print(future_df.round(2))

    out_dir = Path(__file__).resolve().parent / "outputs"
    save_plots(clean_df, y_test.index, y_test, preds, future_df, out_dir)
    print(f"\nSaved charts to: {out_dir}")


if __name__ == "__main__":
    main()
