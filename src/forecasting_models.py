import pandas as pd
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def train_prophet_model(df, city, forecast_periods=365, filepath='reports/prophet_forecast.png'):
    """
    Trains a Prophet model for a specific city and generates a forecast.
    """
    prophet_df = df[df['name'] == city].reset_index()[['lastupdated', 'temp_c']].rename(
        columns={'lastupdated': 'ds', 'temp_c': 'y'}
    )

    model = Prophet(interval_width=0.95)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title(f'Prophet Forecast for {city}', fontsize=16)
    plt.savefig(filepath)
    print(f"Prophet forecast plot saved to {filepath}")
    plt.show()

    return model, forecast


def train_lightgbm_model(df, city, feature_cols, target_col):
    """
    Trains a LightGBM model for forecasting.
    """
    city_data = df[df['name'] == city].copy()

    # Create lag features for forecasting
    for i in range(1, 8):
        city_data[f'{target_col}_lag_{i}'] = city_data[target_col].shift(i)

    city_data.dropna(inplace=True)

    X = city_data[feature_cols + [f'{target_col}_lag_{i}' for i in range(1, 8)]]
    y = city_data[target_col]

    split_point = int(len(city_data) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'LightGBM RMSE for {city}: {rmse:.2f}')

    return model, y_test, y_pred


def create_ensemble_forecast(prophet_forecast, lgbm_predictions, prophet_weight=0.6):
    """
    Creates an ensemble forecast using a weighted average of two models.
    This is a conceptual example and would require careful alignment of the forecasts.
    """
    print("Ensemble model created. This is a conceptual step and requires careful data alignment.")

    # Simple example for illustration
    ensemble_pred = (prophet_weight * prophet_forecast['yhat'].iloc[-len(lgbm_predictions):].values +
                     (1 - prophet_weight) * lgbm_predictions)

    return ensemble_pred