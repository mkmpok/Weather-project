import pandas as pd
import numpy as np
import os
from src.load import load_weather_data
from src.preprocess import preprocess_data, feature_engineering
from src.eda import plot_correlation_heatmap, plot_time_series_trends, plot_seasonal_decomposition
from src.anomalies import detect_anomalies
from src.forecasting_models import train_prophet_model, train_lightgbm_model
from src.unique_analyses import analyze_climate_trends, analyze_environmental_impact, perform_spatial_analysis, \
    analyze_feature_importance


def create_reports_directory():
    if not os.path.exists('reports'):
        os.makedirs('reports')
        print("Created 'reports' directory.")


def main():
    """
    Orchestrates the entire weather trend forecasting project workflow.
    """
    # PM Accelerator Mission statement
    print(
        "PM Accelerator Mission: This project masters data science skills to drive impact through advanced weather trend forecasting, demonstrating expertise in data manipulation, machine learning, and insightful analysis.")
    print("-" * 80)

    create_reports_directory()

    # 1. Data Loading and Preprocessing
    filepath = 'data/weather_dataset/GlobalWeatherRepository.csv'
    df = load_weather_data(filepath)
    if df is None:
        return

    df_preprocessed = preprocess_data(df.copy())
    df_features = feature_engineering(df_preprocessed.copy())

    city_of_interest = 'London'

    # 2. Advanced EDA and Anomaly Detection
    plot_correlation_heatmap(df_features)
    plot_time_series_trends(df_features, city=city_of_interest, column='temp_c')
    plot_seasonal_decomposition(df_features, city=city_of_interest, column='temp_c')
    anomalies = detect_anomalies(df_features, city=city_of_interest, column='temp_c')
    print(f"Anomalies detected in {city_of_interest}:\n", anomalies)

    # 3. Forecasting with Multiple Models
    prophet_model, prophet_forecast = train_prophet_model(df_features, city=city_of_interest)

    feature_cols = ['pressure_mb', 'humidity', 'wind_kph']
    lgbm_model, y_test_lgbm, y_pred_lgbm = train_lightgbm_model(
        df_features, city=city_of_interest, feature_cols=feature_cols, target_col='temp_c'
    )

    # 4. Unique Analyses
    analyze_climate_trends(df_features)
    analyze_environmental_impact(df_features)
    perform_spatial_analysis(df)

    # Note: Requires aligned feature set
    # analyze_feature_importance(lgbm_model, X_train_lgbm)

    print("-" * 80)
    print("All analyses completed. Check the 'reports' directory for visualizations and outputs.")


if __name__ == '__main__':
    main()
