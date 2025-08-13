import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import shap
import lightgbm as lgb
import numpy as np


def analyze_climate_trends(df, filepath='reports/climate_trends.png'):
    """
    Analyzes and visualizes long-term temperature trends by continent.
    """
    df['year'] = df.index.year
    df_yearly_continent = df.groupby(['continent', 'year'])['temp_c'].mean().reset_index()

    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df_yearly_continent, x='year', y='temp_c', hue='continent')
    plt.title('Long-term Temperature Trends by Continent', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel('Average Temperature (°C)')
    plt.legend(title='Continent')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Climate trends plot saved to {filepath}")
    plt.show()


def analyze_environmental_impact(df, filepath='reports/environmental_impact.png'):
    """
    Analyzes air quality and its correlation with various weather parameters.
    """
    if 'co' in df.columns and 'temp_c' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='temp_c', y='co', alpha=0.5)
        plt.title('Correlation between Temperature and Carbon Monoxide (CO)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Carbon Monoxide (µg/m³)')
        plt.tight_layout()
        plt.savefig(filepath)
        print(f"Environmental impact plot saved to {filepath}")
        plt.show()
    else:
        print("Warning: 'co' or 'temp_c' column not found for environmental impact analysis.")


def perform_spatial_analysis(df, filepath='reports/weather_map.html'):
    """
    Creates an interactive map to visualize weather conditions.
    """
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("Warning: Latitude and longitude columns are required for spatial analysis.")
        return

    latest_data = df.sort_index().drop_duplicates(subset=['name'], keep='last')
    m = folium.Map(location=[latest_data['latitude'].mean(), latest_data['longitude'].mean()], zoom_start=2)

    for _, row in latest_data.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color='red' if row['temp_c'] > 25 else 'blue',
            fill=True,
            fill_color='red' if row['temp_c'] > 25 else 'blue',
            fill_opacity=0.6,
            popup=f"City: {row['name']}<br>Temp: {row['temp_c']}°C<br>Condition: {row['condition']}"
        ).add_to(m)

    m.save(filepath)
    print(f"Interactive weather map saved to {filepath}")


def analyze_feature_importance(model, X, filepath='reports/feature_importance.png'):
    """
    Analyzes and visualizes feature importance using SHAP.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"SHAP feature importance plot saved to {filepath}")
    plt.show()