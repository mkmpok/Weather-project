import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df, filepath='reports/correlation_heatmap.png'):
    """
    Generates a heatmap to visualize feature correlations.
    """
    plt.figure(figsize=(20, 15))
    corr_matrix = df.select_dtypes(include='number').corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Weather Features')
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Correlation heatmap saved to {filepath}")
    plt.show()


def plot_time_series_trends(df, city, column, filepath='reports/time_series_trend.png'):
    """
    Plots the time series trend for a specific city and column.
    """
    # Added a check to make sure the city exists
    if city not in df['name'].unique():
        print(f"Warning: City '{city}' not found in the dataset. Skipping plot.")
        return

    city_data = df[df['name'] == city].copy()
    plt.figure(figsize=(15, 7))
    city_data[column].plot(title=f'{column.replace("_", " ").title()} Trend for {city}', fontsize=12)
    plt.xlabel('Date')
    plt.ylabel(column.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Time series trend plot saved to {filepath}")
    plt.show()


def plot_seasonal_decomposition(df, city, column, filepath='reports/seasonal_decomposition.png'):
    """
    Performs and plots seasonal decomposition to identify trend, seasonality, and residuals.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if city not in df['name'].unique():
        print(f"Warning: City '{city}' not found in the dataset. Skipping seasonal decomposition.")
        return

    city_data = df[df['name'] == city].copy()
    city_data = city_data.resample('D').mean(numeric_only=True).dropna()

    if len(city_data) < 2 * 365:
        print(
            f"Warning: Not enough data for seasonal decomposition for {city}. At least two years are recommended. Skipping.")
        return

    decomposition = seasonal_decompose(city_data[column], model='additive', period=365)

    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    fig.suptitle(f'Seasonal Decomposition for {city}', fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Seasonal decomposition plot saved to {filepath}")
    plt.show()
