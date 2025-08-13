import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_anomalies(df, feature='temperature_celsius', contamination=0.01):
    data = df[[feature]].dropna()
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(data)
    df['anomaly_flag'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return df

def plot_anomalies(df, city, feature='temperature_celsius'):
    city_data = df[df['location_name'] == city]
    plt.figure(figsize=(12,6))
    plt.plot(city_data['last_updated'], city_data[feature], label=feature)
    plt.scatter(city_data[city_data['anomaly_flag']=='Anomaly']['last_updated'],
                city_data[city_data['anomaly_flag']=='Anomaly'][feature], color='red', label='Anomaly')
    plt.title(f'{feature} Anomalies in {city}')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.legend()
    plt.show()

# Example usage:
# df = pd.read_csv('data/weather_dataset/GlobalWeatherRepository.csv', parse_dates=['last_updated'])
# df = detect_anomalies(df)
# plot_anomalies(df, 'New York')
