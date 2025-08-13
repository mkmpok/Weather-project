import pandas as pd
from src.config import DATE_COL
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

def add_rolling_lags(ts: pd.DataFrame, target: str, windows=(7,14,30)):
    ts = ts.sort_values(DATE_COL).copy()
    for w in windows:
        ts[f"{target}ma{w}"] = ts[target].rolling(w, min_periods=1).mean()
        ts[f"{target}lag{w}"] = ts[target].shift(w)
    ts = ts.dropna()
    return ts


# from sklearn.ensemble import RandomForestRegressor
# import shap
# import matplotlib.pyplot as plt


def feature_importance_shap(df, features, target='temperature_celsius'):
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.show()

# Example:
# df = pd.read_csv('data/weather_dataset/GlobalWeatherRepository.csv')
# features = ['humidity', 'pressure', 'wind_speed', 'precipitation_mm']
# feature_importance_shap(df, features)
