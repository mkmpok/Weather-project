import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    datetime_cols = ['last_updated', 'sunrise', 'sunset', 'moonrise', 'moonset']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_hour'] = df[col].dt.hour
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)
    return df

@st.cache_data
def preprocess_data(df):
    target_vars = ['temperature_celsius']
    df = df.dropna(subset=target_vars)

    exclude_cols = ['last_updated_epoch', 'location_name', 'timezone']
    X = df.drop(columns=target_vars + [col for col in exclude_cols if col in df.columns])
    y = df[target_vars[0]]

    high_card_cols = ['country']
    for col in high_card_cols:
        if col in X.columns and X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    X = pd.get_dummies(X, drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    return X, y

@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, X_train, X_test, y_train, y_test, y_pred, mse, r2

def main():
    st.title("Weather Temperature Forecasting")

    file_path = r"C:\Users\Shrad\PycharmProjects\pythonWeather\data\weather_dataset\GlobalWeatherRepository.csv"
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at: {file_path}")
        return

    st.info("Loading data...")
    df = load_data(file_path)
    st.success("Data loaded!")

    st.write("### Sample data")
    st.dataframe(df.head())

    X, y = preprocess_data(df)

    st.write(f"### Features shape: {X.shape}")
    st.write(f"### Target shape: {y.shape}")

    with st.spinner("Training Linear Regression model..."):
        model, X_train, X_test, y_train, y_test, y_pred, mse, r2 = train_model(X, y)

    st.success("Model trained!")

    st.write("### Model Evaluation Metrics")
    st.write(f"- Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"- R² Score: {r2:.2f}")

    # Plot Actual vs Predicted
    st.write("### Actual vs Predicted Temperature")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.set_title("Actual vs Predicted Temperature Scatter Plot")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
