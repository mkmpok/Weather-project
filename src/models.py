import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

# ✅ File path (change if needed)
file_path = r"C:\Users\Shrad\PycharmProjects\pythonWeather\data\weather_dataset\GlobalWeatherRepository.csv"

# ✅ Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {file_path}")

print("📂 Loading dataset...")
df = pd.read_csv(file_path)

# ✅ Show available columns
print("\n📌 Available columns in dataset:")
print(df.columns.tolist())

# Convert datetime columns to numeric features (if present)
datetime_cols = ['last_updated', 'sunrise', 'sunset', 'moonrise', 'moonset']
for col in datetime_cols:
    if col in df.columns:
        # Try parsing datetime with fallback to coerce errors
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        df.drop(columns=[col], inplace=True)

# ✅ Target variable (using correct column name)
target_vars = ['temperature_celsius']

# Check for missing target columns
missing_targets = [t for t in target_vars if t not in df.columns]
if missing_targets:
    raise KeyError(f"❌ Target column(s) {missing_targets} not found in dataset. "
                   f"Available columns are: {df.columns.tolist()}")

# Drop rows where target is missing
df = df.dropna(subset=target_vars)

# Columns to exclude from features to avoid high-cardinality or irrelevant columns
exclude_cols = ['last_updated_epoch', 'location_name', 'timezone']

# Prepare features by dropping target and excluded columns
X = df.drop(columns=target_vars + [col for col in exclude_cols if col in df.columns])

# Target variable
y = df[target_vars[0]]

# Label encode high-cardinality categorical columns instead of one-hot encoding
high_card_cols = ['country']
for col in high_card_cols:
    if col in X.columns and X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# One-hot encode remaining categorical columns
X = pd.get_dummies(X, drop_first=True)

# Impute missing values in features with mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Convert imputed numpy array back to DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🚀 Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\n✅ Model training complete and metrics calculated successfully.")
