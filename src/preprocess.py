import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_column_names(df):
    """
    Cleans column names by converting them to lowercase and replacing spaces with underscores.
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def preprocess_data(df):
    """
    Cleans and preprocesses the raw weather data.
    """
    df = clean_column_names(df)

    # Check for the timestamp column name and handle variations
    if 'last_updated' in df.columns:
        timestamp_col = 'last_updated'
    elif 'lastupdated' in df.columns:
        timestamp_col = 'lastupdated'
    else:
        raise KeyError("Timestamp column 'last_updated' or 'lastupdated' not found in DataFrame.")

    # Check for the city name column and handle variations
    if 'city_name' in df.columns:
        df = df.rename(columns={'city_name': 'name'})
    elif 'city' in df.columns:
        df = df.rename(columns={'city': 'name'})

    # Convert timestamp column to datetime and set as index
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df.set_index(timestamp_col, inplace=True)

    # Drop columns that are mostly empty or not useful for this analysis
    df.drop(columns=['time', 'is_day', 'icon'], inplace=True, errors='ignore')

    # Forward-fill missing values for most columns
    df.fillna(method='ffill', inplace=True)

    # Drop any remaining rows with NaN values
    df.dropna(inplace=True)

    print("Data preprocessed successfully.")

    return df


def feature_engineering(df):
    """
    Creates new features from existing data.
    """
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour

    return df