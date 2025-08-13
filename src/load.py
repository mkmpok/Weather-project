import pandas as pd

def load_weather_data(filepath):
    """
    Loads the global weather repository dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
