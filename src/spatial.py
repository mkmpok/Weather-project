import pandas as pd
import matplotlib.pyplot as plt
from src.utils import safe_savefig
import folium
def average_by_country(df: pd.DataFrame, country_col: str, value_col: str) -> pd.DataFrame:
    g = df.dropna(subset=[country_col, value_col]).groupby(country_col)[value_col].mean().reset_index()
    g.columns = [country_col, f"mean_{value_col}"]
    return g

def choropleth_if_available(country_df: pd.DataFrame, country_col: str, value_col: str, out_path: str):
    """
    Draws a crude choropleth using GeoPandas if installed; otherwise writes a CSV summary.
    """
    try:
        import geopandas as gpd
    except Exception:
        # fallback: save table for the report
        country_df.to_csv(out_path.replace(".png", ".csv"), index=False)
        return False

    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # try to merge on name (user may need to align names if dataset differs)
    merged = world.merge(country_df, left_on="name", right_on=country_col, how="left")

    ax = merged.plot(column=f"mean_{value_col}", legend=True, figsize=(10,6))
    ax.set_title(f"Choropleth — mean {value_col} by country")
    safe_savefig(out_path)
    return True



def create_temperature_map(df, output_file='temperature_map.html'):
    latest = df.sort_values('last_updated').groupby('location_name').last().reset_index()
    map_center = [latest['latitude'].mean(), latest['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=2)

    for idx, row in latest.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            popup=f"{row['location_name']}: {row['temperature_celsius']} °C",
            color='blue',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    m.save(output_file)
    print(f"Map saved as {output_file}")

# Example:
# df = pd.read_csv('data/weather_dataset/GlobalWeatherRepository.csv', parse_dates=['last_updated'])
# create_temperature_map(df)
