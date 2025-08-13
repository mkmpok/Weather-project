from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "weather_dataset" / "GlobalWeatherRepository.csv"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"
REPORT_MD = REPORTS_DIR / "report.md"

# Column names (matching your dataset exactly)
DATE_COL = "last_updated_epoch"
CITY_COL = "location_name"

# Targets
TEMP_COL = "temperature_celsius"   # from your dataset
PRECIP_COL = "precip_mm"           # already matches your dataset

# Modeling
TEST_SIZE_FRACTION = 0.2
RANDOM_STATE = 42

# For rolling features
ROLL_WINDOWS = (7, 14, 30)

# PM Accelerator mission
PM_MISSION = (
    "I’m on a mission to help launch *1,000+ AI products* and empower professionals like you "
    "to become the next generation of AI product leaders — impacting *millions of lives* "
    "through real-world innovation. — Dr. Nancy Li"
)
