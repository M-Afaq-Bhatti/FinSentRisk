import os
from dotenv import load_dotenv
import pandas as pd
from fredapi import Fred

# Load API key
load_dotenv()
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    raise ValueError("FRED_API_KEY not found in .env file.")

# Init FRED client
fred = Fred(api_key=api_key)

# Directory for saving data
OUTPUT_DIR = "data/raw/macro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of macro indicators (keyword: series_id)
# These are handpicked for relevance
series_ids = {
    "GDP": "GDP",  # Quarterly
    "Consumer_Price_Index": "CPIAUCSL",  # Monthly
    "Unemployment_Rate": "UNRATE",  # Monthly
    "Interest_Rate_Fed_Funds": "FEDFUNDS",  # Monthly
    "Industrial_Production_Index": "INDPRO"  # Monthly
}

# Date range
start_date = "2000-01-01"
end_date = "2022-12-31"

def download_series(series_name, series_id):
    """Download series from FRED and save to CSV."""
    print(f"Downloading {series_name} ({series_id})...")
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        df = pd.DataFrame(data, columns=[series_name])
        df.index.name = "Date"
        output_path = os.path.join(OUTPUT_DIR, f"{series_name}.csv")
        df.to_csv(output_path)
        print(f"✅ Saved {series_name} to {output_path}")
    except Exception as e:
        print(f"❌ Failed to download {series_name}: {e}")

if __name__ == "__main__":
    for name, sid in series_ids.items():
        download_series(name, sid)

    print("\nAll downloads complete.")
