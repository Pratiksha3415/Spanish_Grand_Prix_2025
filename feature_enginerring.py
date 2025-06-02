import pandas as pd
import numpy as np
import os

# === Paths ===
INPUT_FILE = "/clean_data/cleaned_data_2025_with_positions.csv"
OUTPUT_FILE = "/clean_data/final_features_cleaned.csv"

# === Load data ===
df = pd.read_csv(INPUT_FILE)

# === Drop unnecessary columns ===
drop_cols = [
    "HeadshotUrl", "BroadcastName", "Time", "LapStartTime", "LapStartDate", 
    "Deleted", "FastF1Generated", "IsAccurate", "Unnamed: 0"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# === Barcelona GP Weather (Manual) ===
weather_features = {
    "AvgTemp": np.mean([22, 25, 24, 23]),
    "RainAmount_mm": 0.24,
    "RainProbability": 1.0,
    "WindSpeed_mps": 3.9,
    "Pressure_hPa": 1019,
    "Humidity": 52,
    "UVIndex": 9,
    "DewPoint_C": 15,
}
for key, value in weather_features.items():
    df[key] = value

# === Lap time cleanup ===
df["LapTimeSeconds"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce")
df = df.dropna(subset=["LapTimeSeconds"])

# === Feature Engineering ===
df["StintLength"] = df["Stint"].fillna(0)
df["IsFastLap"] = df.get("IsPersonalBest", 0).astype(int)

# 🏎️ Driver Average Pace
driver_avg = df.groupby("Driver")["LapTimeSeconds"].mean().rename("DriverAvgPace")
df = df.merge(driver_avg, on="Driver", how="left")

# 🔧 Team Median Pace
team_median = df.groupby("Team")["LapTimeSeconds"].median().rename("TeamMedianPace")
df = df.merge(team_median, on="Team", how="left")

# 🇪🇸 Spanish GP indicator
df["IsSpanishGP"] = df["SessionFolder"].str.contains("Spanish", case=False).astype(int)

# 💨 Clean Air Pace
df["IsCleanAir"] = (df["TrackStatus"] == 1).astype(int) if "TrackStatus" in df.columns else np.nan

# 🌧️ Adjusted Lap Time (weather-weighted)
df["AdjustedLapTime"] = df["LapTimeSeconds"] * (
    1 + df["RainProbability"] * 0.05 + df["Humidity"] / 1000
)

# 🧮 Lap count per driver per session
lap_counts = df.groupby(["Driver", "SessionFolder"]).size().rename("DriverSessionLapCount")
df = df.merge(lap_counts, on=["Driver", "SessionFolder"], how="left")

# === Drop rows with missing engineered values ===
essential_cols = ["DriverAvgPace", "TeamMedianPace"]
df.dropna(subset=essential_cols, inplace=True)

# === Save final dataset ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Final cleaned and engineered dataset saved to: {OUTPUT_FILE}")
