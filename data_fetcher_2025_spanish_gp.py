import fastf1
import pandas as pd

# Enable FastF1 cache
fastf1.Cache.enable_cache("f1_cache")  # Ensure this directory exists

# Define sessions
sessions = ["FP1", "FP2", "FP3"]
year = 2025
gp_name = "Spanish Grand Prix"

# Create a list to hold the data
all_fp_data = []

for session_type in sessions:
    session = fastf1.get_session(year, gp_name, session_type)
    session.load()
    laps = session.laps

    # Filter and extract relevant columns
    lap_data = laps[[
        "Driver", "Team", "Compound", "LapTime", 
        "Sector1Time", "Sector2Time", "Sector3Time", "TrackStatus"
    ]].copy()
    lap_data["Session"] = session_type
    all_fp_data.append(lap_data)

# Combine all practice sessions
combined_fp = pd.concat(all_fp_data, ignore_index=True)

# Save to CSV
combined_fp.to_csv("spanish_gp_2025_fp1_fp2_fp3.csv", index=False)
print("✅ Spanish GP 2025 FP1–FP3 data saved to 'spanish_gp_2025_fp1_fp2_fp3.csv'")
