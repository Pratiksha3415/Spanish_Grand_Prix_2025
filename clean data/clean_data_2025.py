import os
import pandas as pd
import numpy as np

# === Setup ===
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "f1_cache", "2025")"
OUTPUT_FILE = "/clean_data/cleaned_data_2025_with_positions.csv"
YEAR_FILTER = "2025"

# === Initialize collections ===
all_sessions = []
race_positions = []
quali_positions = []

# === Loop through all 2025 session folders ===
for folder in sorted(os.listdir(RAW_DATA_DIR)):
    if not folder.startswith(YEAR_FILTER):
        continue

    session_path = os.path.join(RAW_DATA_DIR, folder)
    if not os.path.isdir(session_path):
        continue

    laps_path = os.path.join(session_path, "laps.csv")
    results_path = os.path.join(session_path, "results.csv")
    weather_path = os.path.join(session_path, "weather.csv")

    if not os.path.exists(laps_path):
        continue

    try:
        laps_df = pd.read_csv(laps_path)
        if laps_df.empty or "LapTime" not in laps_df.columns:
            continue

        # Convert LapTime to seconds and clean invalid laps
        laps_df["LapTimeSeconds"] = pd.to_timedelta(laps_df["LapTime"], errors="coerce").dt.total_seconds()
        laps_df = laps_df.dropna(subset=["LapTimeSeconds"])
        laps_df = laps_df[(laps_df["LapTimeSeconds"] > 40) & (laps_df["LapTimeSeconds"] < 200)]

        # Add session metadata
        laps_df["SessionFolder"] = folder
        laps_df["Year"] = YEAR_FILTER

        # Determine session type
        if "_R" in folder:
            laps_df["SessionType"] = "Race"
        elif "_Q" in folder:
            laps_df["SessionType"] = "Qualifying"
        else:
            laps_df["SessionType"] = "Other"

        # Merge driver results
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            if "Abbreviation" in results_df.columns and "Position" in results_df.columns:
                results_df = results_df.rename(columns={"Abbreviation": "Driver"})
                if "_R" in folder:
                    results_df = results_df.rename(columns={"Position": "FinalRacePosition"})
                    race_positions.append(results_df[["Driver", "FinalRacePosition"]])
                elif "_Q" in folder:
                    results_df = results_df.rename(columns={"Position": "FinalQualiPosition"})
                    quali_positions.append(results_df[["Driver", "FinalQualiPosition"]])

                laps_df = laps_df.merge(results_df, on="Driver", how="left")

        # Merge weather if available
        if os.path.exists(weather_path):
            weather_df = pd.read_csv(weather_path)
            if "AirTemp" in weather_df.columns:
                laps_df["AvgAirTemp"] = weather_df["AirTemp"].mean()

        all_sessions.append(laps_df)

    except Exception as e:
        print(f"⚠️ Error in {folder}: {e}")

# === Final merge ===
if all_sessions:
    final_df = pd.concat(all_sessions, ignore_index=True)

    # Compute and add average race finish
    if race_positions:
        race_all = pd.concat(race_positions)
        race_all["FinalRacePosition"] = pd.to_numeric(race_all["FinalRacePosition"], errors="coerce")
        avg_race = race_all.groupby("Driver")["FinalRacePosition"].mean().rename("AvgRaceFinish")
        final_df = final_df.merge(avg_race, on="Driver", how="left")

    # Compute and add average quali position
    if quali_positions:
        quali_all = pd.concat(quali_positions)
        quali_all["FinalQualiPosition"] = pd.to_numeric(quali_all["FinalQualiPosition"], errors="coerce")
        avg_quali = quali_all.groupby("Driver")["FinalQualiPosition"].mean().rename("AvgQualiPosition")
        final_df = final_df.merge(avg_quali, on="Driver", how="left")

    # Save final output
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Final 2025 cleaned dataset saved to: {OUTPUT_FILE}")
else:
    print(" No valid sessions processed for 2025.")
