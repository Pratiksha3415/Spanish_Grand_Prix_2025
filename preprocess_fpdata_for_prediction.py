import pandas as pd
import numpy as np

# === File paths ===
PRACTICE_DATA_PATH = "spanish_gp_2025_fp1_fp2_fp3.csv"  # Updated to include FP3
REFERENCE_DATA_PATH = "clean_data/final_features_cleaned.csv"
OUTPUT_FILE = "spanish_gp_2025_fp1_fp2_fp3_preprocessed.csv"  # Updated output filename

# === Load datasets ===
print("Loading datasets...")
df = pd.read_csv(PRACTICE_DATA_PATH)
ref_df = pd.read_csv(REFERENCE_DATA_PATH, low_memory=False)  # Fix the DtypeWarning

# === DEBUG: Check column names ===
print("\n=== DEBUGGING INFO ===")
print(f"Practice data shape: {df.shape}")
print(f"Reference data shape: {ref_df.shape}")
print("\nPractice data columns:")
print(df.columns.tolist())
print("\nLooking for lap time related columns...")
lap_time_cols = [col for col in df.columns if 'time' in col.lower() or 'lap' in col.lower()]
print(f"Potential lap time columns: {lap_time_cols}")

# Check session distribution
if "SessionFolder" in df.columns:
    print(f"\nSession distribution:")
    session_counts = df["SessionFolder"].value_counts()
    print(session_counts)
elif "Session" in df.columns:
    print(f"\nSession distribution:")
    session_counts = df["Session"].value_counts()
    print(session_counts)
else:
    print("Warning: No session column found to analyze session distribution")

# === Find the correct lap time column ===
# Common variations of lap time column names in F1 data
possible_lap_time_cols = [
    'LapTimeSeconds', 'LapTime', 'Time', 'LapTimeInSeconds', 
    'lap_time', 'lap_time_seconds', 'LapDuration', 'Duration'
]

lap_time_col = None
for col in possible_lap_time_cols:
    if col in df.columns:
        lap_time_col = col
        print(f"✅ Found lap time column: {col}")
        break

if lap_time_col is None:
    print("❌ No lap time column found! Available columns with 'time' or 'lap':")
    for col in df.columns:
        if 'time' in col.lower() or 'lap' in col.lower():
            print(f"  - {col}")
    print("\nPlease check your data and update the script with the correct column name.")
    exit()

# === Drop irrelevant columns if they exist ===
drop_cols = [
    "HeadshotUrl", "BroadcastName", "LapStartTime", "LapStartDate", 
    "Deleted", "FastF1Generated", "IsAccurate", "Unnamed: 0"
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors="ignore")

# === Barcelona Weather — manually set (different values for different sessions) ===
# You can adjust these based on actual weather conditions during each session
session_col = "SessionFolder" if "SessionFolder" in df.columns else ("Session" if "Session" in df.columns else None)

# Default weather features (can be customized per session)
default_weather = {
    "AvgTemp": 23.5,  # Average temperature across all sessions
    "RainAmount_mm": 0.24,
    "RainProbability": 1.0,
    "WindSpeed_mps": 3.9,
    "Pressure_hPa": 1019,
    "Humidity": 52,
    "UVIndex": 9,
    "DewPoint_C": 15,
}

# Session-specific weather adjustments (if needed)
session_weather_adjustments = {
    "FP1": {"AvgTemp": 22, "Humidity": 55},  # Typically cooler in morning
    "FP2": {"AvgTemp": 25, "Humidity": 50},  # Warmer in afternoon
    "FP3": {"AvgTemp": 24, "Humidity": 52},  # Morning session, moderate conditions
}

# Apply weather features
for feature, default_value in default_weather.items():
    df[feature] = default_value

# Apply session-specific adjustments if session column exists
if session_col is not None:
    for session, adjustments in session_weather_adjustments.items():
        session_mask = df[session_col].str.contains(session, case=False, na=False)
        for feature, value in adjustments.items():
            df.loc[session_mask, feature] = value
    print(f"Applied session-specific weather adjustments based on {session_col}")
else:
    print("No session column found, using default weather values for all data")

# === Lap time cleaning ===
print(f"\nSample lap time values:")
print(df[lap_time_col].head(10).tolist())

def parse_lap_time(time_value):
    """Convert lap time from various formats to seconds"""
    if pd.isna(time_value):
        return np.nan
    
    # Handle pandas Timedelta objects (like '0 days 00:01:16.802000')
    if isinstance(time_value, pd.Timedelta):
        return time_value.total_seconds()
    
    # Convert to string and handle string formats
    time_str = str(time_value).strip()
    
    # Handle pandas Timedelta string representation
    if 'days' in time_str and ':' in time_str:
        try:
            # Parse format like '0 days 00:01:16.802000'
            td = pd.to_timedelta(time_str)
            return td.total_seconds()
        except:
            pass
    
    # Handle already numeric values
    try:
        return float(time_str)
    except ValueError:
        pass
    
    # Handle MM:SS.SSS format
    if ':' in time_str and 'days' not in time_str:
        try:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        except ValueError:
            pass
    
    # Handle SS.SSS format (no minutes)
    try:
        return float(time_str)
    except ValueError:
        return np.nan

# Convert lap times to seconds
print("Converting lap times to seconds...")
df["LapTimeSeconds"] = df[lap_time_col].apply(parse_lap_time)

if lap_time_col != "LapTimeSeconds":
    df.drop(columns=[lap_time_col], inplace=True)  # Remove original column

print(f"\nLap time statistics after conversion:")
print(f"Valid lap times: {df['LapTimeSeconds'].notna().sum()}")
print(f"Invalid/missing lap times: {df['LapTimeSeconds'].isna().sum()}")

if df['LapTimeSeconds'].notna().sum() > 0:
    print(f"Lap time range: {df['LapTimeSeconds'].min():.3f}s - {df['LapTimeSeconds'].max():.3f}s")
    print(f"Mean lap time: {df['LapTimeSeconds'].mean():.3f}s")
    
    # Show lap time statistics by session if session column exists
    if session_col is not None:
        print(f"\nLap time statistics by session:")
        session_stats = df.groupby(session_col)['LapTimeSeconds'].agg(['count', 'mean', 'min', 'max'])
        print(session_stats)

# Drop rows with invalid lap times
initial_rows = len(df)
df.dropna(subset=["LapTimeSeconds"], inplace=True)
print(f"Dropped {initial_rows - len(df)} rows with invalid lap times")

if len(df) == 0:
    print("❌ ERROR: No valid lap times found after conversion!")
    print("Please check the lap time format in your data.")
    import sys
    sys.exit()

# === Feature Engineering ===
print("\nEngineering features...")

# Check if required columns exist before using them
if "Stint" in df.columns:
    df["StintLength"] = df["Stint"].fillna(0)
else:
    print("Warning: 'Stint' column not found, setting StintLength to 0")
    df["StintLength"] = 0

if "IsPersonalBest" in df.columns:
    df["IsFastLap"] = df["IsPersonalBest"].astype(int)
else:
    print("Warning: 'IsPersonalBest' column not found, setting IsFastLap to 0")
    df["IsFastLap"] = 0

# Check if Driver column exists
if "Driver" not in df.columns:
    print("❌ ERROR: 'Driver' column not found!")
    driver_cols = [col for col in df.columns if 'driver' in col.lower()]
    print(f"Possible driver columns: {driver_cols}")
    exit()

# Check if Team column exists
team_col = None
possible_team_cols = ["Team", "TeamName", "Constructor", "team"]
for col in possible_team_cols:
    if col in df.columns:
        team_col = col
        break

if team_col is None:
    print("Warning: No team column found, skipping team-based features")
    df["TeamMedianPace"] = df["LapTimeSeconds"].median()  # Use overall median as fallback
else:
    if team_col != "Team":
        df["Team"] = df[team_col]  # Standardize column name

# Driver Avg Pace (across all sessions)
driver_avg = df.groupby("Driver")["LapTimeSeconds"].mean().rename("DriverAvgPace")
df = df.merge(driver_avg, on="Driver", how="left")

# Driver Avg Pace per session
if session_col is not None:
    driver_session_avg = df.groupby(["Driver", session_col])["LapTimeSeconds"].mean().rename("DriverSessionAvgPace")
    df = df.merge(driver_session_avg, on=["Driver", session_col], how="left")

# Team Median Pace (if team column exists)
if team_col is not None:
    team_median = df.groupby("Team")["LapTimeSeconds"].median().rename("TeamMedianPace")
    df = df.merge(team_median, on="Team", how="left")
    
    # Team Median Pace per session
    if session_col is not None:
        team_session_median = df.groupby(["Team", session_col])["LapTimeSeconds"].median().rename("TeamSessionMedianPace")
        df = df.merge(team_session_median, on=["Team", session_col], how="left")

# Spanish GP Indicator
df["IsSpanishGP"] = 1

# Session Type encoding (if session column exists)
if session_col is not None:
    df["IsFP1"] = df[session_col].str.contains("FP1", case=False, na=False).astype(int)
    df["IsFP2"] = df[session_col].str.contains("FP2", case=False, na=False).astype(int)
    df["IsFP3"] = df[session_col].str.contains("FP3", case=False, na=False).astype(int)
    
    # Session progression (FP1=1, FP2=2, FP3=3)
    df["SessionProgression"] = (
        df["IsFP1"] * 1 + 
        df["IsFP2"] * 2 + 
        df["IsFP3"] * 3
    )
else:
    print("Warning: No session column found, cannot create session-specific features")
    df["IsFP1"] = 0
    df["IsFP2"] = 0
    df["IsFP3"] = 0
    df["SessionProgression"] = 1

# Clean Air flag
if "TrackStatus" in df.columns:
    df["IsCleanAir"] = (df["TrackStatus"] == 1).astype(int)
else:
    print("Warning: 'TrackStatus' column not found, setting IsCleanAir to 0")
    df["IsCleanAir"] = 0

# Adjusted Lap Time based on weather
df["AdjustedLapTime"] = df["LapTimeSeconds"] * (
    1 + df["RainProbability"] * 0.05 + df["Humidity"] / 1000
)

# Lap count per driver per session
if session_col is not None:
    lap_counts = df.groupby(["Driver", session_col]).size().rename("DriverSessionLapCount")
    df = df.merge(lap_counts, on=["Driver", session_col], how="left")
else:
    print("Warning: No session column found, setting DriverSessionLapCount to lap count per driver")
    lap_counts = df.groupby("Driver").size().rename("DriverSessionLapCount")
    df = df.merge(lap_counts, on="Driver", how="left")

# Track evolution (assume track gets faster over sessions)
if session_col is not None:
    # Calculate track evolution factor based on session
    session_factors = {"FP1": 1.0, "FP2": 0.98, "FP3": 0.96}  # Track gets ~2% faster each session
    df["TrackEvolutionFactor"] = 1.0
    for session, factor in session_factors.items():
        session_mask = df[session_col].str.contains(session, case=False, na=False)
        df.loc[session_mask, "TrackEvolutionFactor"] = factor
else:
    df["TrackEvolutionFactor"] = 1.0

# === Drop rows missing critical engineered features ===
critical_features = ["DriverAvgPace"]
if team_col is not None:
    critical_features.append("TeamMedianPace")

initial_rows = len(df)
df.dropna(subset=critical_features, inplace=True)
print(f"Dropped {initial_rows - len(df)} rows missing critical features")

# === Final data info ===
print(f"\n=== FINAL PROCESSED DATA ===")
print(f"Final shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Unique drivers: {df['Driver'].nunique()}")
if team_col is not None:
    print(f"Unique teams: {df['Team'].nunique()}")

if session_col is not None:
    print(f"Session distribution after processing:")
    print(df[session_col].value_counts())
    print(f"FP1 laps: {df['IsFP1'].sum()}")
    print(f"FP2 laps: {df['IsFP2'].sum()}")
    print(f"FP3 laps: {df['IsFP3'].sum()}")

# Show sample of key features by session
if session_col is not None and len(df) > 0:
    print(f"\nSample statistics by session:")
    session_summary = df.groupby(session_col).agg({
        'LapTimeSeconds': ['count', 'mean', 'min'],
        'DriverAvgPace': 'mean',
        'AdjustedLapTime': 'mean'
    }).round(3)
    print(session_summary)

# === Save preprocessed data ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Preprocessed FP1/FP2/FP3 data saved to: {OUTPUT_FILE}")
print(f"Saved {len(df)} rows with {len(df.columns)} columns")

# Show final column list
print(f"\nFinal columns in processed data:")
print(df.columns.tolist())