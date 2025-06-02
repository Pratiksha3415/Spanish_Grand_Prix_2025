import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import os

# === Paths ===
DATA_PATH = "clean_data/final_features_cleaned.csv"
MODEL_OUTPUT_DIR = "Spanish_GP_2025"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(DATA_PATH)

# === Filter only Race Sessions ===
df = df[df["SessionType"].str.lower() == "race"]
df = df.dropna(subset=["FinalRacePosition"])
df["FinalRacePosition"] = pd.to_numeric(df["FinalRacePosition"], errors="coerce")
y = df["FinalRacePosition"]

# === Drop unnecessary columns ===
drop_cols = [
    "FinalRacePosition", "FinalQualiPosition", "Position", "DriverNumber",
    "LapTime", "Date", "Time", "SessionFolder", "DriverHeadshotUrl", "TeamColor", 
    "Driver", "Team", "Compound", "TrackStatus", "Time", "Q1", "Q2", "Q3"
]
X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
X = X.select_dtypes(include=[np.number])

# === Save feature list for future predictions ===
feature_list = X.columns.tolist()
with open(os.path.join(MODEL_OUTPUT_DIR, "race_model_features.txt"), "w") as f:
    for col in feature_list:
        f.write(f"{col}\n")

# === Handle NaNs ===
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# === Models to compare ===
models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, objective='reg:squarederror', random_state=42)
}

# === Train and Evaluate ===
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    results[name] = mae
    print(f"📊 {name} MAE: {mae:.3f}")

# === Save Best Model ===
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
model_path = os.path.join(MODEL_OUTPUT_DIR, f"best_race_model_{best_model_name}.pkl")
joblib.dump(best_model, model_path)

print(f"\n✅ Best Model: {best_model_name} (MAE: {results[best_model_name]:.3f})")
print(f"📁 Model saved to: {model_path}")
print(f"📄 Feature list saved to: race_model_features.txt")