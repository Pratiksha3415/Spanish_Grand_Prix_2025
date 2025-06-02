Spanish_Grand_Prix_2025🏎️

Welcome to the Spanish Grand Prix 2025 Prediction Project, where we build two different models to predict race outcomes using only practice session data (FP1–FP3). This repo showcases the power of combining machine learning with domain logic in Formula 1 sports analytics.

## 📊 Data Sources  
- **FastF1 API**: Live telemetry, session info, and lap data  
- **2025 Practice Sessions**: FP1, FP2, FP3 data fetched live  
- **Driver/Team Historical Trends**: Built into feature engineering logic  
- **Weather Data** *(Optional)*

  
📁 File Structure
├── f1_cache/                              # Practice session cache

├── Spanish_GP_2025/                       # Utility folder
├── data_fetcher_2025_spanish_gp.py       # Download FP1-FP3 data
├── preprocess_fpdata_for_prediction.py   # Clean and preprocess data
├── train_model.py                         # Train XGBoost model
├── spanish_gp_2025_predictor.py           # Final prediction script
├── spanish_gp_2025_fp1_fp2_fp3.csv        # Raw data
├── spanish_gp_2025_fp1_fp2_fp3_preprocessed.csv  # Preprocessed data
