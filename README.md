Spanish_Grand_Prix_2025🏎️

Welcome to the Spanish Grand Prix 2025 Prediction Project, where we build two different models to predict race outcomes using only practice session data (FP1–FP3). This repo showcases the power of combining machine learning with domain logic in Formula 1 sports analytics.

The model leverages:  
- FastF1 API for 2025 practice session data  
- Feature engineering on driver/team performance  
- Grid position, practice pace, and reliability  
- Realistic ranking logic for podium predictions

## 📊 Data Sources  
- **FastF1 API**: Live telemetry, session info, and lap data  
- **2025 Practice Sessions**: FP1, FP2, FP3 data fetched live  
- **Driver/Team Historical Trends**: Built into feature engineering logic  
- **Weather Data** *(Optional)*

## 🔧 Dependencies  
Install required libraries using:  
```bash
pip install fastf1 pandas numpy scikit-learn xgboost
