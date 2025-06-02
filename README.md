Spanish_Grand_Prix_2025🏎️

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
