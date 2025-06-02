import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class RealisticSpanishGPPredictor:
    def __init__(self):
        self.model = None
        
        # ACTUAL 2025 F1 driver ratings based on current grid and form
        self.driver_ratings = {
            # McLaren - Currently dominant team
            'NOR': 9.3,  # Norris - McLaren's lead driver, championship contender
            'PIA': 9.1,  # Piastri - McLaren pace, proven winner
            
            # Ferrari - Hamilton joins!
            'LEC': 8.9,  # Leclerc - Ferrari's established star
            'HAM': 8.7,  # Hamilton - GOAT but age factor, new team
            
            # Red Bull - Verstappen + new teammate Lawson
            'VER': 9.5,  # Verstappen - Still the benchmark, Spanish GP king
            'LAW': 7.8,  # Lawson - New to Red Bull, promising but unproven
            
            # Mercedes - Russell + Antonelli (rookie)
            'RUS': 8.5,  # Russell - Mercedes leader, consistent
            'ANT': 7.2,  # Antonelli - Rookie, huge potential but inexperienced
            
            # Aston Martin - Alonso + Stroll
            'ALO': 8.3,  # Alonso - Experience + home race advantage
            'STR': 6.9,  # Stroll - Improved but still pay driver level
            
            # Alpine - Gasly + Doohan/Colapinto rotation
            'GAS': 7.7,  # Gasly - Solid midfield performer
            'DOO': 7.0,  # Doohan - Rookie when racing
            'COL': 7.4,  # Colapinto - Impressive 2024 showings
            
            # Haas - Bearman + Ocon
            'BEA': 7.3,  # Bearman - Promising young talent
            'OCO': 7.5,  # Ocon - Experienced, race winner
            
            # Racing Bulls - Tsunoda + Hadjar
            'TSU': 7.4,  # Tsunoda - Fast but inconsistent
            'HAD': 6.8,  # Hadjar - Rookie from F2
            
            # Williams - Albon + Sainz Jr
            'ALB': 7.6,  # Albon - Williams leader, solid performer
            'SAI': 8.1,  # Sainz Jr - Ex-Ferrari, very experienced
            
            # Sauber - Hulkenberg + Bortoleto
            'HUL': 7.2,  # Hulkenberg - Veteran consistency
            'BOR': 6.6,  # Bortoleto - F2 champion rookie
        }
        
        # Team strength multipliers for 2025 (realistic current order)
        self.team_strength = {
            'McLaren': 1.12,        # Dominant in 2024/early 2025
            'Ferrari': 1.08,        # Strong with Hamilton addition
            'Red Bull Racing': 1.06, # Still strong but not peak dominance
            'Mercedes': 1.02,       # Improving but behind top 3
            'Aston Martin': 0.98,   # Midfield battler
            'Alpine': 0.96,         # Struggling recently
            'Williams': 0.94,       # Improving with Sainz
            'Haas F1 Team': 0.92,   # Solid midfield
            'Racing Bulls': 0.91,   # Sister team struggles
            'Kick Sauber': 0.88,    # Back of grid
        }
        
        # Spanish GP specific bonuses (track specialists)
        self.spanish_gp_bonus = {
            'VER': 0.5,   # 3 consecutive wins (2022, 2023, 2024) - HUGE bonus
            'ALO': 0.25,  # Home race + extensive Barcelona experience
            'HAM': 0.15,  # Won there 6 times (2014, 2017, 2018, 2019, 2020, 2021)
            'LEC': 0.1,   # Ferrari traditionally strong at Barcelona
            'RUS': 0.08,  # Good at technical tracks
            'SAI': 0.12,  # Spanish driver + track knowledge
        }
    
    def load_data(self, practice_file):
        """Load and process practice data with 2025 grid"""
        print("📊 Loading practice data for 2025 F1 grid...")
        
        try:
            self.practice_data = pd.read_csv(practice_file)
            print(f"✅ Practice data loaded: {len(self.practice_data)} rows")
        except FileNotFoundError:
            print("⚠️  Creating realistic 2025 grid sample data...")
            
            # ACTUAL 2025 F1 GRID
            drivers_teams = [
                ('NOR', 'McLaren'), ('PIA', 'McLaren'),
                ('LEC', 'Ferrari'), ('HAM', 'Ferrari'),  # Hamilton to Ferrari!
                ('VER', 'Red Bull Racing'), ('LAW', 'Red Bull Racing'),  # Lawson replaces Perez
                ('RUS', 'Mercedes'), ('ANT', 'Mercedes'),  # Antonelli rookie
                ('ALO', 'Aston Martin'), ('STR', 'Aston Martin'),
                ('GAS', 'Alpine'), ('COL', 'Alpine'),  # Colapinto currently racing
                ('BEA', 'Haas F1 Team'), ('OCO', 'Haas F1 Team'),
                ('TSU', 'Racing Bulls'), ('HAD', 'Racing Bulls'),  # Hadjar rookie
                ('ALB', 'Williams'), ('SAI', 'Williams'),  # Sainz to Williams
                ('HUL', 'Kick Sauber'), ('BOR', 'Kick Sauber'),  # Bortoleto rookie
            ]
            
            # Generate realistic lap times based on 2025 form
            sample_data = []
            base_time = 75.2  # Realistic Barcelona lap time for 2025 cars
            
            for session in ['FP1', 'FP2', 'FP3']:
                session_multiplier = {'FP1': 1.015, 'FP2': 1.0, 'FP3': 0.995}[session]
                
                for driver, team in drivers_teams:
                    driver_rating = self.driver_ratings.get(driver, 7.0)
                    team_mult = self.team_strength.get(team, 0.95)
                    spanish_bonus = self.spanish_gp_bonus.get(driver, 0)
                    
                    # Calculate realistic lap time
                    # Convert rating to time delta (10.0 - rating gives penalty)
                    time_delta = (9.7 - driver_rating) * 0.25
                    time_delta *= (2.1 - team_mult)  # Team factor
                    time_delta -= spanish_bonus * 0.6  # Track specialist bonus
                    time_delta *= session_multiplier
                    
                    # Add session-specific randomness
                    session_noise = np.random.normal(0, 0.08)
                    if session == 'FP1':
                        session_noise += np.random.uniform(0, 0.15)  # More variation in FP1
                    
                    lap_time = base_time + time_delta + session_noise
                    
                    sample_data.append({
                        'Driver': driver,
                        'Team': team,
                        'Session': session,
                        'Time': max(74.0, lap_time)  # Ensure reasonable minimum
                    })
            
            self.practice_data = pd.DataFrame(sample_data)
            print("✅ Realistic 2025 grid sample data created")
        
        return self
    
    def calculate_practice_performance(self):
        """Calculate comprehensive practice performance metrics"""
        print("🔧 Analyzing practice performance for 2025 grid...")
        
        driver_performance = {}
        
        for driver in self.practice_data['Driver'].unique():
            driver_data = self.practice_data[self.practice_data['Driver'] == driver]
            team = driver_data['Team'].iloc[0]
            
            # Best time across all sessions
            best_time = driver_data['Time'].min()
            
            # Consistency (lower std = more consistent)
            consistency = driver_data['Time'].std()
            
            # Long run pace (average excluding fastest 25%)
            sorted_times = driver_data['Time'].sort_values()
            long_run_times = sorted_times.iloc[int(len(sorted_times) * 0.25):]
            long_run_pace = long_run_times.mean()
            
            # Session progression (improvement from FP1 to FP3)
            fp1_times = driver_data[driver_data['Session'] == 'FP1']['Time']
            fp3_times = driver_data[driver_data['Session'] == 'FP3']['Time']
            
            if len(fp1_times) > 0 and len(fp3_times) > 0:
                progression = fp1_times.min() - fp3_times.min()
            else:
                progression = 0
            
            driver_performance[driver] = {
                'team': team,
                'best_time': best_time,
                'consistency': consistency,
                'long_run_pace': long_run_pace,
                'progression': progression,
                'driver_rating': self.driver_ratings.get(driver, 7.0),
                'team_strength': self.team_strength.get(team, 0.95),
                'spanish_bonus': self.spanish_gp_bonus.get(driver, 0)
            }
        
        self.performance_data = driver_performance
        print(f"✅ Performance calculated for {len(driver_performance)} drivers")
        return self
    
    def predict_race_positions(self):
        """Make realistic race predictions for 2025 grid"""
        print("🏁 Generating realistic race predictions for 2025 Spanish GP...")
        
        predictions = []
        
        # Get fastest lap time as reference
        all_times = [perf['best_time'] for perf in self.performance_data.values()]
        fastest_time = min(all_times)
        
        for driver, perf in self.performance_data.items():
            # Base score from practice times (gap to fastest)
            time_gap = perf['best_time'] - fastest_time
            
            # Convert to position prediction (smaller gaps = better positions)
            base_position = time_gap * 12  # Scale factor
            
            # Driver skill adjustment
            skill_adjustment = (9.5 - perf['driver_rating']) * 1.8
            
            # Team strength factor (inverted - better teams get negative adjustment)
            team_adjustment = (1.12 - perf['team_strength']) * 12
            
            # Spanish GP track specialist bonus (negative = better position)
            track_adjustment = -perf['spanish_bonus'] * 4
            
            # Consistency factor (more consistent = better race position)
            consistency_penalty = max(0, (perf['consistency'] - 0.15) * 8)
            
            # Long run pace factor (crucial for race)
            long_run_gap = perf['long_run_pace'] - fastest_time - 0.3
            long_run_adjustment = max(0, long_run_gap * 10)
            
            # Final predicted position
            predicted_pos = (1 + base_position + skill_adjustment + team_adjustment + 
                           track_adjustment + consistency_penalty + long_run_adjustment)
            
            # Add controlled randomness
            predicted_pos += np.random.normal(0, 0.4)
            
            # Ensure realistic bounds
            predicted_pos = max(1, min(20, predicted_pos))
            
            predictions.append({
                'Driver': driver,
                'Team': perf['team'],
                'Predicted_Position': predicted_pos,
                'Driver_Rating': perf['driver_rating'],
                'Spanish_Bonus': perf['spanish_bonus'],
                'Team_Strength': perf['team_strength'],
                'Best_Time': perf['best_time']
            })
        
        # Create results dataframe and sort
        self.results = pd.DataFrame(predictions)
        self.results = self.results.sort_values('Predicted_Position').reset_index(drop=True)
        self.results['Position'] = range(1, len(self.results) + 1)
        
        return self
    
    def display_predictions(self):
        """Display realistic 2025 Spanish GP predictions"""
        print("\n" + "="*70)
        print("🏁 REALISTIC SPANISH GP 2025 RACE PREDICTIONS 🏁")
        print("🔧 Based on ACTUAL 2025 F1 Grid + Track History + Current Form")
        print("="*70)
        
        medals = ['🥇', '🥈', '🥉']
        
        for i, row in self.results.head(20).iterrows():
            position = row['Position']
            driver = row['Driver']
            team = row['Team']
            rating = row['Driver_Rating']
            bonus = row['Spanish_Bonus'] if 'Spanish_Bonus' in row else 0
            team_str = row['Team_Strength']
            
            medal = medals[i] if i < 3 else '🏁'
            
            # Indicators for special factors
            indicators = ""
            if bonus >= 0.4:
                indicators += "🔥"  # Verstappen's Spanish GP dominance
            elif bonus >= 0.2:
                indicators += "⭐"  # Significant track bonus
            elif bonus > 0:
                indicators += "✨"  # Minor track bonus
            
            if team_str >= 1.08:
                indicators += "💪"  # Top team
            elif team_str >= 1.02:
                indicators += "🔧"  # Strong team
                
            if rating >= 9.0:
                indicators += "👑"  # Elite driver
            elif driver in ['HAM', 'ALO']:  # Legends
                indicators += "🐐"
            
            # Team colors for display
            team_short = team.replace(' F1 Team', '').replace(' Racing', '')
            
            print(f"{medal} P{position:2d}: {driver:3s} ({team_short:15s}) {indicators:8s}")
        
        print("\n🔍 Legend:")
        print("🔥 = Spanish GP Domination (Verstappen - 3 wins)")
        print("⭐ = Major Track Specialist | ✨ = Track Knowledge")
        print("💪 = Elite Team | 🔧 = Strong Team")
        print("👑 = Elite Current Driver | 🐐 = F1 Legend")
        
        # Show key insights
        print(f"\n💡 2025 Spanish GP Insights:")
        winner = self.results.iloc[0]
        print(f"🏆 Predicted winner: {winner['Driver']} ({winner['Team']})")
        
        podium = self.results.head(3)
        podium_teams = podium['Team'].unique()
        print(f"🏅 Podium teams: {', '.join(podium_teams)}")
        
        verstappen_pos = self.results[self.results['Driver'] == 'VER']['Position'].iloc[0]
        hamilton_pos = self.results[self.results['Driver'] == 'HAM']['Position'].iloc[0]
        print(f"🔥 Verstappen (3x Spanish GP winner): P{verstappen_pos}")
        print(f"🐐 Hamilton (Ferrari debut season): P{hamilton_pos}")
        
        mclaren_drivers = self.results[self.results['Team'] == 'McLaren'][['Driver', 'Position']]
        print(f"🧡 McLaren: {mclaren_drivers.iloc[0]['Driver']} P{mclaren_drivers.iloc[0]['Position']}, {mclaren_drivers.iloc[1]['Driver']} P{mclaren_drivers.iloc[1]['Position']}")
        
        return self

def main():
    """Run the realistic 2025 Spanish GP prediction"""
    print("🏎️  SPANISH GP 2025 PREDICTION - ACTUAL F1 GRID")
    print("="*55)
    print("🔄 Updated with real 2025 driver lineup!")
    print("📋 Hamilton to Ferrari | Lawson to Red Bull | Antonelli to Mercedes")
    print("="*55)
    
    try:
        predictor = RealisticSpanishGPPredictor()
        
        # Run prediction pipeline
        (predictor
         .load_data('practice_data.csv')
         .calculate_practice_performance()
         .predict_race_positions()
         .display_predictions())
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()