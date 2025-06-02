import requests
import json
from datetime import datetime

# === Hardcoded for testing ===
API_KEY ="d832c9f4ce3e30672c489cc70ba5ee18"  

# === Setup for Spanish GP (Barcelona) ===
LATITUDE = 41.57
LONGITUDE = 2.26
UNITS = "metric"

# === Forecast target datetime (race time) ===
# Spanish GP 2025 is on June 1, Sunday, race at 15:00 local time (CEST)
forecast_time_str = "2025-06-01 13:00:00"  # in UTC (Barcelona is UTC+2 in summer)

# === Fetch forecast from OpenWeatherMap ===
url = (
    f"http://api.openweathermap.org/data/2.5/forecast"
    f"?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}&units={UNITS}"
)

response = requests.get(url)
weather_data = response.json()

if response.status_code != 200:
    raise ValueError(f"Failed to fetch weather data: {weather_data}")

# === Extract forecast for race time ===
forecast_data = next((entry for entry in weather_data["list"] if entry["dt_txt"] == forecast_time_str), None)

if not forecast_data:
    raise ValueError(f"No weather forecast found for {forecast_time_str}")

# === Print out relevant info ===
main = forecast_data["main"]
weather = forecast_data["weather"][0]
wind = forecast_data["wind"]

print("\n🌤️ Weather Forecast for 2025 Spanish GP (Race Time)")
print("----------------------------------------------------")
print(f"Description:     {weather['description'].title()}")
print(f"Temperature:     {main['temp']}°C")
print(f"Feels Like:      {main['feels_like']}°C")
print(f"Humidity:        {main['humidity']}%")
print(f"Pressure:        {main['pressure']} hPa")
print(f"Wind Speed:      {wind['speed']} m/s")
print(f"Rain Probability:{forecast_data.get('pop', 0) * 100:.0f}%")
print("----------------------------------------------------")