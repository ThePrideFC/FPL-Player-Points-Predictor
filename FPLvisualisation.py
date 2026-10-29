import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
url = "https://fantasy.premierleague.com/api/bootstrap-static/"
import requests
import matplotlib.pyplot as plt
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
raw_data = requests.get(url, headers = headers).json()
df = pd.DataFrame(raw_data["elements"])
df = df[df["minutes"]> 180].copy()
df["expected_goals"] = df["expected_goals"].astype(float)
df["expected_assists"] = df["expected_assists"].astype(float)
df["minutes"] = df["minutes"].astype(float)
df["clean_sheets"] = df["minutes"].astype(float)
df["goals_conceded"] = df["goals_conceded"].astype(float)
df["saves"] = df["saves"].astype(float)
df["bonus"] = df["bonus"].astype(float)
df["penalties_saved"] = df["penalties_saved"].astype(float)
df["own_goals"] = df["own_goals"].astype(float)
df["penalties_missed"] = df["penalties_missed"].astype(float)
df["yellow_cards"] = df["yellow_cards"].astype(float)
df["red_cards"] = df["red_cards"].astype(float)
df["total_points"] = df["total_points"].astype(float)
x = df[["expected_goals","expected_assists", "minutes", "clean_sheets", "goals_conceded", "saves", "bonus", "penalties_saved", "own_goals", "penalties_missed", "yellow_cards", "red_cards"]]
y = df["total_points"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x)
print("===MODEL WEIGHTS===")
print(f"Base intercept points: {model.intercept_:.2f}")
print(f"Points gained per 1.0xG: {model.coef_[0]:.2f}")
print(f"Points gained per 1.0xA: {model.coef_[1]:.2f}")
results = pd.DataFrame({"Player": df["web_name"].values,
                        "Real Points": y,
                        "Predicted Points":predictions.round(1)})
results["Value Gap"] = (results["Predicted Points"] - results["Real Points"]).round(1)
results = results.sort_values(by = "Value Gap", ascending = False)
print("\n === SAMPLE PREDICTIONS === ")
print(results.head(10))
chart_data = results.head(15)
plt.figure(figsize = (12, 6))
x_indexes = range(len(chart_data))
bar_width = 0.35
plt.bar([i - bar_width/2 for i in x_indexes], chart_data["Real Points"],
        width = bar_width, label = "Real Points", color = "#38003c")
plt.bar([i + bar_width/2 for i in x_indexes], chart_data["Predicted Points"],
        width = bar_width, label = "Predicted Points", color = "#00ff87")
plt.title("FPL Analytics: Real vs Predicted Points (Top Hidden Gems Fist)", fontsize = 14, fontweight = "bold")
plt.xlabel("Player Name", fontsize = 11)
plt.ylabel("Total Season Points", fontsize = 11)
plt.xticks(x_indexes, chart_data["Player"], rotation = 45, ha = "right")
plt.legend()
plt.tight_layout()
plt.show()