# ============================================
# TASK 2 - Linear Regression: House Price Prediction
# QSkill Internship - Slab 1
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import os

print("=" * 55)
print("   TASK 2 — House Price Prediction (Linear Regression)")
print("=" * 55)

# ── Step 1: Create Synthetic Dataset (Kaggle-style) ──────────────────────────
np.random.seed(0)
n = 300

locations = ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"]
loc_price  = {"Mumbai": 15000, "Pune": 9000, "Nagpur": 5000,
              "Nashik": 4000, "Aurangabad": 3500}

rows = []
for _ in range(n):
    loc      = np.random.choice(locations)
    rooms    = np.random.randint(1, 6)
    size_sqft = rooms * np.random.randint(200, 350)
    age      = np.random.randint(0, 30)
    parking  = np.random.randint(0, 3)
    base     = loc_price[loc]
    price    = (base * size_sqft / 1000
                + rooms * 50000
                + parking * 30000
                - age * 5000
                + np.random.randint(-50000, 50000))
    price    = max(price, 200000)
    rows.append([rooms, loc, size_sqft, age, parking, round(price, -3)])

df = pd.DataFrame(rows, columns=["Rooms", "Location", "SizeSqft",
                                  "AgeYears", "Parking", "Price"])
csv_path = "house_prices.csv"
os.makedirs("outputs", exist_ok=True)
print(f"\n✅ Dataset created: {csv_path}  ({n} records)")
print(f"\n📊 Sample Data:")
print(df.head().to_string(index=False))

# ── Step 2: Preprocessing ────────────────────────────────────────────────────
print("\n🔧 Preprocessing...")
print(f"   Null values: {df.isnull().sum().sum()}")

le = LabelEncoder()
df["Location_enc"] = le.fit_transform(df["Location"])

features = ["Rooms", "Location_enc", "SizeSqft", "AgeYears", "Parking"]
X = df[features]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── Step 3: Train Model ───────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model trained!")

print("\n📌 Feature Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"   {feat:15s}: ₹{coef:,.0f}")
print(f"   {'Intercept':15s}: ₹{model.intercept_:,.0f}")

# ── Step 4: Evaluate ──────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation:")
print(f"   MAE  (Mean Absolute Error) : ₹{mae:,.0f}")
print(f"   RMSE (Root Mean Sq. Error) : ₹{rmse:,.0f}")
print(f"   R²   (Accuracy Score)      : {r2:.4f}  ({r2*100:.1f}%)")

# ── Step 5: Visualizations ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("House Price Prediction — Linear Regression", fontsize=14, fontweight="bold")

# Chart 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color="#4C72B0", edgecolors="gray", s=50)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")
axes[0].set_title("Actual vs Predicted Price")
axes[0].set_xlabel("Actual Price (₹)")
axes[0].set_ylabel("Predicted Price (₹)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Chart 2: Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=20, color="#DD8452", edgecolor="black", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
axes[1].set_title("Residuals Distribution")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].set_ylabel("Count")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out_path = "outputs/task2_regression.png"
os.makedirs("outputs", exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Chart saved: {out_path}")

# ── Step 6: Predict New House ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   🏠 PREDICT NEW HOUSE PRICE")
print("=" * 55)

new_houses = pd.DataFrame([
    {"Rooms": 3, "Location": "Pune",   "SizeSqft": 900, "AgeYears": 5,  "Parking": 1},
    {"Rooms": 2, "Location": "Mumbai", "SizeSqft": 650, "AgeYears": 10, "Parking": 0},
    {"Rooms": 4, "Location": "Nagpur", "SizeSqft": 1200,"AgeYears": 2,  "Parking": 2},
])
new_houses["Location_enc"] = le.transform(new_houses["Location"])
preds = model.predict(new_houses[features])

for i, (_, row) in enumerate(new_houses.iterrows()):
    print(f"\n   House {i+1}:")
    print(f"      Location  : {row['Location']}")
    print(f"      Rooms     : {int(row['Rooms'])}")
    print(f"      Size      : {int(row['SizeSqft'])} sq.ft")
    print(f"      Age       : {int(row['AgeYears'])} years")
    print(f"      Parking   : {int(row['Parking'])} spots")
    print(f"   💰 Predicted Price: ₹{preds[i]:,.0f}")

print("\n✅ Task 2 Complete!\n")
