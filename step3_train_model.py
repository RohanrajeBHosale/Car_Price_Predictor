import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

# 1. Load Clean Data
print("🧠 Loading clean data for training...")
df = pd.read_csv("data/vehicles_cleaned.csv")

# 2. Select Features (The columns we want to use to predict Price)
# We drop 'region' and 'state' to keep the model fast for this MVP.
features = ['manufacturer', 'fuel', 'odometer', 'transmission', 'drive', 'type', 'car_age', 'cylinders']
target = 'price'

# Keep only relevant columns
df = df[features + [target]]

# 3. Convert Text to Numbers (One-Hot Encoding)
# AI cannot understand "Ford". It needs "1" or "0".
print("⚙️ Encoding categorical data...")
df_encoded = pd.get_dummies(df, columns=['manufacturer', 'fuel', 'transmission', 'drive', 'type', 'cylinders'])

# 4. Split Data (80% for Training, 20% for Testing)
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model (XGBoost)
print("🚀 Training XGBoost Model... (This may take a minute)")
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=7)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"✅ Model Trained Successfully!")
print(f"📉 Mean Absolute Error: ${mae:.2f}")
print(f"📊 Accuracy (R2 Score): {r2:.2f}")
print("-" * 30)

# 7. Save the Model (So we can load it in the app later)
# We also save the 'columns' so we know the order of features for the app.
joblib.dump(model, "data/car_price_model.pkl")
joblib.dump(list(X.columns), "data/model_columns.pkl")
print("💾 Model saved to 'data/car_price_model.pkl'")