import pandas as pd
import numpy as np

# 1. Load Raw Data
print("🧹 Starting Data Cleaning Pipeline...")
df = pd.read_csv("data/vehicles.csv")

# 2. Drop Useless Columns
# 'county' is empty. 'url'/'image_url'/'vin' are unique IDs that don't help predict price.
cols_to_drop = ['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county', 'lat', 'long', 'posting_date']
df = df.drop(columns=cols_to_drop)
print(f"✅ Dropped {len(cols_to_drop)} useless columns.")

# 3. Handle Missing Values
# For categorical columns (like 'manufacturer'), fill missing with 'unknown'
# For numerical columns (like 'odometer'), fill with the median value
df['manufacturer'] = df['manufacturer'].fillna('unknown')
df['model'] = df['model'].fillna('unknown')
df['fuel'] = df['fuel'].fillna('gas')
df['transmission'] = df['transmission'].fillna('automatic')
df['odometer'] = df['odometer'].fillna(df['odometer'].median())

# 4. Filter "Spam" Prices and Odometers
# We only want cars between $1,000 and $100,000.
# We only want cars with less than 300,000 miles.
df = df[(df['price'] > 1000) & (df['price'] < 100000)]
df = df[(df['odometer'] > 1000) & (df['odometer'] < 300000)]
print(f"✅ Filtered outliers. Rows remaining: {df.shape[0]}")

# 5. Feature Engineering: Calculate "Car Age"
# A 2010 car is 16 years old in 2026. This is more useful to AI than just "2010".
df['year'] = df['year'].fillna(2015) # Fill missing years with a median year
df['car_age'] = 2026 - df['year']
df = df.drop(columns=['year']) # We don't need 'year' anymore, we have 'age'

# 6. Save Clean Data
df.to_csv("data/vehicles_cleaned.csv", index=False)
print("💾 Saved clean dataset to 'data/vehicles_cleaned.csv'")
print("-" * 30)
print(df.head())