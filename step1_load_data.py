import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
# low_memory=False is needed because the file is large and has mixed types
print("⏳ Loading dataset... (this might take a moment)")
df = pd.read_csv("data/vehicles.csv")

# 2. Basic Inspection
print(f"✅ Data Loaded! Shape: {df.shape}")
print("-" * 30)
print("First 5 rows:")
print(df.head())

# 3. Check for Missing Values
# This is crucial. Real data is full of "NaN" (Not a Number).
print("-" * 30)
print("Missing Values per Column:")
print(df.isnull().sum())

# 4. Visualize the Price Distribution (The Target)
# We want to predict 'price'. Let's see what it looks like.
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution of Car Prices (Raw Data)")
plt.xlabel("Price")
plt.show()