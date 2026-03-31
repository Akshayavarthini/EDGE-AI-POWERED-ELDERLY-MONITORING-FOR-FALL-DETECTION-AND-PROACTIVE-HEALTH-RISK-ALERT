# import library
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import csv files
health = pd.read_csv("biometric_wearable_dataset.csv")
print(health)

# Drop unnecessary columns
drop_cols = ["timestamp","step_count","activity","accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
health = health.drop(columns=drop_cols)

# remove duplicates
health = health.drop_duplicates()
print(health)

# check dataset info
print(health.info())
print(health.describe())
print(health.describe(include="all"))
print(health.isnull().sum())
print("Duplicates:", health.duplicated().sum())
print("Unique values per column:\n", health.nunique())
print(health.columns)
print(health.sample(5))

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Features and Labels
X = health[['heart_rate','blood_oxygen','body_temp','resp_rate']].values
y = health['is_tampered'].values  


# ------------------------------
# 2. Split into train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 3. Fit StandardScaler
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 4. Print mean and std
# ------------------------------
print("Scaler mean:", scaler.mean_[0])
print("Scaler std :", scaler.scale_[0])

# Print full scaler mean and std arrays
print("Full Scaler Means:", scaler.mean_)
print("Full Scaler Stds :", scaler.scale_)