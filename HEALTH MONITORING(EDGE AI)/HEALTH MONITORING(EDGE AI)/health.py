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

# ML / DL imports
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Features and Labels
X = health[['heart_rate','blood_oxygen','body_temp','resp_rate']].values
y = health['is_tampered'].values  

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

scaler_mean = scaler.mean_[0]
scaler_std = np.sqrt(scaler.var_[0])
print(f"\nScaler params -> mean: {scaler_mean:.3f}, std: {scaler_std:.3f}")

# Step 5: One-hot encode labels (num_classes auto-detected)
num_classes = len(np.unique(y))
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# Step 6: Lightweight NN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # match features
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train with early stopping
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train_cat,
    epochs=300,
    batch_size=8,
    verbose=1,
    validation_data=(X_test_scaled, y_test_cat),
    callbacks=[callback]
)

# Step 8: Evaluate
loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---- FIXED CLASSIFICATION REPORT ----
labels = np.unique(y)  # detect how many classes actually exist
label_map = {0: "SAFE", 1: "UNSAFE"}  # extendable if 3 classes appear

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    labels=labels,
    target_names=[label_map[i] for i in labels],
    zero_division=0
))

# Step 9: Convert to Float32 TFLite (ESP32 safe)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("health_model1.tflite", "wb") as f:
    f.write(tflite_model)

print("\n🚀 Float32 TFLite model saved as 'health_model1.tflite'")

# ------------------------------
# Step 10: Quick test with TFLite model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="health_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(sample_row):
    """
    sample_row: list or array with 4 features [heart_rate, blood_oxygen, body_temp, resp_rate]
    """
    sample_scaled = scaler.transform([sample_row]).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample_scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_class = int(np.argmax(output))
    return pred_class, output

# Example test
test_samples = [
    [70.0, 100.2, 36.5, 18.0],
    [120.0, 88.0, 39.2, 25.0],
    [89.56, 91.2, 45.08 , 13.25]
]

print("\n🔍 Quick Test Predictions:")
for row in test_samples:
    pred_class, probs = predict_tflite(row)
    print(f"Input={row} -> Class={label_map[pred_class]}, Probabilities={probs}")
