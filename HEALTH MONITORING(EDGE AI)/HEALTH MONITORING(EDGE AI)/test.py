import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ------------------------------
# Load TFLite Model
# ------------------------------
interpreter = tf.lite.Interpreter(model_path="health_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ Model Loaded")
print("Input details:", input_details)
print("Output details:", output_details)

# ------------------------------
# Scaler Parameters (replace with your real values)
# ------------------------------
feature_means = np.array([119.86852921,  94.99273024,  37.53872393 , 18.06759055])  # Example
feature_stds  = np.array([34.50526787,  3.60296324 , 2.42256493,  4.37493476])    # Example

# ------------------------------
# Test Sample (dummy sensor input)
# ------------------------------
sample = np.array([53.0,  93.0, 39.0, 19.1], dtype=np.float32)   

# Apply scaling
sample_scaled = (sample - feature_means) / feature_stds
sample_scaled = sample_scaled.reshape(1, -1).astype(np.float32)

print("\nOriginal Sample:", sample)
print("Scaled Sample  :", sample_scaled)

# ------------------------------
# Run Inference
# ------------------------------
interpreter.set_tensor(input_details[0]['index'], sample_scaled)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

print("\nPrediction Probabilities:", output_data)
print("Predicted Class        :", predicted_class)
