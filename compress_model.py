import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model("models/leaf_model.keras")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save compressed model
with open("models/leaf_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")