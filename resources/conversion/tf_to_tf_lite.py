import tensorflow as tf

# Load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model("gen_ppo_tf")

# Convert the model
tflite_model = converter.convert()

# Save to file
with open("gen_ppo.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as gen_ppo.tflite")
