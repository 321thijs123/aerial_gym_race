import numpy as np
import onnxruntime as ort
import tensorflow as tf

# Input settings
input_shape = (1, 33)
input_data = np.zeros(input_shape, dtype=np.float32)

input_data = np.array([[19.84083, -0.86518, -0.49538, 0.07780, 0.49663, -0.86795, -0.00382, -0.00166, -0.00208, 0.01674, -0.00131, -0.00500, -0.00052, 25.00162, 0.00134, 1.84045, 0.00000, 1.00000, 20.00162, 5.00135, 1.84045, -1.00000, 0.00000, 15.00162, 0.00134, 1.84045, 0.00000, -1.00000, 20.00162, -4.99865, 1.84045, 0.00000, 1.00000]], dtype=np.float32)


# --- Run ONNX model ---
onnx_session = ort.InferenceSession("gen_ppo.onnx", providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output = onnx_session.run(None, {onnx_input_name: input_data})[0]
print("ONNX Output:", onnx_output)

# --- Run TFLite model ---
interpreter = tf.lite.Interpreter(model_path="gen_ppo.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details[0]['index'])
print("TFLite Output:", tflite_output)

# --- Compare outputs ---
difference = np.abs(onnx_output - tflite_output)
print("Absolute difference:", difference)
print("Max difference:", np.max(difference))

