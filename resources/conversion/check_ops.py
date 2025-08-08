import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="gen_ppo.tflite")
interpreter.allocate_tensors()

ops = interpreter._get_ops_details()
for i, op in enumerate(ops):
    print(f"Op #{i}: {op['op_name']}")

