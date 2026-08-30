from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/quantized/"
        "ds_cnn_v1_int8.tflite"
    ).expanduser()
)

CACHE_DIR = (
    Path(
        "~/projects/tinyml-esp32/cache"
    ).expanduser()
)


# ============================================================
# Load test data
# ============================================================

test_x = np.load(
    CACHE_DIR / "test_features.npy",
    mmap_mode="r",
)

test_y = np.load(
    CACHE_DIR / "test_labels.npy",
    mmap_mode="r",
)


# ============================================================
# Create TFLite interpreter
# ============================================================

interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()


input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


print("=" * 60)
print("INT8 TFLite Evaluation")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "\nInput details:"
)

print(input_details)

print(
    "\nOutput details:"
)

print(output_details)


# ============================================================
# Read quantization parameters
# ============================================================

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)


print("\nQuantization:")

print(
    "Input scale:",
    input_scale,
)

print(
    "Input zero point:",
    input_zero_point,
)

print(
    "Output scale:",
    output_scale,
)

print(
    "Output zero point:",
    output_zero_point,
)


if input_scale == 0:
    raise RuntimeError(
        "Invalid input quantization scale."
    )


# ============================================================
# Evaluate
# ============================================================

correct = 0

num_samples = len(test_y)

for i in range(num_samples):

    # Float32 normalized MFCC
    sample = test_x[
        i:i + 1
    ].astype(np.float32)

    # Float -> INT8
    quantized_sample = np.round(
        sample / input_scale
        + input_zero_point
    )

    quantized_sample = np.clip(
        quantized_sample,
        -128,
        127,
    ).astype(np.int8)


    interpreter.set_tensor(
        input_details[0]["index"],
        quantized_sample,
    )

    interpreter.invoke()


    output = interpreter.get_tensor(
        output_details[0]["index"]
    )


    # INT8 output -> float
    dequantized_output = (
        output.astype(np.float32)
        - output_zero_point
    ) * output_scale


    prediction = int(
        np.argmax(
            dequantized_output
        )
    )


    true_label = int(
        test_y[i]
    )


    if prediction == true_label:
        correct += 1


accuracy = (
    correct / num_samples
)


# ============================================================
# Result
# ============================================================

print("\n" + "=" * 60)
print("INT8 Result")
print("=" * 60)

print(
    "Correct:",
    correct,
    "/",
    num_samples,
)

print(
    f"Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    "FP32 reference: 75.80%"
)

print(
    f"Accuracy drop: "
    f"{(0.758019 - accuracy) * 100:.2f} percentage points"
)
