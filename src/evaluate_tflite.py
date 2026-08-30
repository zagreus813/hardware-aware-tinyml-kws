from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

CACHE_DIR = (
    Path(
        "~/projects/tinyml-esp32/cache"
    ).expanduser()
)

QUANTIZED_DIR = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/quantized"
    ).expanduser()
)

FLOAT_MODEL = (
    QUANTIZED_DIR
    / "ds_cnn_v1_float.tflite"
)

INT8_MODEL = (
    QUANTIZED_DIR
    / "ds_cnn_v1_int8.tflite"
)

BATCH_SIZE = 1


# ============================================================
# Load test set
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
# Generic TFLite evaluator
# ============================================================

def evaluate_tflite(
    model_path,
    test_x,
    test_y,
):
    print("\n" + "=" * 60)
    print(
        f"Evaluating: {model_path.name}"
    )
    print("=" * 60)

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_info = input_details[0]
    output_info = output_details[0]

    print(
        "Input dtype:",
        input_info["dtype"],
    )

    print(
        "Output dtype:",
        output_info["dtype"],
    )

    print(
        "Input shape:",
        input_info["shape"],
    )

    print(
        "Output shape:",
        output_info["shape"],
    )

    print(
        "Input quantization:",
        input_info["quantization"],
    )

    print(
        "Output quantization:",
        output_info["quantization"],
    )

    input_scale, input_zero_point = (
        input_info["quantization"]
    )

    output_scale, output_zero_point = (
        output_info["quantization"]
    )

    input_dtype = input_info["dtype"]

    output_dtype = output_info["dtype"]

    correct = 0

    num_samples = len(test_y)

    for i in range(num_samples):

        sample = test_x[
            i:i + 1
        ].astype(np.float32)

        # ----------------------------------------------------
        # Quantize input if required
        # ----------------------------------------------------

        if input_dtype == np.float32:

            input_tensor = sample

        else:

            if input_scale == 0:
                raise RuntimeError(
                    "Invalid input quantization scale."
                )

            input_tensor = np.round(
                sample / input_scale
                + input_zero_point
            )

            if input_dtype == np.int8:

                input_tensor = np.clip(
                    input_tensor,
                    -128,
                    127,
                ).astype(np.int8)

            elif input_dtype == np.uint8:

                input_tensor = np.clip(
                    input_tensor,
                    0,
                    255,
                ).astype(np.uint8)

            else:

                raise RuntimeError(
                    f"Unsupported input dtype: "
                    f"{input_dtype}"
                )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        interpreter.set_tensor(
            input_info["index"],
            input_tensor,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        )

        # ----------------------------------------------------
        # Dequantize output if required
        # ----------------------------------------------------

        if output_dtype == np.float32:

            scores = output

        else:

            scores = (
                output.astype(np.float32)
                - output_zero_point
            ) * output_scale

        prediction = int(
            np.argmax(scores)
        )

        target = int(
            test_y[i]
        )

        if prediction == target:

            correct += 1

    accuracy = (
        correct / num_samples
    )

    print(
        "\nCorrect:",
        correct,
        "/",
        num_samples,
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    return accuracy


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TFLite FP32 vs INT8 Evaluation")
    print("=" * 60)

    print(
        "Test samples:",
        len(test_y),
    )

    fp32_accuracy = evaluate_tflite(
        FLOAT_MODEL,
        test_x,
        test_y,
    )

    int8_accuracy = evaluate_tflite(
        INT8_MODEL,
        test_x,
        test_y,
    )

    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)

    print(
        f"TFLite FP32: "
        f"{fp32_accuracy * 100:.4f}%"
    )

    print(
        f"TFLite INT8: "
        f"{int8_accuracy * 100:.4f}%"
    )

    print(
        f"INT8 - FP32: "
        f"{(int8_accuracy - fp32_accuracy) * 100:.4f} "
        f"percentage points"
    )
