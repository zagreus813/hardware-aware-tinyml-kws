from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)


# ============================================================
# Paths
# ============================================================

CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

QUANTIZED_DIR = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/quantized"
    ).expanduser()
)

FP32_MODEL = (
    QUANTIZED_DIR
    / "ds_cnn_v1_float.tflite"
)

INT8_MODEL = (
    QUANTIZED_DIR
    / "ds_cnn_v1_int8.tflite"
)


CLASS_NAMES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
]


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
# Generic inference
# ============================================================

def predict_tflite(
    model_path,
    test_x,
):
    """
    Run inference on every test sample.

    Supports:
        FP32 input
        INT8 input
    """

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    input_info = (
        interpreter.get_input_details()[0]
    )

    output_info = (
        interpreter.get_output_details()[0]
    )

    input_scale, input_zero_point = (
        input_info["quantization"]
    )

    output_scale, output_zero_point = (
        output_info["quantization"]
    )

    input_dtype = input_info["dtype"]
    output_dtype = output_info["dtype"]

    predictions = []

    for i in range(len(test_x)):

        sample = test_x[
            i:i + 1
        ].astype(np.float32)

        # ----------------------------------------------------
        # Input quantization
        # ----------------------------------------------------

        if input_dtype == np.float32:

            input_tensor = sample

        elif input_dtype == np.int8:

            input_tensor = np.round(
                sample / input_scale
                + input_zero_point
            )

            input_tensor = np.clip(
                input_tensor,
                -128,
                127,
            ).astype(np.int8)

        elif input_dtype == np.uint8:

            input_tensor = np.round(
                sample / input_scale
                + input_zero_point
            )

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

        interpreter.set_tensor(
            input_info["index"],
            input_tensor,
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        )

        # ----------------------------------------------------
        # Output dequantization
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

        predictions.append(
            prediction
        )

    return np.asarray(
        predictions,
        dtype=np.int32,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FP32 vs INT8 Class-wise Comparison")
    print("=" * 60)

    print(
        "Test samples:",
        len(test_y),
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nRunning FP32 inference...")

    fp32_pred = predict_tflite(
        FP32_MODEL,
        test_x,
    )

    print(
        "Running INT8 inference..."
    )

    int8_pred = predict_tflite(
        INT8_MODEL,
        test_x,
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    fp32_accuracy = np.mean(
        fp32_pred == test_y
    )

    int8_accuracy = np.mean(
        int8_pred == test_y
    )

    print("\n" + "=" * 60)
    print("Overall Accuracy")
    print("=" * 60)

    print(
        f"FP32: {fp32_accuracy * 100:.4f}%"
    )

    print(
        f"INT8: {int8_accuracy * 100:.4f}%"
    )

    print(
        f"INT8 - FP32: "
        f"{(int8_accuracy - fp32_accuracy) * 100:.4f} "
        f"percentage points"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    fp32_cm = confusion_matrix(
        test_y,
        fp32_pred,
    )

    int8_cm = confusion_matrix(
        test_y,
        int8_pred,
    )

    print("\n" + "=" * 60)
    print("FP32 Confusion Matrix")
    print("=" * 60)

    print(fp32_cm)

    print("\n" + "=" * 60)
    print("INT8 Confusion Matrix")
    print("=" * 60)

    print(int8_cm)

    # --------------------------------------------------------
    # Classification reports
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FP32 Classification Report")
    print("=" * 60)

    print(
        classification_report(
            test_y,
            fp32_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    print("\n" + "=" * 60)
    print("INT8 Classification Report")
    print("=" * 60)

    print(
        classification_report(
            test_y,
            int8_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

    # --------------------------------------------------------
    # Per-sample prediction changes
    # --------------------------------------------------------

    changed = (
        fp32_pred != int8_pred
    )

    num_changed = int(
        changed.sum()
    )

    print("\n" + "=" * 60)
    print("Prediction Changes")
    print("=" * 60)

    print(
        "Changed predictions:",
        num_changed,
        "/",
        len(test_y),
    )

    print(
        f"Percentage changed: "
        f"{100 * num_changed / len(test_y):.4f}%"
    )

    # Correctness changes
    fp32_correct = (
        fp32_pred == test_y
    )

    int8_correct = (
        int8_pred == test_y
    )

    fp32_to_int8_worse = (
        fp32_correct
        & ~int8_correct
    )

    fp32_to_int8_better = (
        ~fp32_correct
        & int8_correct
    )

    print(
        "FP32 correct → INT8 wrong:",
        int(fp32_to_int8_worse.sum()),
    )

    print(
        "FP32 wrong → INT8 correct:",
        int(fp32_to_int8_better.sum()),
    )
