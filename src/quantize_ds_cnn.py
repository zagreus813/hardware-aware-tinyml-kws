from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

MODEL_PATH = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/ds_cnn_v1.h5"
    ).expanduser()
)

CACHE_DIR = (
    Path(
        "~/projects/tinyml-esp32/cache"
    ).expanduser()
)

OUTPUT_DIR = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/quantized"
    ).expanduser()
)

FLOAT_TFLITE_PATH = (
    OUTPUT_DIR
    / "ds_cnn_v1_float.tflite"
)

INT8_TFLITE_PATH = (
    OUTPUT_DIR
    / "ds_cnn_v1_int8.tflite"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "quantization_report.json"
)


# ============================================================
# Configuration
# ============================================================

REPRESENTATIVE_SAMPLES = 500

BATCH_SIZE = 64


# ============================================================
# Prepare output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("Loading DS-CNN v1")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

model.summary()


# ============================================================
# Representative dataset
# ============================================================

train_x = np.load(
    CACHE_DIR / "train_features.npy",
    mmap_mode="r",
)


def representative_dataset():
    """
    Yield representative calibration samples
    from the training split only.
    """

    for i in range(
        min(
            REPRESENTATIVE_SAMPLES,
            len(train_x),
        )
    ):

        sample = (
            train_x[i:i + 1]
            .astype(np.float32)
        )

        yield [sample]


# ============================================================
# Float32 TFLite conversion
# ============================================================

print("\n" + "=" * 60)
print("Converting FP32 TFLite model")
print("=" * 60)

converter = (
    tf.lite.TFLiteConverter
    .from_keras_model(model)
)

float_model = (
    converter.convert()
)

with open(
    FLOAT_TFLITE_PATH,
    "wb",
) as file:

    file.write(
        float_model
    )


print(
    "Saved:",
    FLOAT_TFLITE_PATH
)


# ============================================================
# INT8 conversion
# ============================================================

print("\n" + "=" * 60)
print("Converting INT8 TFLite model")
print("=" * 60)

converter = (
    tf.lite.TFLiteConverter
    .from_keras_model(model)
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = (
    tf.int8
)

converter.inference_output_type = (
    tf.int8
)


int8_model = (
    converter.convert()
)

with open(
    INT8_TFLITE_PATH,
    "wb",
) as file:

    file.write(
        int8_model
    )


print(
    "Saved:",
    INT8_TFLITE_PATH
)


# ============================================================
# File sizes
# ============================================================

float_size = (
    FLOAT_TFLITE_PATH.stat().st_size
)

int8_size = (
    INT8_TFLITE_PATH.stat().st_size
)


size_reduction = (
    1.0
    - int8_size / float_size
)


print("\n" + "=" * 60)
print("Model sizes")
print("=" * 60)

print(
    "FP32:",
    float_size,
    "bytes"
)

print(
    "INT8:",
    int8_size,
    "bytes"
)

print(
    f"Reduction: "
    f"{size_reduction * 100:.2f}%"
)


# ============================================================
# Save report
# ============================================================

report = {
    "source_model": str(
        MODEL_PATH
    ),

    "representative_samples": (
        REPRESENTATIVE_SAMPLES
    ),

    "float_tflite_bytes": (
        float_size
    ),

    "int8_tflite_bytes": (
        int8_size
    ),

    "size_reduction_fraction": (
        size_reduction
    ),
}


with open(
    REPORT_PATH,
    "w",
) as file:

    json.dump(
        report,
        file,
        indent=2,
    )


print(
    "\nReport:",
    REPORT_PATH
)
