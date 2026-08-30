from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf

from dataset import (
    TARGET_CLASSES,
    create_tf_dataset,
    load_normalization_stats,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 1e-3

MODEL_DIR = (
    Path(
        "~/projects/tinyml-esp32/experiments"
    )
    .expanduser()
)

MODEL_PATH = (
    MODEL_DIR
    / "baseline_dense.h5"
)

HISTORY_PATH = (
    MODEL_DIR
    / "baseline_history.json"
)


# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load normalization statistics
# ============================================================

mean, std = load_normalization_stats()

print("=" * 60)
print("Baseline Dense Model")
print("=" * 60)

print(
    "Classes:",
    TARGET_CLASSES
)

print(
    "Number of classes:",
    len(TARGET_CLASSES)
)

print(
    "MFCC mean shape:",
    mean.shape
)

print(
    "MFCC std shape:",
    std.shape
)


# ============================================================
# Create TensorFlow datasets
# ============================================================

print("\nCreating datasets...")

train_ds = create_tf_dataset(
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    normalize=True,
    mean=mean,
    std=std,
)

val_ds = create_tf_dataset(
    split="validation",
    batch_size=BATCH_SIZE,
    shuffle=False,
    normalize=True,
    mean=mean,
    std=std,
)

test_ds = create_tf_dataset(
    split="test",
    batch_size=BATCH_SIZE,
    shuffle=False,
    normalize=True,
    mean=mean,
    std=std,
)


# ============================================================
# Inspect one batch
# ============================================================

print("\nInspecting one batch...")

sample_x, sample_y = next(
    iter(train_ds)
)

print(
    "Input shape:",
    sample_x.shape
)

print(
    "Label shape:",
    sample_y.shape
)

print(
    "Input dtype:",
    sample_x.dtype
)

print(
    "Label dtype:",
    sample_y.dtype
)

print(
    "Input min:",
    float(
        tf.reduce_min(sample_x)
    )
)

print(
    "Input max:",
    float(
        tf.reduce_max(sample_x)
    )
)


# ============================================================
# Build baseline model
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(13, 98, 1)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            len(TARGET_CLASSES),
            activation="softmax",
        ),
    ],
    name="baseline_dense",
)


# ============================================================
# Model summary
# ============================================================

print("\n")
model.summary()


# ============================================================
# Compile
# ============================================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
    ],
)


# ============================================================
# Callbacks
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,

    ),
]


# ============================================================
# Train
# ============================================================

print("\n")
print("=" * 60)
print("Training")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

training_time = (
    time.time()
    - start_time
)


# ============================================================
# Final test evaluation
# ============================================================

print("\n")
print("=" * 60)
print("Test Evaluation")
print("=" * 60)

test_loss, test_accuracy = (
    model.evaluate(
        test_ds,
        verbose=1,
    )
)


# ============================================================
# Save training history
# ============================================================

history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values in history.history.items()
}

history_data["training_time_seconds"] = (
    float(training_time)
)

history_data["test_loss"] = float(
    test_loss
)

history_data["test_accuracy"] = float(
    test_accuracy
)

with open(
    HISTORY_PATH,
    "w",
) as file:

    json.dump(
        history_data,
        file,
        indent=2,
    )


# ============================================================
# Final report
# ============================================================

print("\n")
print("=" * 60)
print("Baseline Result")
print("=" * 60)

print(
    f"Test accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print(
    "Best model:",
    MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)
