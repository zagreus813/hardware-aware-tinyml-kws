import numpy as np

from dataset import (
    get_split_files,
    process_file,
    load_normalization_stats,
)


# ------------------------------------------------------------
# Load training normalization statistics
# ------------------------------------------------------------

mean, std = load_normalization_stats()


# ------------------------------------------------------------
# First training sample
# ------------------------------------------------------------

train_files = get_split_files("train")
path = train_files[0]

print("Testing:", path)


# ------------------------------------------------------------
# Direct preprocessing
# ------------------------------------------------------------

direct_x, direct_y = process_file(
    path,
    normalize=True,
    mean=mean,
    std=std,
)


# ------------------------------------------------------------
# Cached preprocessing
# ------------------------------------------------------------

cached_x = np.load(
    "../cache/train_features.npy",
    mmap_mode="r",
)

cached_y = np.load(
    "../cache/train_labels.npy",
    mmap_mode="r",
)


cached_x0 = cached_x[0]
cached_y0 = cached_y[0]


# ------------------------------------------------------------
# Compare labels
# ------------------------------------------------------------

print("\nLabels:")
print("Direct:", direct_y)
print("Cache :", cached_y0)


# ------------------------------------------------------------
# Compare shapes
# ------------------------------------------------------------

print("\nShapes:")
print("Direct:", direct_x.shape)
print("Cache :", cached_x0.shape)


# ------------------------------------------------------------
# Compare feature values
# ------------------------------------------------------------

difference = np.abs(
    direct_x - cached_x0
)

print("\nFeature comparison:")

print(
    "Max absolute difference:",
    float(difference.max()),
)

print(
    "Mean absolute difference:",
    float(difference.mean()),
)

print(
    "Direct min:",
    float(direct_x.min()),
)

print(
    "Cache min:",
    float(cached_x0.min()),
)

print(
    "Direct max:",
    float(direct_x.max()),
)

print(
    "Cache max:",
    float(cached_x0.max()),
)

print(
    "All close:",
    np.allclose(
        direct_x,
        cached_x0,
        rtol=1e-5,
        atol=1e-6,
    ),
)


# ------------------------------------------------------------
# Final validation
# ------------------------------------------------------------

assert direct_y == cached_y0
assert direct_x.shape == cached_x0.shape

assert np.allclose(
    direct_x,
    cached_x0,
    rtol=1e-5,
    atol=1e-6,
)

print("\nCache verification PASSED.")
