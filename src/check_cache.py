import numpy as np
from collections import Counter


for split in [
    "train",
    "validation",
    "test",
]:
    x = np.load(
        f"../cache/{split}_features.npy",
        mmap_mode="r",
    )

    y = np.load(
        f"../cache/{split}_labels.npy",
        mmap_mode="r",
    )

    print("=" * 60)
    print(split)

    print("X shape:", x.shape)
    print("X dtype:", x.dtype)

    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    print(
        "label distribution:",
        Counter(y.tolist()),
    )

    print(
        "feature min:",
        float(x.min()),
    )

    print(
        "feature max:",
        float(x.max()),
    )

    print(
        "feature mean:",
        float(x.mean()),
    )

    print(
        "feature std:",
        float(x.std()),
    )
