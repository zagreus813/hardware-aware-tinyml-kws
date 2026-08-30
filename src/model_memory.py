from pathlib import Path

import tensorflow as tf


MODEL_PATHS = {
    "dense": (
        Path(
            "~/projects/tinyml-esp32/"
            "experiments/baseline_dense.h5"
        ).expanduser()
    ),

    "cnn_v2": (
        Path(
            "~/projects/tinyml-esp32/"
            "experiments/small_cnn_v2.h5"
        ).expanduser()
    ),

    "ds_cnn_v1": (
        Path(
            "~/projects/tinyml-esp32/"
            "experiments/ds_cnn_v1.h5"
        ).expanduser()
    ),
}


def tensor_memory_bytes(
    shape,
    bytes_per_element=4,
):
    """
    Estimate memory required by a tensor
    assuming dense storage.
    """

    elements = 1

    for dimension in shape:
        if dimension is None:
            continue

        elements *= int(dimension)

    return (
        elements
        * bytes_per_element
    )


def analyze_model(name, path):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = tf.keras.models.load_model(
        path
    )

    peak_activation = 0
    peak_layer = None

    print("\nLayer activation memory:")

    for layer in model.layers:

        try:
            output_shape = (
                layer.output.shape
            )

            memory = tensor_memory_bytes(
                output_shape,
                bytes_per_element=4,
            )

        except Exception:
            continue

        if memory > peak_activation:
            peak_activation = memory
            peak_layer = layer.name

        print(
            f"{layer.name:30s}"
            f"{str(output_shape):25s}"
            f"{memory / 1024:10.2f} KB"
        )

    print("\nPeak activation:")
    print(
        f"Layer: {peak_layer}"
    )

    print(
        f"FP32 memory: "
        f"{peak_activation / 1024:.2f} KB"
    )

    print(
        f"INT8 memory: "
        f"{peak_activation / 1024 / 4:.2f} KB"
    )


if __name__ == "__main__":

    for name, path in MODEL_PATHS.items():

        if not path.exists():

            print(
                "Missing:",
                path
            )

            continue

        analyze_model(
            name,
            path
        )
