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


def count_conv_macs(layer):
    """
    Compute MACs for a standard Conv2D layer.

    Assumes:
        stride = 1
        standard convolution
    """

    config = layer.get_config()

    kernel_h, kernel_w = (
        config["kernel_size"]
    )

    output_shape = layer.output.shape

    output_h = int(
        output_shape[1]
    )

    output_w = int(
        output_shape[2]
    )

    output_channels = int(
        output_shape[3]
    )

    input_channels = int(
        layer.input.shape[3]
    )

    macs = (
        output_h
        * output_w
        * output_channels
        * kernel_h
        * kernel_w
        * input_channels
    )

    return macs


def count_dense_macs(layer):
    """
    Compute MACs for Dense layer.
    """

    input_features = int(
        layer.input.shape[-1]
    )

    output_features = int(
        layer.output.shape[-1]
    )

    return (
        input_features
        * output_features
    )

def count_depthwise_conv_macs(layer):
    """
    Compute MACs for DepthwiseConv2D.

    For a depthwise convolution:

        MACs =
        H_out * W_out * C_in * K_h * K_w

    Each input channel has its own spatial kernel.
    """

    config = layer.get_config()

    kernel_h, kernel_w = (
        config["kernel_size"]
    )

    output_shape = layer.output.shape

    output_h = int(
        output_shape[1]
    )

    output_w = int(
        output_shape[2]
    )

    input_channels = int(
        layer.input.shape[3]
    )

    return (
        output_h
        * output_w
        * input_channels
        * kernel_h
        * kernel_w
    )
def analyze_model(name, path):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model = tf.keras.models.load_model(
        path
    )

    total_macs = 0

    print(
        "\nLayer complexity:"
    )

    for layer in model.layers:

        macs = 0

        if isinstance(
            layer,
            tf.keras.layers.Conv2D,
        ):

            macs = count_conv_macs(
                layer
            )
        elif isinstance(
            layer,
            tf.keras.layers.DepthwiseConv2D,
        ):

            macs = count_depthwise_conv_macs(
                layer
        )
        elif isinstance(
            layer,
            tf.keras.layers.Dense,
        ):

            macs = count_dense_macs(
                layer
            )

        if macs > 0:

            print(
                f"{layer.name:30s}"
                f"{macs:15,} MACs"
            )

            total_macs += macs

    print(
        "\nTotal parameters:",
        model.count_params(),
    )

    print(
        "Total MACs:",
        f"{total_macs:,}",
    )

    print(
        "Total MACs (millions):",
        f"{total_macs / 1e6:.3f} M",
    )


if __name__ == "__main__":

    for name, path in MODEL_PATHS.items():

        if not path.exists():

            print(
                "Missing:",
                path,
            )

            continue

        analyze_model(
            name,
            path,
        )
