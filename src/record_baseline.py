from pathlib import Path
import json


OUTPUT = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/baseline_summary.json"
    ).expanduser()
)


results = {
    "dense": {
        "parameters": 163974,
        "macs": 163840,
        "test_accuracy": 0.819732,
    },

    "cnn_v2": {
        "parameters": 15446,
        "macs": 2202976,
        "test_accuracy": 0.614697,
    },

    "ds_cnn_v1": {
        "parameters": 4966,
        "macs": 509376,
        "test_accuracy": 0.758019,
        "peak_activation_fp32_kb": 79.62,
    },
}


with open(
    OUTPUT,
    "w",
) as file:
    json.dump(
        results,
        file,
        indent=2,
    )


print(
    "Saved:",
    OUTPUT,
)
