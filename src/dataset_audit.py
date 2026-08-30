from pathlib import Path
from collections import Counter
import wave


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_ROOT = Path(
    "~/projects/tinyml-esp32/data/speech_commands_v0.02"
).expanduser()

TARGET_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
]

SPLIT_FILES = {
    "validation": DATASET_ROOT / "validation_list.txt",
    "test": DATASET_ROOT / "testing_list.txt",
}


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def read_split_file(path):
    """
    Read a Speech Commands split file.

    Each line contains a relative path such as:
        yes/abcd_nohash_0.wav
    """
    with open(path, "r") as file:
        return {
            line.strip()
            for line in file
            if line.strip()
        }


def get_all_wavs(class_name):
    """
    Return all WAV files for a given class.
    """
    class_dir = DATASET_ROOT / class_name

    return sorted(class_dir.glob("*.wav"))


def inspect_audio(path):
    """
    Read basic WAV properties.
    """
    with wave.open(str(path), "rb") as wav:

        return {
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "frames": wav.getnframes(),
        }


# --------------------------------------------------
# Dataset statistics
# --------------------------------------------------

print("=" * 60)
print("Speech Commands Dataset Audit")
print("=" * 60)

print("\nDataset root:")
print(DATASET_ROOT)

print("\nTarget classes:")
for cls in TARGET_CLASSES:
    print(f"  - {cls}")


# --------------------------------------------------
# Count samples per class
# --------------------------------------------------

print("\n" + "=" * 60)
print("Class Distribution")
print("=" * 60)

class_counts = {}

for cls in TARGET_CLASSES:

    files = get_all_wavs(cls)
    class_counts[cls] = len(files)

    print(f"{cls:>8}: {len(files):>6} samples")


# --------------------------------------------------
# Read official validation/test split lists
# --------------------------------------------------

validation_files = read_split_file(
    SPLIT_FILES["validation"]
)

test_files = read_split_file(
    SPLIT_FILES["test"]
)

print("\n" + "=" * 60)
print("Official Split Lists")
print("=" * 60)

print(
    "Validation entries:",
    len(validation_files)
)

print(
    "Test entries:",
    len(test_files)
)


# --------------------------------------------------
# Count split samples for target classes
# --------------------------------------------------

print("\n" + "=" * 60)
print("Target Class Split Distribution")
print("=" * 60)

for split_name, split_files in [
    ("validation", validation_files),
    ("test", test_files),
]:

    counter = Counter()

    for relative_path in split_files:

        class_name = Path(
            relative_path
        ).parts[0]

        if class_name in TARGET_CLASSES:
            counter[class_name] += 1

    print(f"\n{split_name.upper()}:")

    for cls in TARGET_CLASSES:
        print(
            f"{cls:>8}: "
            f"{counter[cls]:>6}"
        )


# --------------------------------------------------
# Inspect audio properties
# --------------------------------------------------

print("\n" + "=" * 60)
print("Audio Properties")
print("=" * 60)

reference_file = None

for cls in TARGET_CLASSES:

    files = get_all_wavs(cls)

    if files:
        reference_file = files[0]
        break


if reference_file is not None:

    info = inspect_audio(reference_file)

    print("\nReference file:")
    print(reference_file)

    print(
        "Sample rate:",
        info["sample_rate"],
        "Hz"
    )

    print(
        "Channels:",
        info["channels"]
    )

    print(
        "Sample width:",
        info["sample_width"],
        "bytes"
    )

    print(
        "Frames:",
        info["frames"]
    )

    duration = (
        info["frames"]
        / info["sample_rate"]
    )

    print(
        "Duration:",
        f"{duration:.4f}",
        "seconds"
    )


# --------------------------------------------------
# Check all target WAV files
# --------------------------------------------------

print("\n" + "=" * 60)
print("Audio Consistency Check")
print("=" * 60)

sample_rates = Counter()
channels = Counter()
durations = Counter()

total_files = 0
corrupted_files = []

for cls in TARGET_CLASSES:

    files = get_all_wavs(cls)

    for path in files:

        total_files += 1

        try:

            info = inspect_audio(path)

            sample_rates[
                info["sample_rate"]
            ] += 1

            channels[
                info["channels"]
            ] += 1

            duration = round(
                info["frames"]
                / info["sample_rate"],
                3,
            )

            durations[duration] += 1

        except Exception as exc:

            corrupted_files.append(
                (str(path), str(exc))
            )


print(
    "Total target WAV files:",
    total_files
)

print(
    "\nSample rates:"
)

for rate, count in sample_rates.items():
    print(
        f"  {rate} Hz: {count}"
    )

print(
    "\nChannels:"
)

for channel_count, count in channels.items():
    print(
        f"  {channel_count} channel(s): {count}"
    )

print(
    "\nMost common durations:"
)

for duration, count in durations.most_common(10):
    print(
        f"  {duration:.3f} sec: {count}"
    )

print(
    "\nCorrupted files:",
    len(corrupted_files)
)

if corrupted_files:

    print("\nExamples:")

    for path, error in corrupted_files[:10]:
        print(path)
        print("  ", error)


print("\n" + "=" * 60)
print("Audit Complete")
print("=" * 60)



# --------------------------------------------------
# Speaker leakage audit
# --------------------------------------------------

print("\n" + "=" * 60)
print("Speaker Leakage Audit")
print("=" * 60)


def get_speaker_id(relative_path):
    """
    Extract speaker ID from filenames such as:

        yes/004ae714_nohash_0.wav

    Speaker ID = 004ae714
    """
    filename = Path(relative_path).name
    return filename.split("_nohash_")[0]


# Build official split membership
validation_set = validation_files
test_set = test_files


train_files = set()

for cls in TARGET_CLASSES:

    for path in get_all_wavs(cls):

        relative_path = path.relative_to(
            DATASET_ROOT
        )

        relative_path = str(
            relative_path
        )

        if relative_path not in validation_set \
                and relative_path not in test_set:

            train_files.add(relative_path)


# Extract speakers for each split
train_speakers = {
    get_speaker_id(path)
    for path in train_files
}

validation_speakers = {
    get_speaker_id(path)
    for path in validation_set
    if Path(path).parts[0] in TARGET_CLASSES
}

test_speakers = {
    get_speaker_id(path)
    for path in test_set
    if Path(path).parts[0] in TARGET_CLASSES
}


print(
    "Train speakers:",
    len(train_speakers)
)

print(
    "Validation speakers:",
    len(validation_speakers)
)

print(
    "Test speakers:",
    len(test_speakers)
)


train_val_overlap = (
    train_speakers
    & validation_speakers
)

train_test_overlap = (
    train_speakers
    & test_speakers
)

val_test_overlap = (
    validation_speakers
    & test_speakers
)


print(
    "\nTrain ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if not train_val_overlap \
        and not train_test_overlap \
        and not val_test_overlap:

    print(
        "\nRESULT: No speaker overlap detected."
    )

else:

    print(
        "\nWARNING: Speaker overlap detected!"
    )
