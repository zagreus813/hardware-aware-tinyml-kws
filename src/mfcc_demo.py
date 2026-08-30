import numpy as np
import matplotlib.pyplot as plt


FS = 16000
DURATION = 1.0

FRAME_LENGTH = 400
HOP_LENGTH = 160

N_MELS = 40
N_MFCC = 13

F_MIN = 20
F_MAX = 4000


def hz_to_mel(freq):
    return 2595 * np.log10(1 + freq / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def build_mel_filterbank(
    n_fft,
    n_mels,
    f_min,
    f_max,
    sample_rate,
):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor(
        (n_fft + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    filters = np.zeros(
        (n_mels, n_fft // 2 + 1)
    )

    for m in range(1, n_mels + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        for k in range(left, center):
            if center > left:
                filters[m - 1, k] = (
                    (k - left)
                    / (center - left)
                )

        for k in range(center, right):
            if right > center:
                filters[m - 1, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters


# --------------------------------------------------
# Generate synthetic signal
# --------------------------------------------------

t = np.arange(
    0,
    DURATION,
    1 / FS
)

signal = (
    np.sin(2 * np.pi * 440 * t)
    + 0.5 * np.sin(2 * np.pi * 1000 * t)
)


# --------------------------------------------------
# STFT
# --------------------------------------------------

window = np.hanning(FRAME_LENGTH)

frames = []

for start in range(
    0,
    len(signal) - FRAME_LENGTH + 1,
    HOP_LENGTH,
):

    frame = signal[
        start:start + FRAME_LENGTH
    ]

    frame = frame * window

    spectrum = np.fft.rfft(frame)

    # Power spectrum
    power = np.abs(spectrum) ** 2

    frames.append(power)


power_spectrogram = np.array(frames).T


# --------------------------------------------------
# Mel filter bank
# --------------------------------------------------

mel_filters = build_mel_filterbank(
    FRAME_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    FS,
)


# --------------------------------------------------
# Apply Mel filters
# --------------------------------------------------

mel_spectrogram = (
    mel_filters @ power_spectrogram
)


# Avoid log(0)
EPSILON = 1e-10

log_mel = np.log(
    mel_spectrogram + EPSILON
)


# --------------------------------------------------
# DCT
# --------------------------------------------------

num_mel = log_mel.shape[0]

n = np.arange(num_mel)

dct_basis = np.zeros(
    (N_MFCC, num_mel)
)

for k in range(N_MFCC):

    dct_basis[k, :] = np.cos(
        np.pi
        * k
        * (2 * n + 1)
        / (2 * num_mel)
    )


mfcc = dct_basis @ log_mel


# --------------------------------------------------
# Plot log-Mel spectrogram
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.imshow(
    log_mel,
    origin="lower",
    aspect="auto",
)

plt.xlabel("Time Frame")
plt.ylabel("Mel Filter")
plt.title("Log-Mel Spectrogram")

plt.colorbar(
    label="Log Energy"
)

plt.tight_layout()

plt.savefig(
    "experiments/log_mel.png",
    dpi=150,
)


# --------------------------------------------------
# Plot MFCC
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.imshow(
    mfcc,
    origin="lower",
    aspect="auto",
)

plt.xlabel("Time Frame")
plt.ylabel("MFCC Coefficient")
plt.title("MFCC Representation")

plt.colorbar(
    label="Coefficient"
)

plt.tight_layout()

plt.savefig(
    "experiments/mfcc.png",
    dpi=150,
)


print("Input samples:", len(signal))
print("Mel features:", mel_spectrogram.shape)
print("Log-Mel shape:", log_mel.shape)
print("MFCC shape:", mfcc.shape)
print("Number of MFCC coefficients:", N_MFCC)

print("Saved:")
print("  experiments/log_mel.png")
print("  experiments/mfcc.png")
