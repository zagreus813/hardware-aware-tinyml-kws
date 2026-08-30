import numpy as np
import matplotlib.pyplot as plt


FS = 16000
FRAME_LENGTH = 400

N_MELS = 40
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
            if center != left:
                filters[m - 1, k] = (
                    (k - left)
                    / (center - left)
                )

        for k in range(center, right):
            if right != center:
                filters[m - 1, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters, hz_points


# --------------------------------------------------
# Build Mel filter bank
# --------------------------------------------------

filters, hz_points = build_mel_filterbank(
    FRAME_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    FS,
)


# --------------------------------------------------
# Plot filter bank
# --------------------------------------------------

frequencies = np.fft.rfftfreq(
    FRAME_LENGTH,
    1 / FS
)


plt.figure(figsize=(10, 5))

for i in range(N_MELS):
    plt.plot(
        frequencies,
        filters[i]
    )

plt.xlim(F_MIN, F_MAX)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter response")

plt.title("Mel Filter Bank")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "experiments/mel_filterbank.png",
    dpi=150
)

print("Number of Mel filters:", N_MELS)
print("FFT bins:", FRAME_LENGTH // 2 + 1)
print("Frequency range:", F_MIN, "Hz ->", F_MAX, "Hz")
print("Saved: experiments/mel_filterbank.png")
