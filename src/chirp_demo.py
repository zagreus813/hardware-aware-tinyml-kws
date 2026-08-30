import numpy as np
import matplotlib.pyplot as plt


FS = 16000
DURATION = 1.0

t = np.arange(0, DURATION, 1 / FS)

# Frequency increases from 200 Hz to 2000 Hz.
F_START = 200
F_END = 2000

# Linear chirp:
# f(t) = F_START + k*t
k = (F_END - F_START) / DURATION

phase = 2 * np.pi * (
    F_START * t
    + 0.5 * k * t**2
)

signal = np.sin(phase)


# STFT parameters
FRAME_LENGTH = 400
HOP_LENGTH = 160

window = np.hanning(FRAME_LENGTH)

frames = []

for start in range(
    0,
    len(signal) - FRAME_LENGTH + 1,
    HOP_LENGTH
):
    frame = signal[
        start:start + FRAME_LENGTH
    ]

    frame = frame * window

    spectrum = np.fft.rfft(frame)

    magnitude = np.abs(spectrum)

    frames.append(magnitude)


spectrogram = np.array(frames).T

frequencies = np.fft.rfftfreq(
    FRAME_LENGTH,
    d=1 / FS
)

times = (
    np.arange(spectrogram.shape[1])
    * HOP_LENGTH
    / FS
)


plt.figure(figsize=(10, 5))

plt.imshow(
    spectrogram,
    origin="lower",
    aspect="auto",
    extent=[
        times[0],
        times[-1],
        frequencies[0],
        frequencies[-1]
    ]
)

plt.ylim(0, 2500)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.title("Spectrogram of a Linear Chirp")

plt.colorbar(label="Magnitude")

plt.tight_layout()

plt.savefig(
    "experiments/chirp_spectrogram.png",
    dpi=150
)

print("Sampling rate:", FS)
print("Duration:", DURATION)
print("Starting frequency:", F_START)
print("Ending frequency:", F_END)
print("Frames:", spectrogram.shape[1])
print("Frequency bins:", spectrogram.shape[0])
print("Saved: experiments/chirp_spectrogram.png")
