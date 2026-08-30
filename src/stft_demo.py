import numpy as np
import matplotlib.pyplot as plt


FS = 16000
DURATION = 1.0

t = np.arange(0, DURATION, 1 / FS)

# A simple time-varying signal
signal = (
    np.sin(2 * np.pi * 440 * t)
    + 0.5 * np.sin(2 * np.pi * 1000 * t)
)

# STFT parameters
FRAME_LENGTH = 400   # 25 ms at 16 kHz
HOP_LENGTH = 160     # 10 ms at 16 kHz

window = np.hanning(FRAME_LENGTH)

frames = []

for start in range(0, len(signal) - FRAME_LENGTH + 1, HOP_LENGTH):
    frame = signal[start:start + FRAME_LENGTH]

    # Apply window
    frame = frame * window

    # FFT
    spectrum = np.fft.rfft(frame)

    # Magnitude
    magnitude = np.abs(spectrum)

    frames.append(magnitude)

spectrogram = np.array(frames).T

frequencies = np.fft.rfftfreq(
    FRAME_LENGTH,
    d=1 / FS
)

times = (
    np.arange(spectrogram.shape[1]) * HOP_LENGTH / FS
)

# Plot
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

plt.ylim(0, 2000)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("STFT Spectrogram")

plt.colorbar(label="Magnitude")

plt.tight_layout()

plt.savefig(
    "experiments/spectrogram.png",
    dpi=150
)

print("Input samples:", len(signal))
print("Frame length:", FRAME_LENGTH)
print("Hop length:", HOP_LENGTH)
print("Number of frames:", spectrogram.shape[1])
print("Frequency bins:", spectrogram.shape[0])
print("Saved: experiments/spectrogram.png")
