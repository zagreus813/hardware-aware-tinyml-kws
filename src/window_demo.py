import numpy as np
import matplotlib.pyplot as plt


FS = 16000
FRAME_LENGTH = 400

t = np.arange(FRAME_LENGTH) / FS

# Choose a frequency that does NOT align perfectly
# with the FFT frequency bins.
FREQ = 440

signal = np.sin(2 * np.pi * FREQ * t)

# FFT without window
fft_raw = np.fft.rfft(signal)
mag_raw = np.abs(fft_raw)

# Hanning window
window = np.hanning(FRAME_LENGTH)

windowed_signal = signal * window

fft_windowed = np.fft.rfft(windowed_signal)
mag_windowed = np.abs(fft_windowed)

frequencies = np.fft.rfftfreq(
    FRAME_LENGTH,
    d=1 / FS
)

plt.figure(figsize=(10, 5))

plt.plot(
    frequencies,
    mag_raw,
    label="Without Window"
)

plt.plot(
    frequencies,
    mag_windowed,
    label="Hanning Window"
)

plt.xlim(300, 600)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title("Effect of Windowing on Spectral Leakage")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "experiments/window_comparison.png",
    dpi=150
)

print("Sampling rate:", FS)
print("Frame length:", FRAME_LENGTH)
print("Frequency resolution:", FS / FRAME_LENGTH)
print("Signal frequency:", FREQ)
print("Saved: experiments/window_comparison.png")
