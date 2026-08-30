import numpy as np
import matplotlib.pyplot as plt


# Sampling configuration
FS = 16000
DURATION = 1.0

# Time axis
t = np.arange(0, DURATION, 1 / FS)

# Two sinusoidal components
f1 = 440
f2 = 1000

signal = (
    np.sin(2 * np.pi * f1 * t)
    + 0.5 * np.sin(2 * np.pi * f2 * t)
)

# FFT
spectrum = np.fft.rfft(signal)
frequencies = np.fft.rfftfreq(len(signal), 1 / FS)

magnitude = np.abs(spectrum)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(frequencies, magnitude)
plt.xlim(0, 2000)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("FFT of Synthetic Audio Signal")

plt.grid(True)
plt.tight_layout()
plt.savefig(
    "experiments/fft_demo.png",
    dpi=150
)

print("Signal samples:", len(signal))
print("Sampling rate:", FS, "Hz")
print("Saved: experiments/fft_demo.png")
