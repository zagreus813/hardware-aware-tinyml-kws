from pathlib import Path
import wave

import numpy as np
import librosa


DATASET_ROOT = Path(
    "~/projects/tinyml-esp32/data/speech_commands_v0.02"
).expanduser()

FILE = (
    DATASET_ROOT
    / "yes"
    / "004ae714_nohash_0.wav"
)

TARGET_LENGTH = 16000
SAMPLE_RATE = 16000

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160

def load_audio(path):
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    audio = np.frombuffer(
        raw,
        dtype=np.int16
    ).astype(np.float32)

    audio /= 32768.0

    return audio, sample_rate


def pad_or_trim(audio, target_length):
    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
            mode="constant"
        )
    else:
        audio = audio[:target_length]

    return audio

def hz_to_mel(freq):
    """
    Convert frequency from Hertz to Mel scale.
    """
    return 2595.0 * np.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):
    """
    Convert frequency from Mel scale back to Hertz.
    """
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def build_mel_filterbank(
    n_fft,
    n_mels,
    f_min,
    f_max,
    sample_rate,
):
    """
    Construct a triangular Mel filter bank.

    Returns:
        filters:
            Shape = (n_mels, n_fft // 2 + 1)

        hz_points:
            Frequency boundaries of the filters.
    """

    # Mel-domain boundaries
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    # Equally spaced points in Mel scale
    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2
    )

    # Convert Mel points back to Hz
    hz_points = mel_to_hz(
        mel_points
    )

    # Convert Hz frequencies to FFT-bin indices
    bin_points = np.floor(
        (n_fft + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    # Number of positive-frequency FFT bins
    n_bins = n_fft // 2 + 1

    filters = np.zeros(
        (n_mels, n_bins),
        dtype=np.float32
    )

    # Build triangular filters
    for m in range(1, n_mels + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        # Rising edge
        if center > left:

            for k in range(left, center):

                filters[m - 1, k] = (
                    (k - left)
                    / (center - left)
                )

        # Falling edge
        if right > center:

            for k in range(center, right):

                filters[m - 1, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters, hz_points
def extract_mfcc(audio):
    frame_length = 400
    hop_length = 160
    n_fft = 512
    n_mels = 40
    n_mfcc = 13

    # ---------------------------------------------
    # 1. Build 40-sample-overlap frames
    # ---------------------------------------------

    n_frames = (
        1
        + (len(audio) - frame_length)
        // hop_length
    )

    window = np.hanning(frame_length)

    mel_filterbank, _ = build_mel_filterbank(
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=20,
        f_max=4000,
        sample_rate=SAMPLE_RATE,
    )

    spectra = []

    for i in range(n_frames):

        start = i * hop_length
        end = start + frame_length

        frame = audio[start:end]

        # Hann window
        frame = frame * window

        # -----------------------------------------
        # Zero-pad 400 → 512
        # -----------------------------------------

        padded = np.zeros(
            n_fft,
            dtype=np.float32
        )

        padded[:frame_length] = frame

        # -----------------------------------------
        # 512-point FFT
        # -----------------------------------------

        spectrum = np.fft.rfft(padded)

        # Power spectrum
        power = (
            np.abs(spectrum) ** 2
        )

        spectra.append(power)

    power_spectrogram = np.array(
        spectra,
        dtype=np.float32
    ).T

    # ---------------------------------------------
    # 2. Mel filter bank
    # ---------------------------------------------

    mel_energy = (
        mel_filterbank
        @ power_spectrogram
    )

    # ---------------------------------------------
    # 3. Log compression
    # ---------------------------------------------

    log_mel = np.log(
        mel_energy + 1e-10
    )

    # ---------------------------------------------
    # 4. DCT → MFCC
    # ---------------------------------------------

    n = np.arange(n_mels)

    dct_basis = np.zeros(
        (n_mfcc, n_mels),
        dtype=np.float32
    )

    for k in range(n_mfcc):

        dct_basis[k, :] = np.cos(
            np.pi
            * k
            * (2 * n + 1)
            / (2 * n_mels)
        )

    mfcc = (
        dct_basis
        @ log_mel
    )

    return mfcc.astype(
        np.float32
    )
audio, sample_rate = load_audio(FILE)

print("Original:")
print("  Sample rate:", sample_rate)
print("  Samples:", len(audio))
print("  Duration:", len(audio) / sample_rate)


audio = pad_or_trim(
    audio,
    TARGET_LENGTH
)

print("\nAfter pad/trim:")
print("  Shape:", audio.shape)


mfcc = extract_mfcc(audio)

print("\nMFCC:")
print("  Shape:", mfcc.shape)
print("  dtype:", mfcc.dtype)
print("  min:", mfcc.min())
print("  max:", mfcc.max())
print("  mean:", mfcc.mean())
print("  std:", mfcc.std())


np.save(
    "experiments/sample_mfcc.npy",
    mfcc
)

print(
    "\nSaved: experiments/sample_mfcc.npy"
)
