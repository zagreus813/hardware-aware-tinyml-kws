import numpy as np


SAMPLE_RATE = 16000

FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 512

N_MELS = 40
N_MFCC = 13

F_MIN = 20
F_MAX = 4000


def hz_to_mel(freq):
    return 2595.0 * np.log10(
        1.0 + freq / 700.0
    )


def mel_to_hz(mel):
    return 700.0 * (
        10.0 ** (mel / 2595.0) - 1.0
    )


def build_mel_filterbank(
    n_fft=N_FFT,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
    sample_rate=SAMPLE_RATE,
):
    """
    Build triangular Mel filters.

    Returns:
        filters: shape (n_mels, n_fft // 2 + 1)
    """

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = np.linspace(
        mel_min,
        mel_max,
        n_mels + 2,
    )

    hz_points = mel_to_hz(
        mel_points
    )

    bin_points = np.floor(
        (n_fft + 1)
        * hz_points
        / sample_rate
    ).astype(int)

    n_bins = n_fft // 2 + 1

    filters = np.zeros(
        (n_mels, n_bins),
        dtype=np.float32,
    )

    for m in range(1, n_mels + 1):

        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]

        if center > left:

            for k in range(left, center):
                filters[m - 1, k] = (
                    (k - left)
                    / (center - left)
                )

        if right > center:

            for k in range(center, right):
                filters[m - 1, k] = (
                    (right - k)
                    / (right - center)
                )

    return filters


def compute_dct_basis(
    n_mfcc=N_MFCC,
    n_mels=N_MELS,
):
    """
    Build the DCT-II basis used for MFCC extraction.
    """

    n = np.arange(n_mels)

    basis = np.zeros(
        (n_mfcc, n_mels),
        dtype=np.float32,
    )

    for k in range(n_mfcc):

        basis[k, :] = np.cos(
            np.pi
            * k
            * (2 * n + 1)
            / (2 * n_mels)
        )

    return basis


def extract_mfcc(audio):
    """
    Convert a fixed-length waveform into MFCC features.

    Output:
        shape = (13, 98)
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) != 16000:
        raise ValueError(
            "Expected exactly 16000 samples."
        )

    # Number of frames is determined by
    # frame length and hop length.
    n_frames = (
        1
        + (len(audio) - FRAME_LENGTH)
        // HOP_LENGTH
    )

    # Hann window
    window = np.hanning(
        FRAME_LENGTH
    ).astype(np.float32)

    mel_filterbank = (
        build_mel_filterbank()
    )

    spectra = []

    for frame_idx in range(n_frames):

        start = (
            frame_idx * HOP_LENGTH
        )

        end = (
            start + FRAME_LENGTH
        )

        frame = audio[start:end]

        # Windowing
        frame = frame * window

        # Zero-padding: 400 -> 512
        padded = np.zeros(
            N_FFT,
            dtype=np.float32,
        )

        padded[:FRAME_LENGTH] = frame

        # FFT
        spectrum = np.fft.rfft(
            padded
        )

        # Power spectrum
        power = (
            np.abs(spectrum) ** 2
        )

        spectra.append(power)

    power_spectrogram = np.asarray(
        spectra,
        dtype=np.float32,
    ).T

    # Mel filter bank
    mel_energy = (
        mel_filterbank
        @ power_spectrogram
    )

    # Numerical stability
    log_mel = np.log(
        mel_energy + 1e-10
    )

    # DCT
    dct_basis = compute_dct_basis()

    mfcc = (
        dct_basis
        @ log_mel
    )

    return mfcc.astype(
        np.float32
    )

def extract_log_mel(audio):
    """
    Convert a fixed-length waveform into a Log-Mel
    spectrogram.

    Output:
        shape = (40, 98)
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) != 16000:
        raise ValueError(
            "Expected exactly 16000 samples."
        )

    n_frames = (
        1
        + (len(audio) - FRAME_LENGTH)
        // HOP_LENGTH
    )

    window = np.hanning(
        FRAME_LENGTH
    ).astype(np.float32)

    mel_filterbank = (
        build_mel_filterbank()
    )

    spectra = []

    for frame_idx in range(n_frames):

        start = (
            frame_idx * HOP_LENGTH
        )

        end = (
            start + FRAME_LENGTH
        )

        frame = audio[start:end]

        # Hann window
        frame = frame * window

        # Zero-padding: 400 -> 512
        padded = np.zeros(
            N_FFT,
            dtype=np.float32,
        )

        padded[:FRAME_LENGTH] = frame

        # FFT
        spectrum = np.fft.rfft(
            padded
        )

        # Power spectrum
        power = (
            np.abs(spectrum) ** 2
        )

        spectra.append(power)

    power_spectrogram = np.asarray(
        spectra,
        dtype=np.float32,
    ).T

    # Mel energies
    mel_energy = (
        mel_filterbank
        @ power_spectrogram
    )

    # Log compression
    log_mel = np.log(
        mel_energy + 1e-10
    )

    return log_mel.astype(
        np.float32
    )
