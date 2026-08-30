# Hardware-Aware TinyML Keyword Spotting

A research-oriented TinyML project for six-class speech command recognition, combining digital signal processing, neural network design, model compression, and integer-only deployment preparation.

The project investigates the trade-off between classification accuracy, model size, compute, and memory for resource-constrained embedded systems. The final software-side candidate is a depthwise-separable CNN (DS-CNN) converted to fully integer INT8 TensorFlow Lite.

## Project scope

The project is intentionally split into two stages:

1. **Software / TinyML research:** dataset analysis, deterministic audio preprocessing, MFCC extraction, neural network baselines, architecture comparison, complexity analysis, and INT8 quantization.
2. **Hardware deployment:** a companion ESP32 repository will perform the real embedded implementation and measure actual RAM, Flash, latency, and energy consumption.

This repository covers the first stage.

## Research question

> How far can a keyword-spotting model be compressed in parameters and computation while retaining useful classification accuracy for resource-constrained microcontrollers?

The work therefore treats model accuracy as only one objective. Parameter count, MACs, estimated activation memory, and quantized model size are evaluated together.

## Dataset

The experiments use Speech Commands v0.02 with six target classes:

- yes
- no
- up
- down
- left
- right

Dataset audit results:

| Property | Value |
|---|---:|
| Training samples | 18,496 |
| Validation samples | 2,245 |
| Test samples | 2,463 |
| Total selected samples | 23,204 |
| Sample rate | 16 kHz |
| Channels | Mono |
| Corrupted files | 0 |

The official validation and test lists are preserved. A speaker-overlap audit found zero speaker overlap between training, validation, and test partitions.

## Audio preprocessing

The preprocessing pipeline was implemented explicitly rather than relying on a black-box MFCC function.

```text
16-bit PCM WAV
      |
      v
Float32 normalization
      |
      v
Pad / trim to 16,000 samples
      |
      v
25 ms Hann window
      |
      v
10 ms hop
      |
      v
512-point FFT
      |
      v
Power spectrum
      |
      v
40-band Mel filterbank
      |
      v
Log compression
      |
      v
DCT
      |
      v
13 x 98 MFCC
```

The resulting CNN input tensor is:

```text
(13, 98, 1)
```

Normalization statistics are calculated from the training split only and then applied unchanged to validation and test data.

## DSP concepts investigated

The preprocessing stage covers:

- Sampling and Nyquist limits
- FFT and frequency resolution
- Spectral leakage
- Windowing
- STFT and spectrograms
- Mel frequency scale
- Mel filter banks
- Log-Mel representations
- DCT and MFCCs

These concepts are implemented explicitly so that the training representation and eventual embedded implementation can follow the same specification.

## Baseline experiments

### Dense baseline

The first baseline flattens the MFCC representation:

```text
13 x 98
  |
Flatten
  |
1274
  |
Dense(128)
  |
Dense(6)
```

Result:

- Parameters: 163,974
- MACs: 0.164 M
- Test accuracy: 81.97%

### Conventional CNN

A compact convolutional network was then evaluated.

Result:

- Parameters: 15,446
- MACs: 2.203 M
- Test accuracy: 61.47%

The experiment demonstrates that reducing parameter count alone does not guarantee computational efficiency or accuracy.

### DS-CNN

Depthwise-separable convolutions were introduced to reduce the compute cost of convolutional layers.

Result:

- Parameters: 4,966
- MACs: 0.509 M
- FP32 test accuracy: 75.80%

Compared with the Dense baseline, the DS-CNN reduces parameter count by approximately 96.97%.

Compared with the conventional CNN, DS-CNN reduces MACs by approximately 76.9% while improving test accuracy from 61.47% to 75.80%.

## Representation ablation

A controlled experiment compared MFCC and normalized 40-band Log-Mel features using the same CNN architecture.

The evaluated configuration produced:

| Representation | Test accuracy |
|---|---:|
| MFCC + CNN | 61.47% |
| Log-Mel + CNN | 25.50% |

This result is treated as a configuration-specific ablation, not as a claim that Log-Mel is inherently inferior to MFCC.

## Complexity and memory

One important finding is that parameter count and runtime memory are different optimization objectives.

For DS-CNN:

- FP32 parameter storage estimate: approximately 19.4 KB
- Peak activation estimate: approximately 79.6 KB
- MACs: 0.509 M

The largest activation is produced by the first convolution:

```text
13 x 98 x 16
```

which corresponds to approximately 79.6 KB in FP32.

This illustrates why TinyML optimization must consider both model weights and intermediate activations.

The activation figures in this repository are architecture-level estimates, not measurements of an actual ESP32 tensor arena. Real hardware measurements belong to the companion deployment repository.

## INT8 post-training quantization

The DS-CNN was converted to a fully integer TensorFlow Lite model.

```text
DS-CNN FP32
     |
     v
Post-training quantization
     |
     v
INT8 TFLite
```

Measured results:

| Metric | FP32 TFLite | INT8 TFLite |
|---|---:|---:|
| Model size | 23,200 bytes | 14,416 bytes |
| Test accuracy | 75.80% | 76.61% |

Observed model-size reduction:

```text
37.86%
```

The complete TFLite input/output path is INT8:

```text
Input scale       0.0546097383
Input zero point -15

Output scale      0.00390625
Output zero point -128
```

On the test set, only 46 of 2,463 predictions changed after quantization (1.87%). Among them:

- 7 changed from correct to incorrect
- 27 changed from incorrect to correct

The net effect in this experiment was a 0.812 percentage-point increase in test accuracy.

The observed accuracy increase should not be interpreted as evidence that INT8 quantization intrinsically improves accuracy; it is the result observed for this specific model, calibration set, and test partition.

## Final benchmark

| Model | Parameters | MACs | Test accuracy |
|---|---:|---:|---:|
| Dense FP32 | 163,974 | 0.164 M | 81.97% |
| CNN v2 FP32 | 15,446 | 2.203 M | 61.47% |
| DS-CNN v1 FP32 | 4,966 | 0.509 M | 75.80% |
| DS-CNN v1 INT8 | 4,966 | 0.509 M | **76.61%** |

The current software-side deployment candidate is therefore:

```text
DS-CNN v1
    |
    v
INT8 TFLite
```

It combines a small parameter footprint, reduced convolutional compute, and full-integer inference.

## Visual results

The repository includes figures generated during the DSP and model-analysis stages.

Suggested assets:

```text
docs/assets/
├── fft_demo.png
├── spectrogram.png
├── chirp_spectrogram.png
├── mel_filterbank.png
├── log_mel.png
└── mfcc.png
```

Benchmark figures are stored in:

```text
results/figures/
├── accuracy_vs_parameters.png
├── accuracy_vs_macs.png
├── accuracy_vs_activation_memory.png
└── tflite_model_sizes.png
```

## Repository structure

```text
hardware-aware-tinyml-kws/
├── src/
├── experiments/
├── cache/
├── theory/
├── results/
│   ├── final_benchmark.csv
│   ├── final_benchmark.json
│   └── figures/
├── docs/
│   ├── plot_results.py
│   └── assets/
└── README.md
```

Large raw datasets and generated cache files should not be committed to Git. The repository should keep the dataset acquisition instructions and reproducible preprocessing scripts instead.

## Reproducibility

The experiments use:

- Python 3.8.10
- TensorFlow 2.13.1
- NumPy 1.24.4
- SciPy 1.10.1
- scikit-learn 1.3.2
- Matplotlib 3.7.5

The software pipeline uses fixed preprocessing parameters and explicit training/test splits.

## Current limitations

This repository does not yet report real MCU measurements.

In particular, the following are hardware-dependent and intentionally reserved for the companion ESP32 repository:

- Actual tensor-arena RAM usage
- Real Flash/RAM allocation
- End-to-end inference latency on the microcontroller
- Audio acquisition latency
- Energy per inference
- Real microphone robustness
- Real-time keyword spotting behavior

The current memory values are therefore estimates used for architecture-level comparison.

## Companion hardware project

The final INT8 TFLite model produced here will be deployed in a companion ESP32 project covering:

```text
Microphone
    |
    v
Embedded audio acquisition
    |
    v
Embedded preprocessing
    |
    v
INT8 TFLite Micro
    |
    v
Keyword prediction
```

That repository will report measured RAM, Flash, latency, throughput, and energy consumption.

## Status

Software-side TinyML research pipeline:

- [x] DSP pipeline
- [x] Dataset audit
- [x] Speaker-independent split verification
- [x] MFCC implementation
- [x] Dense baseline
- [x] CNN baseline
- [x] DS-CNN
- [x] Complexity analysis
- [x] Memory analysis
- [x] FP32 TFLite conversion
- [x] INT8 post-training quantization
- [x] FP32 vs INT8 evaluation
- [x] Final benchmark
- [ ] Final documentation cleanup
- [ ] Public repository release

Hardware deployment is intentionally maintained as a separate companion project.
