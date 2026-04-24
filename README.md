# OCRCoreML

SwiftPM package for the CoreML neural stages of
[NVIDIA Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2).

Model bundle:
[mweinbach1/nemotron-ocr-v2-coreml](https://huggingface.co/mweinbach1/nemotron-ocr-v2-coreml)

Swift package:
[github.com/mweinbach/OCRCoreML](https://github.com/mweinbach/OCRCoreML)

## What Is Included

This package bundles the three converted CoreML sub-models needed by the OCR
pipeline:

| stage | resource | input | outputs |
|---|---|---|---|
| detector | `DetectorGPUInt8_768.mlpackage` | `image: Float32[1, 3, 768, 768]` | `prob`, `rboxes`, `features` |
| recognizer | `RecognizerFeaturesInt8.mlpackage` | `regions: Float32[128, 128, 8, 32]` | `logits`, `features` |
| relational | `RelationalInt8.mlpackage` | rectified regions, quads, recognizer features, valid count | `words`, `lines`, `line_log_var` |

The recognizer model emits transformer `features`, which are required by the
relational stage. That is the key difference from the earlier detector-only
package.

The package also includes `charset.txt` and `model_config.json` from the
English Nemotron OCR v2 checkpoint.

## Important Boundary

The bundled CoreML models cover the neural pipeline. A production app still
needs to implement the non-neural geometry and graph post-processing around
them:

- detector probability thresholding and rotated-box NMS
- `rboxes` to quads
- feature-map quad rectification and bilinear grid sampling
- recognizer sequence filtering/decoding
- relational graph decoding and reading-order formatting

Those steps are CUDA/C++ helpers in the upstream Python package. The SwiftPM
package exposes the raw tensors and a simple recognizer decoder, but it does
not pretend to be a complete image-to-text OCR engine yet.

## Add To An App

Add the package to `Package.swift`:

```swift
.package(url: "https://github.com/mweinbach/OCRCoreML.git", from: "0.2.1")
```

Then add `OCRCoreML` to your app target dependencies.

Supported platforms:

- iOS 18+
- macOS 15+

## Basic Usage

Create the pipeline once and reuse it:

```swift
import CoreML
import OCRCoreML

let pipeline = try OCRPipeline(computeUnits: .cpuAndGPU)

let detectorPrediction = try pipeline.detect(image: cgImage)
let detectorFeatures = detectorPrediction.output.features

// `regions` must be rectified detector feature crops shaped [128, 128, 8, 32].
let recognizerPrediction = try pipeline.recognize(regions: regions)
let decoded = try pipeline.recognizer.decode(
    logits: recognizerPrediction.output.logits,
    count: detectedRegionCount
)

let relationalPrediction = try pipeline.relate(
    rectifiedQuads: relationalRegionFeatures,
    originalQuads: originalQuads,
    recognizerFeatures: recognizerPrediction.output.features,
    numValid: detectedRegionCount
)
```

You can also load stages independently:

```swift
let detector = try OCRDetector(computeUnits: .cpuAndGPU)
let recognizer = try OCRRecognizer(computeUnits: .cpuAndGPU)
let relational = try OCRRelationalModel(computeUnits: .cpuAndGPU)
```

## Performance

Local median latencies after warmup:

| stage | GPU/ALL median | CPU+NE median | CPU median |
|---|---:|---:|---:|
| detector | 10.65 ms | 50.46 ms | 157.71 ms |
| recognizer + features | 4.53 ms | 11.04 ms | 47.58 ms |
| relational | 1.72 ms | 6.38 ms | 34.53 ms |

For this model size and single-image workload, `.cpuAndGPU` is the practical
latency default. `.cpuAndNeuralEngine` is useful when preserving GPU time matters
more than raw latency.

## CLI Smoke Test

Run the bundled model smoke test:

```bash
swift run ocr-coreml-smoke --compute-units gpu
```

The command loads all three CoreML models, runs the detector on the bundled
sample image, then runs recognizer and relational predictions on shape-correct
smoke tensors to prove that the complete neural bundle is loadable and callable.

## License

The converted model weights inherit the
[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
See `LICENSE` and `NOTICE`.
