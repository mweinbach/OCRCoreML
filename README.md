# OCRCoreMLDetector

SwiftPM package for running the CoreML detector artifact from
[mweinbach1/ocr-coreml-detector-224](https://huggingface.co/mweinbach1/ocr-coreml-detector-224).

This package includes:

- `OCRCoreMLDetector`: a library target with a bundled `Detector224.mlpackage`
- `ocr-detector-infer`: a command-line smoke test and benchmarking helper

The GitHub package is published at
[mweinbach/OCRCoreMLDetector](https://github.com/mweinbach/OCRCoreMLDetector).

## Model

`Detector224.mlpackage` is a batch-1 CoreML conversion of the detector stage
from [NVIDIA Nemotron OCR v2](https://huggingface.co/nvidia/nemotron-ocr-v2).
It is the selected 768 px, int8 per-channel quantized detector from
`experiments/224_detector_ane_decomposed_int8_768`.

Input:

- name: `image`
- shape: `Float32[1, 3, 768, 768]`
- layout: RGB planar, normalized to `[0, 1]`

Outputs:

- `prob`: text probability map, `Float32[1, 192, 192]`
- `rboxes`: rotated box geometry, `Float32[1, 192, 192, 5]`
- `features`: detector feature map, `Float32[1, 128, 192, 192]`

This is the detector only. Full OCR still needs post-processing, crop/rectify,
recognition, and layout/relational stages.

## Add To An App

Add the package to `Package.swift`:

```swift
.package(url: "https://github.com/mweinbach/OCRCoreMLDetector.git", from: "0.1.0")
```

Then add `OCRCoreMLDetector` to your app target dependencies.

For Xcode projects, use **File > Add Package Dependencies...** and enter:

```text
https://github.com/mweinbach/OCRCoreMLDetector.git
```

Supported platforms:

- iOS 18+
- macOS 15+

More implementation notes are in [Docs/AppIntegration.md](Docs/AppIntegration.md).

## Basic Usage

Create one detector and reuse it for repeated calls. Model initialization
compiles and loads the bundled `.mlpackage`, so it should not happen inside a
hot per-frame loop.

```swift
import CoreML
import OCRCoreMLDetector

let detector = try OCRDetector(computeUnits: .cpuAndGPU)
let prediction = try detector.prediction(for: cgImage)

let prob = prediction.output.prob
let rboxes = prediction.output.rboxes
let features = prediction.output.features
print(prediction.predictionTimeMs)
```

Load from a file URL:

```swift
let imageURL = URL(fileURLWithPath: "/path/to/page.png")
let prediction = try detector.prediction(forImageAt: imageURL)
```

Use a custom compiled or packaged model:

```swift
let modelURL = Bundle.main.url(forResource: "Detector224", withExtension: "mlpackage")!
let detector = try OCRDetector(modelURL: modelURL, computeUnits: .all)
```

If your app already has image preprocessing, pass the exact CoreML tensor:

```swift
let imageArray = try OCRDetector.makeInputArray(from: cgImage)
let prediction = try detector.prediction(input: imageArray)
```

## Compute Units

On the test Apple Silicon machine, GPU/ALL is the best default for latency.
CPU+ANE works but has higher single-shot setup overhead for this detector.

| compute units | median prediction latency |
|---|---:|
| `.all` | 13.53 ms |
| `.cpuAndGPU` | 13.65 ms |
| `.cpuAndNeuralEngine` | 54.51 ms |
| `.cpuOnly` | 298.28 ms |

These timings are for the bundled sample after warmup using the batch-1
`Detector224.mlpackage`.

## CLI Smoke Test

Run the bundled sample:

```bash
swift run ocr-detector-infer --compute-units gpu
```

Run another image or repeat predictions:

```bash
swift run ocr-detector-infer /path/to/image.png --compute-units gpu --repeat 5
swift run ocr-detector-infer /path/to/image.png --compute-units ane --warmup 1 --repeat 3
```

Valid `--compute-units` values are `all`, `gpu`, `ane`, and `cpu`.

## License

The converted model weights inherit the
[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
See `LICENSE` and `NOTICE` in this repository.
