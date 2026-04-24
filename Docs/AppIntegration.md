# App Integration

This package is intentionally small: it loads a bundled CoreML detector, turns a
`CGImage` into the expected tensor, runs prediction, and returns raw CoreML
outputs.

## Lifecycle

Create `OCRDetector` once and keep it alive:

```swift
final class PageDetector {
    private let detector: OCRDetector

    init() throws {
        detector = try OCRDetector(computeUnits: .cpuAndGPU)
    }

    func detect(cgImage: CGImage) throws -> OCRDetectorPrediction {
        try detector.prediction(for: cgImage)
    }
}
```

`OCRDetector.init` compiles and loads the `.mlpackage`. That is expensive enough
that it should be treated like app startup or pipeline setup work.

## Choosing Compute Units

Use `.cpuAndGPU` for lowest observed latency on the packaged detector. Use
`.cpuAndNeuralEngine` when the app needs to reserve GPU time for rendering or
other workloads and can tolerate higher detector latency.

```swift
let latencyDetector = try OCRDetector(computeUnits: .cpuAndGPU)
let aneDetector = try OCRDetector(computeUnits: .cpuAndNeuralEngine)
```

## Input Contract

`OCRDetector.makeInputArray(from:)` resizes the image to `768 x 768`, converts to
RGB planar order, normalizes pixels into `[0, 1]`, and returns
`Float32[1, 3, 768, 768]`.

If your app uses a different image pipeline, make sure the tensor shape and
normalization match exactly before calling:

```swift
let prediction = try detector.prediction(input: imageArray)
```

The package validates the input shape before calling CoreML.

## Output Contract

`OCRDetectorPrediction.output` contains:

| property | CoreML output | shape |
|---|---|---|
| `prob` | text probability map | `[1, 192, 192]` |
| `rboxes` | rotated box geometry | `[1, 192, 192, 5]` |
| `features` | detector feature map | `[1, 128, 192, 192]` |

The package does not implement FOTS/Nemotron post-processing. App code still
needs thresholding, rotated-box decode, non-maximum suppression, crop/rectify,
recognition, and any layout ordering needed by the product.

## Model Source

The same `Detector224.mlpackage` is available on Hugging Face:

<https://huggingface.co/mweinbach1/ocr-coreml-detector-224>

The SwiftPM repository is:

<https://github.com/mweinbach/OCRCoreMLDetector>
