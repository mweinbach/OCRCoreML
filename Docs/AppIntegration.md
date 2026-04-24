# App Integration

`OCRCoreML` packages the neural stages of the Nemotron OCR v2 English pipeline
for Apple platforms. The app owns the geometric and graph post-processing that
sits between those neural stages.

## Model Lifecycle

Create `OCRPipeline` once and keep it alive:

```swift
import OCRCoreML

final class OCRService {
    private let pipeline: OCRPipeline

    init() throws {
        pipeline = try OCRPipeline(computeUnits: .cpuAndGPU)
    }
}
```

Each model initializer compiles and loads an `.mlpackage`. Do this at app
startup or pipeline setup time, not inside a per-image loop.

## Stage Contracts

Detector:

```swift
let detectorInput = try OCRDetector.makeInputArray(from: cgImage)
let detectorPrediction = try pipeline.detector.prediction(input: detectorInput)
```

Input is `Float32[1, 3, 768, 768]`, RGB planar in `[0, 1]`.

Detector outputs:

| output | shape |
|---|---:|
| `prob` | `[1, 192, 192]` |
| `rboxes` | `[1, 192, 192, 5]` |
| `features` | `[1, 128, 192, 192]` |

Recognizer:

```swift
let recognizerPrediction = try pipeline.recognizer.prediction(input: regions)
let decoded = try pipeline.recognizer.decode(
    logits: recognizerPrediction.output.logits,
    count: detectedRegionCount
)
```

`regions` must be feature crops shaped `Float32[128, 128, 8, 32]`. Pad unused
rows with zeros and pass the real count to downstream code.

Recognizer outputs:

| output | shape |
|---|---:|
| `logits` | `[128, 32, 858]` |
| `features` | `[128, 32, 256]` |

Relational:

```swift
let relationalPrediction = try pipeline.relational.prediction(
    rectifiedQuads: relationalRegionFeatures,
    originalQuads: originalQuads,
    recognizerFeatures: recognizerPrediction.output.features,
    numValid: detectedRegionCount
)
```

Relational inputs:

| input | shape |
|---|---:|
| `rectified_quads` | `[128, 128, 2, 3]` |
| `original_quads` | `[128, 4, 2]` |
| `recog_features` | `[128, 32, 256]` |
| `num_valid` | `[1] Int32` |

Relational outputs:

| output | shape |
|---|---:|
| `words` | `[128, 129]` |
| `lines` | `[128, 129]` |
| `line_log_var` | `[128, 129]` |

## Post-Processing Work Still Needed

The upstream package relies on CUDA/C++ helpers for the non-neural pipeline.
When integrating into an app, port or replace these pieces:

| upstream helper | role |
|---|---|
| `rrect_to_quads` | convert dense detector rotated boxes to quadrangles |
| `quad_non_maximal_suppression` | filter overlapping detector candidates |
| `quad_rectify_forward` | build normalized sampling grids for each quad |
| `IndirectGridSample` | sample detector feature maps into recognizer/relational crops |
| `decode_sequences` | convert token IDs to text; a greedy Swift decoder is included |
| `dense_relations_to_graph` | convert relation matrices into ordered text groups |

Use the CLI smoke test to verify model packaging and model runtime before adding
post-processing:

```bash
swift run ocr-coreml-smoke --compute-units gpu
```

## Model Bundle

The same artifacts are published on Hugging Face:

<https://huggingface.co/mweinbach1/nemotron-ocr-v2-coreml>
