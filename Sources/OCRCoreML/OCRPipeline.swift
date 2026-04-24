import CoreGraphics
import CoreML
import Foundation

public final class OCRPipeline {
    public let detector: OCRDetector
    public let recognizer: OCRRecognizer
    public let relational: OCRRelationalModel

    public init(modelURLs: OCRModelURLs? = nil, computeUnits: MLComputeUnits = .all) throws {
        let urls = try modelURLs ?? OCRModelURLs.bundled
        detector = try OCRDetector(modelURL: urls.detector, computeUnits: computeUnits)
        recognizer = try OCRRecognizer(modelURL: urls.recognizer, computeUnits: computeUnits)
        relational = try OCRRelationalModel(modelURL: urls.relational, computeUnits: computeUnits)
    }

    public func detect(image: CGImage) throws -> OCRDetectorPrediction {
        try detector.prediction(for: image)
    }

    public func recognize(regions: MLMultiArray) throws -> OCRRecognizerPrediction {
        try recognizer.prediction(input: regions)
    }

    public func relate(
        rectifiedQuads: MLMultiArray,
        originalQuads: MLMultiArray,
        recognizerFeatures: MLMultiArray,
        numValid: Int
    ) throws -> OCRRelationalPrediction {
        try relational.prediction(
            rectifiedQuads: rectifiedQuads,
            originalQuads: originalQuads,
            recognizerFeatures: recognizerFeatures,
            numValid: numValid
        )
    }
}
