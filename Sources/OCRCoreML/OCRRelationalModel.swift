import CoreML
import Foundation

public struct OCRRelationalOutput {
    public let words: MLMultiArray
    public let lines: MLMultiArray
    public let lineLogVar: MLMultiArray

    public init(words: MLMultiArray, lines: MLMultiArray, lineLogVar: MLMultiArray) {
        self.words = words
        self.lines = lines
        self.lineLogVar = lineLogVar
    }
}

public struct OCRRelationalPrediction {
    public let output: OCRRelationalOutput
    public let predictionTimeMs: Double

    public init(output: OCRRelationalOutput, predictionTimeMs: Double) {
        self.output = output
        self.predictionTimeMs = predictionTimeMs
    }
}

public final class OCRRelationalModel {
    public static let maxRegions = 128
    public static let rectifiedShape = [maxRegions, 128, 2, 3]
    public static let originalQuadsShape = [maxRegions, 4, 2]
    public static let recognizerFeaturesShape = [maxRegions, 32, 256]
    public static let outputShape = [maxRegions, maxRegions + 1]

    public let modelURL: URL
    public let model: MLModel

    public static var bundledModelURL: URL {
        get throws {
            try OCRModelURLs.bundled.relational
        }
    }

    public init(modelURL: URL? = nil, computeUnits: MLComputeUnits = .all) throws {
        let resolvedModelURL = try modelURL ?? Self.bundledModelURL
        self.modelURL = resolvedModelURL
        self.model = try loadModel(at: resolvedModelURL, computeUnits: computeUnits)
    }

    public func prediction(
        rectifiedQuads: MLMultiArray,
        originalQuads: MLMultiArray,
        recognizerFeatures: MLMultiArray,
        numValid: Int
    ) throws -> OCRRelationalPrediction {
        try validateShape(rectifiedQuads, name: "rectified_quads", expected: Self.rectifiedShape)
        try validateShape(originalQuads, name: "original_quads", expected: Self.originalQuadsShape)
        try validateShape(recognizerFeatures, name: "recog_features", expected: Self.recognizerFeaturesShape)

        let numValidArray = try MLMultiArray(shape: [1], dataType: .int32)
        let numValidPointer = numValidArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
        numValidPointer[0] = Int32(min(max(numValid, 0), Self.maxRegions))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "rectified_quads": MLFeatureValue(multiArray: rectifiedQuads),
            "original_quads": MLFeatureValue(multiArray: originalQuads),
            "recog_features": MLFeatureValue(multiArray: recognizerFeatures),
            "num_valid": MLFeatureValue(multiArray: numValidArray),
        ])

        let predictionStart = DispatchTime.now()
        let output = try model.prediction(from: input)
        let predictionTimeMs = millisecondsSince(predictionStart)

        guard let words = output.featureValue(for: "words")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("words")
        }
        guard let lines = output.featureValue(for: "lines")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("lines")
        }
        guard let lineLogVar = output.featureValue(for: "line_log_var")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("line_log_var")
        }

        return OCRRelationalPrediction(
            output: OCRRelationalOutput(words: words, lines: lines, lineLogVar: lineLogVar),
            predictionTimeMs: predictionTimeMs
        )
    }

    public static func makeZeroRectifiedQuads() throws -> MLMultiArray {
        try zeroFloatArray(shape: rectifiedShape.map { NSNumber(value: $0) })
    }

    public static func makeZeroOriginalQuads() throws -> MLMultiArray {
        try zeroFloatArray(shape: originalQuadsShape.map { NSNumber(value: $0) })
    }
}
