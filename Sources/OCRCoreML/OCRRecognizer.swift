import CoreML
import Foundation

public struct OCRRecognizerOutput {
    public let logits: MLMultiArray
    public let features: MLMultiArray

    public init(logits: MLMultiArray, features: MLMultiArray) {
        self.logits = logits
        self.features = features
    }
}

public struct OCRRecognizerPrediction {
    public let output: OCRRecognizerOutput
    public let predictionTimeMs: Double

    public init(output: OCRRecognizerOutput, predictionTimeMs: Double) {
        self.output = output
        self.predictionTimeMs = predictionTimeMs
    }
}

public struct OCRRecognizedSequence {
    public let text: String
    public let confidence: Double

    public init(text: String, confidence: Double) {
        self.text = text
        self.confidence = confidence
    }
}

public final class OCRRecognizer {
    public static let regionCount = 128
    public static let channels = 128
    public static let height = 8
    public static let width = 32
    public static let sequenceLength = 32
    public static let tokenCount = 858
    public static let featureDepth = 256
    public static let inputShape = [regionCount, channels, height, width]

    public let modelURL: URL
    public let model: MLModel
    public let charset: [String]

    public static var bundledModelURL: URL {
        get throws {
            try OCRModelURLs.bundled.recognizer
        }
    }

    public init(modelURL: URL? = nil, computeUnits: MLComputeUnits = .all, charset: [String]? = nil) throws {
        let resolvedModelURL = try modelURL ?? Self.bundledModelURL
        self.modelURL = resolvedModelURL
        self.model = try loadModel(at: resolvedModelURL, computeUnits: computeUnits)
        self.charset = try charset ?? Self.loadBundledCharset()
    }

    public func prediction(input regions: MLMultiArray) throws -> OCRRecognizerPrediction {
        try validateShape(regions, name: "regions", expected: Self.inputShape)

        let input = try MLDictionaryFeatureProvider(dictionary: ["regions": MLFeatureValue(multiArray: regions)])
        let predictionStart = DispatchTime.now()
        let output = try model.prediction(from: input)
        let predictionTimeMs = millisecondsSince(predictionStart)

        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("logits")
        }
        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("features")
        }

        return OCRRecognizerPrediction(
            output: OCRRecognizerOutput(logits: logits, features: features),
            predictionTimeMs: predictionTimeMs
        )
    }

    public func decode(logits: MLMultiArray, count: Int = regionCount) throws -> [OCRRecognizedSequence] {
        try Self.decode(logits: logits, charset: charset, count: count)
    }

    public static func makeZeroRegions() throws -> MLMultiArray {
        try zeroFloatArray(shape: [
            NSNumber(value: regionCount),
            NSNumber(value: channels),
            NSNumber(value: height),
            NSNumber(value: width),
        ])
    }

    public static func loadBundledCharset() throws -> [String] {
        let url = try bundledURL(resource: "charset", extension: "txt")
        let data = try Data(contentsOf: url)
        guard let charset = try JSONSerialization.jsonObject(with: data) as? [String] else {
            throw OCRCoreMLError.missingCharset
        }
        return charset
    }

    public static func decode(
        logits: MLMultiArray,
        charset: [String],
        count: Int = regionCount
    ) throws -> [OCRRecognizedSequence] {
        try validateShape(logits, name: "logits", expected: [regionCount, sequenceLength, tokenCount])
        let regionLimit = min(max(count, 0), regionCount)
        let values = logits.dataPointer.bindMemory(to: Float32.self, capacity: logits.count)
        let strides = logits.strides.map(\.intValue)
        var decoded: [OCRRecognizedSequence] = []
        decoded.reserveCapacity(regionLimit)

        for region in 0..<regionLimit {
            var text = ""
            var logProbability = 0.0
            var tokenSteps = 0

            for step in 0..<sequenceLength {
                var maxToken = 0
                var maxLogit = -Float.greatestFiniteMagnitude
                for token in 0..<tokenCount {
                    let value = values[region * strides[0] + step * strides[1] + token * strides[2]]
                    if value > maxLogit {
                        maxLogit = value
                        maxToken = token
                    }
                }

                let probability = softmaxProbability(
                    values: values,
                    strides: strides,
                    region: region,
                    step: step,
                    maxLogit: maxLogit
                )
                logProbability += log(max(Double(probability), 1e-8))
                tokenSteps += 1

                if maxToken == 0 {
                    continue
                }
                if maxToken == 1 {
                    break
                }
                if maxToken == 2 {
                    text.append("^")
                    continue
                }

                let charsetIndex = maxToken - 3
                if charset.indices.contains(charsetIndex) {
                    text.append(charset[charsetIndex])
                }
            }

            let confidence = tokenSteps > 0 ? exp(logProbability / Double(tokenSteps)) : 0
            decoded.append(OCRRecognizedSequence(text: text, confidence: confidence))
        }

        return decoded
    }
}

private func softmaxProbability(
    values: UnsafeMutablePointer<Float32>,
    strides: [Int],
    region: Int,
    step: Int,
    maxLogit: Float
) -> Float {
    var denominator: Float = 0
    for token in 0..<OCRRecognizer.tokenCount {
        let value = values[region * strides[0] + step * strides[1] + token * strides[2]]
        denominator += exp(value - maxLogit)
    }
    return 1 / denominator
}
