import CoreML
import Foundation

public enum OCRCoreMLError: Error, CustomStringConvertible {
    case missingBundledResource(String)
    case invalidInputShape(name: String, expected: [Int], actual: [Int])
    case missingOutput(String)
    case missingCharset

    public var description: String {
        switch self {
        case .missingBundledResource(let name):
            "Missing bundled resource: \(name)"
        case .invalidInputShape(let name, let expected, let actual):
            "Invalid \(name) shape. Expected \(expected), got \(actual)"
        case .missingOutput(let name):
            "Model output \(name) was not returned"
        case .missingCharset:
            "Could not load bundled charset.txt"
        }
    }
}

public struct OCRModelURLs {
    public let detector: URL
    public let recognizer: URL
    public let relational: URL

    public init(detector: URL, recognizer: URL, relational: URL) {
        self.detector = detector
        self.recognizer = recognizer
        self.relational = relational
    }

    public static var bundled: OCRModelURLs {
        get throws {
            OCRModelURLs(
                detector: try bundledURL(resource: "DetectorGPUInt8_768", extension: "mlpackage"),
                recognizer: try bundledURL(resource: "RecognizerFeaturesInt8", extension: "mlpackage"),
                relational: try bundledURL(resource: "RelationalInt8", extension: "mlpackage")
            )
        }
    }
}

internal func bundledURL(resource: String, extension ext: String) throws -> URL {
    guard let url = Bundle.module.url(forResource: resource, withExtension: ext) else {
        throw OCRCoreMLError.missingBundledResource("\(resource).\(ext)")
    }
    return url
}

internal func loadModel(at url: URL, computeUnits: MLComputeUnits) throws -> MLModel {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    if url.pathExtension == "mlmodelc" {
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    let compiledURL = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiledURL, configuration: configuration)
}

internal func millisecondsSince(_ start: DispatchTime) -> Double {
    let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(elapsed) / 1_000_000.0
}

internal func validateShape(_ array: MLMultiArray, name: String, expected: [Int]) throws {
    let actual = array.shape.map(\.intValue)
    guard actual == expected else {
        throw OCRCoreMLError.invalidInputShape(name: name, expected: expected, actual: actual)
    }
}

internal func zeroFloatArray(shape: [NSNumber]) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape, dataType: .float32)
    let values = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
    for index in 0..<array.count {
        values[index] = 0
    }
    return array
}
