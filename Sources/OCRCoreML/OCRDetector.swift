import CoreGraphics
import CoreML
import Foundation
import ImageIO

public enum OCRDetectorError: Error, CustomStringConvertible {
    case imageDecodeFailed(URL)
    case imageContextFailed

    public var description: String {
        switch self {
        case .imageDecodeFailed(let url):
            "Could not decode image at \(url.path)"
        case .imageContextFailed:
            "Could not create image resize context"
        }
    }
}

public struct OCRDetectorOutput {
    public let prob: MLMultiArray
    public let rboxes: MLMultiArray
    public let features: MLMultiArray

    public init(prob: MLMultiArray, rboxes: MLMultiArray, features: MLMultiArray) {
        self.prob = prob
        self.rboxes = rboxes
        self.features = features
    }
}

public struct OCRDetectorPrediction {
    public let output: OCRDetectorOutput
    public let predictionTimeMs: Double

    public init(output: OCRDetectorOutput, predictionTimeMs: Double) {
        self.output = output
        self.predictionTimeMs = predictionTimeMs
    }
}

public final class OCRDetector {
    public static let inputWidth = 768
    public static let inputHeight = 768
    public static let inputShape = [1, 3, inputHeight, inputWidth]

    public let modelURL: URL
    public let model: MLModel

    public static var bundledModelURL: URL {
        get throws {
            try OCRModelURLs.bundled.detector
        }
    }

    public init(modelURL: URL? = nil, computeUnits: MLComputeUnits = .all) throws {
        let resolvedModelURL = try modelURL ?? Self.bundledModelURL

        self.modelURL = resolvedModelURL
        self.model = try loadModel(at: resolvedModelURL, computeUnits: computeUnits)
    }

    public func prediction(for image: CGImage) throws -> OCRDetectorPrediction {
        let imageArray = try Self.makeInputArray(from: image)
        return try prediction(input: imageArray)
    }

    public func prediction(forImageAt url: URL) throws -> OCRDetectorPrediction {
        let image = try Self.loadImage(url)
        return try prediction(for: image)
    }

    public func prediction(input imageArray: MLMultiArray) throws -> OCRDetectorPrediction {
        try validateShape(imageArray, name: "image", expected: Self.inputShape)

        let input = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: imageArray)])
        let predictionStart = DispatchTime.now()
        let output = try model.prediction(from: input)
        let predictionTimeMs = millisecondsSince(predictionStart)

        guard let prob = output.featureValue(for: "prob")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("prob")
        }
        guard let rboxes = output.featureValue(for: "rboxes")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("rboxes")
        }
        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw OCRCoreMLError.missingOutput("features")
        }

        return OCRDetectorPrediction(
            output: OCRDetectorOutput(prob: prob, rboxes: rboxes, features: features),
            predictionTimeMs: predictionTimeMs
        )
    }

    public static func makeInputArray(from image: CGImage) throws -> MLMultiArray {
        let width = inputWidth
        let height = inputHeight
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixels = [UInt8](repeating: 0, count: height * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue

        guard
            let context = CGContext(
                data: &pixels,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        else {
            throw OCRDetectorError.imageContextFailed
        }

        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        let values = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
        let strides = array.strides.map(\.intValue)

        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = y * bytesPerRow + x * bytesPerPixel
                setMultiArray(values, strides, channel: 0, y: y, x: x, Float32(pixels[pixelOffset]) / 255.0)
                setMultiArray(values, strides, channel: 1, y: y, x: x, Float32(pixels[pixelOffset + 1]) / 255.0)
                setMultiArray(values, strides, channel: 2, y: y, x: x, Float32(pixels[pixelOffset + 2]) / 255.0)
            }
        }

        return array
    }

    public static func loadImage(_ url: URL) throws -> CGImage {
        guard
            let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw OCRDetectorError.imageDecodeFailed(url)
        }
        return image
    }

}

private func setMultiArray(
    _ values: UnsafeMutablePointer<Float32>,
    _ strides: [Int],
    channel: Int,
    y: Int,
    x: Int,
    _ value: Float32
) {
    let offset = channel * strides[1] + y * strides[2] + x * strides[3]
    values[offset] = value
}
