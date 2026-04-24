import CoreML
import Foundation
import OCRCoreMLDetector

private struct Options {
    var imageURL: URL?
    var modelURL: URL?
    var computeUnits: MLComputeUnits = .all
    var warmupCount = 1
    var repeatCount = 1
}

private enum CLIError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case missingResource(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            message
        case .missingResource(let name):
            "Missing bundled resource: \(name)"
        }
    }
}

private func usage() -> String {
    """
    Usage:
      ocr-detector-infer [image-path] [--model path] [--compute-units all|gpu|ane|cpu] [--warmup N] [--repeat N]

    Defaults:
      image: bundled ocr-example-input-1.png
      model: bundled Detector224.mlpackage
      compute units: all
      warmup: 1 prediction
    """
}

private func parseOptions(_ arguments: [String]) throws -> Options {
    var options = Options()
    var index = 0

    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "-h", "--help":
            print(usage())
            exit(0)
        case "--model":
            index += 1
            guard index < arguments.count else {
                throw CLIError.invalidArgument("--model requires a path")
            }
            options.modelURL = URL(fileURLWithPath: arguments[index])
        case "--compute-units":
            index += 1
            guard index < arguments.count else {
                throw CLIError.invalidArgument("--compute-units requires a value")
            }
            options.computeUnits = try parseComputeUnits(arguments[index])
        case "--warmup":
            index += 1
            guard index < arguments.count, let count = Int(arguments[index]), count >= 0 else {
                throw CLIError.invalidArgument("--warmup requires a non-negative integer")
            }
            options.warmupCount = count
        case "--repeat":
            index += 1
            guard index < arguments.count, let count = Int(arguments[index]), count > 0 else {
                throw CLIError.invalidArgument("--repeat requires a positive integer")
            }
            options.repeatCount = count
        default:
            if argument.hasPrefix("-") {
                throw CLIError.invalidArgument("Unknown option: \(argument)")
            }
            if options.imageURL != nil {
                throw CLIError.invalidArgument("Only one image path can be supplied")
            }
            options.imageURL = URL(fileURLWithPath: argument)
        }
        index += 1
    }

    return options
}

private func parseComputeUnits(_ raw: String) throws -> MLComputeUnits {
    switch raw.lowercased().replacingOccurrences(of: "_", with: "-") {
    case "all":
        return .all
    case "gpu", "cpu-and-gpu", "cpu+gpu":
        return .cpuAndGPU
    case "ane", "ne", "cpu-and-ne", "cpu+ne", "cpu-and-neural-engine":
        return .cpuAndNeuralEngine
    case "cpu", "cpu-only":
        return .cpuOnly
    default:
        throw CLIError.invalidArgument("Unsupported compute units: \(raw)")
    }
}

private func bundledURL(resource: String, extension ext: String) throws -> URL {
    guard let url = Bundle.module.url(forResource: resource, withExtension: ext) else {
        throw CLIError.missingResource("\(resource).\(ext)")
    }
    return url
}

private func millisecondsSince(_ start: DispatchTime) -> Double {
    let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    return Double(elapsed) / 1_000_000.0
}

private func describe(_ array: MLMultiArray) -> String {
    let shape = array.shape.map(\.stringValue).joined(separator: "x")
    let values = array.dataPointer.bindMemory(to: Float32.self, capacity: array.count)
    var minValue = Float32.greatestFiniteMagnitude
    var maxValue = -Float32.greatestFiniteMagnitude
    var sum = Float64(0)

    for index in 0..<array.count {
        let value = values[index]
        minValue = min(minValue, value)
        maxValue = max(maxValue, value)
        sum += Float64(value)
    }

    let mean = sum / Double(array.count)
    return "shape=\(shape) min=\(String(format: "%.6f", minValue)) max=\(String(format: "%.6f", maxValue)) mean=\(String(format: "%.6f", mean))"
}

private func run() throws {
    let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
    let imageURL = try options.imageURL ?? bundledURL(resource: "ocr-example-input-1", extension: "png")

    let loadStart = DispatchTime.now()
    let detector = try OCRDetector(modelURL: options.modelURL, computeUnits: options.computeUnits)
    let loadMs = millisecondsSince(loadStart)

    let preprocessStart = DispatchTime.now()
    let image = try OCRDetector.loadImage(imageURL)
    let imageArray = try OCRDetector.makeInputArray(from: image)
    let preprocessMs = millisecondsSince(preprocessStart)

    for _ in 0..<options.warmupCount {
        _ = try detector.prediction(input: imageArray)
    }

    var predictionMs: [Double] = []
    var lastPrediction: OCRDetectorPrediction?
    for _ in 0..<options.repeatCount {
        let prediction = try detector.prediction(input: imageArray)
        predictionMs.append(prediction.predictionTimeMs)
        lastPrediction = prediction
    }

    guard let prediction = lastPrediction else {
        throw CLIError.invalidArgument("No prediction was run")
    }

    print("model: \(detector.modelURL.path)")
    print("image: \(imageURL.path)")
    print("computeUnits: \(options.computeUnits)")
    print("loadMs: \(String(format: "%.2f", loadMs))")
    print("preprocessMs: \(String(format: "%.2f", preprocessMs))")
    print("warmupCount: \(options.warmupCount)")
    print("predictionMs: \(predictionMs.map { String(format: "%.2f", $0) }.joined(separator: ", "))")
    print("prob: \(describe(prediction.output.prob))")
    print("rboxes: \(describe(prediction.output.rboxes))")
    print("features: \(describe(prediction.output.features))")
}

do {
    try run()
} catch {
    fputs("error: \(error)\n\n\(usage())\n", stderr)
    exit(1)
}
