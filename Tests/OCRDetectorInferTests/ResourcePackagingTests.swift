import Foundation
import OCRCoreMLDetector
import Testing

@Test func bundledDetectorResourcesExist() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let detectorResources = packageRoot.appendingPathComponent("Sources/OCRCoreMLDetector/Resources")
    let inferResources = packageRoot.appendingPathComponent("Sources/OCRDetectorInfer/Resources")
    let expectedFiles = [
        "Detector224.mlpackage/Manifest.json",
        "Detector224.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "Detector224.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    ]
    let expectedInferFiles = [
        "ocr-example-input-1.png",
    ]

    for file in expectedFiles {
        #expect(FileManager.default.fileExists(atPath: detectorResources.appendingPathComponent(file).path))
    }
    for file in expectedInferFiles {
        #expect(FileManager.default.fileExists(atPath: inferResources.appendingPathComponent(file).path))
    }
}

@Test func bundledDetectorModelURLIsAvailable() throws {
    let modelURL = try OCRDetector.bundledModelURL
    #expect(FileManager.default.fileExists(atPath: modelURL.path))
}
