import Foundation
import OCRCoreML
import Testing

@Test func bundledPipelineResourcesExist() throws {
    let testFile = URL(fileURLWithPath: #filePath)
    let packageRoot = testFile
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let modelResources = packageRoot.appendingPathComponent("Sources/OCRCoreML/Resources")
    let smokeResources = packageRoot.appendingPathComponent("Sources/OCRCoreMLSmoke/Resources")
    let expectedFiles = [
        "Detector224.mlpackage/Manifest.json",
        "Detector224.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "Detector224.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "RecognizerFeaturesInt8.mlpackage/Manifest.json",
        "RecognizerFeaturesInt8.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "RecognizerFeaturesInt8.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "RelationalInt8.mlpackage/Manifest.json",
        "RelationalInt8.mlpackage/Data/com.apple.CoreML/model.mlmodel",
        "RelationalInt8.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        "charset.txt",
        "model_config.json",
    ]

    for file in expectedFiles {
        #expect(FileManager.default.fileExists(atPath: modelResources.appendingPathComponent(file).path))
    }
    #expect(FileManager.default.fileExists(atPath: smokeResources.appendingPathComponent("ocr-example-input-1.png").path))
}

@Test func bundledModelURLsAreAvailable() throws {
    let urls = try OCRModelURLs.bundled
    #expect(FileManager.default.fileExists(atPath: urls.detector.path))
    #expect(FileManager.default.fileExists(atPath: urls.recognizer.path))
    #expect(FileManager.default.fileExists(atPath: urls.relational.path))
}

@Test func bundledCharsetLoads() throws {
    let charset = try OCRRecognizer.loadBundledCharset()
    #expect(charset.count == 855)
    #expect(charset.first == " ")
}
