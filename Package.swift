// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "OCRCoreMLDetector",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "OCRCoreMLDetector", targets: ["OCRCoreMLDetector"]),
        .executable(name: "ocr-detector-infer", targets: ["OCRDetectorInfer"])
    ],
    targets: [
        .target(
            name: "OCRCoreMLDetector",
            resources: [
                .copy("Resources/Detector224.mlpackage"),
            ]
        ),
        .executableTarget(
            name: "OCRDetectorInfer",
            dependencies: ["OCRCoreMLDetector"],
            resources: [
                .copy("Resources/ocr-example-input-1.png"),
            ]
        ),
        .testTarget(
            name: "OCRDetectorInferTests",
            dependencies: ["OCRCoreMLDetector"]
        )
    ]
)
