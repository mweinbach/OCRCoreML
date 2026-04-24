// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "OCRCoreML",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "OCRCoreML", targets: ["OCRCoreML"]),
        .executable(name: "ocr-coreml-smoke", targets: ["OCRCoreMLSmoke"])
    ],
    targets: [
        .target(
            name: "OCRCoreML",
            resources: [
                .copy("Resources/Detector224.mlpackage"),
                .copy("Resources/RecognizerFeaturesInt8.mlpackage"),
                .copy("Resources/RelationalInt8.mlpackage"),
                .copy("Resources/charset.txt"),
                .copy("Resources/model_config.json"),
            ]
        ),
        .executableTarget(
            name: "OCRCoreMLSmoke",
            dependencies: ["OCRCoreML"],
            resources: [
                .copy("Resources/ocr-example-input-1.png"),
            ]
        ),
        .testTarget(
            name: "OCRCoreMLTests",
            dependencies: ["OCRCoreML"]
        )
    ]
)
