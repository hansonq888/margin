// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "margin-overlay",
    platforms: [.macOS(.v14)],
    targets: [.executableTarget(name: "margin-overlay", path: "Sources/margin-overlay")]
)
