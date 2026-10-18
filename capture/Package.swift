// swift-tools-version:5.9
import Foundation
import PackageDescription

// Embed Info.plist so macOS shows a proper camera-permission prompt for this CLI.
let infoPlist = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Info.plist")
    .path

let package = Package(
    name: "ipad-capture",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "ipad-capture",
            path: "Sources/ipad-capture",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlist,
                ]),
            ]
        ),
    ]
)
