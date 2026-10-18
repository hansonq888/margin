// ipad-capture: read a USB-connected iPad's screen as a video device and write frames to disk.
//
//   ipad-capture --list
//   ipad-capture --out ../frames [--fps 4] [--record] [--seconds N]
//
// Writes <out>/latest.jpg atomically at the requested rate. With --record, also writes
// numbered frames to <out>/rec/ for offline replay.

import AVFoundation
import CoreImage
import CoreMediaIO
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Arguments

struct Options {
    var list = false
    var out = "frames"
    var fps = 4.0
    var record = false
    var timeout = 15.0
    var seconds = 0.0  // 0 = run until Ctrl-C
}

func parseOptions() -> Options {
    var opts = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        switch arg {
        case "--list": opts.list = true
        case "--out": opts.out = args.next() ?? opts.out
        case "--fps": opts.fps = Double(args.next() ?? "") ?? opts.fps
        case "--record": opts.record = true
        case "--timeout": opts.timeout = Double(args.next() ?? "") ?? opts.timeout
        case "--seconds": opts.seconds = Double(args.next() ?? "") ?? opts.seconds
        case "-h", "--help":
            print("usage: ipad-capture [--list] [--out DIR] [--fps N] [--record] [--timeout SECONDS] [--seconds N]")
            exit(0)
        default:
            fputs("unknown argument: \(arg)\n", stderr)
            exit(2)
        }
    }
    return opts
}

// MARK: - Device discovery

/// Opt this process in to seeing iOS/iPadOS screens as capture devices (what QuickTime does).
func allowScreenCaptureDevices() {
    var address = CMIOObjectPropertyAddress(
        mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyAllowScreenCaptureDevices),
        mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
        mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain)
    )
    var allow: UInt32 = 1
    let status = CMIOObjectSetPropertyData(
        CMIOObjectID(kCMIOObjectSystemObject), &address, 0, nil,
        UInt32(MemoryLayout<UInt32>.size), &allow
    )
    if status != 0 {
        fputs("warning: could not enable screen capture devices (CMIO status \(status))\n", stderr)
    }
}

func screenDevices() -> [AVCaptureDevice] {
    AVCaptureDevice.DiscoverySession(
        deviceTypes: [.external], mediaType: nil, position: .unspecified
    ).devices.filter { $0.hasMediaType(.muxed) }
}

/// Devices show up a moment after opting in, so poll while spinning the run loop.
func waitForDevices(timeout: Double) -> [AVCaptureDevice] {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        let found = screenDevices()
        if !found.isEmpty { return found }
        RunLoop.main.run(until: Date().addingTimeInterval(0.5))
    }
    return []
}

func requestCameraAccess() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized: return true
    case .notDetermined:
        let sem = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .video) { ok in granted = ok; sem.signal() }
        sem.wait()
        return granted
    default: return false
    }
}

// MARK: - Frame writer

final class FrameWriter: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let outDir: URL
    let recDir: URL?
    let minInterval: Double
    let context = CIContext()
    var lastWrite = 0.0
    var lastFrameAt = CFAbsoluteTimeGetCurrent()  // for the stall watchdog
    var received = 0
    var written = 0
    var lastReport = Date()
    var size = CGSize.zero

    init(outDir: URL, fps: Double, record: Bool) {
        self.outDir = outDir
        self.minInterval = 1.0 / max(fps, 0.1)
        self.recDir = record ? outDir.appendingPathComponent("rec") : nil
        super.init()
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        if let recDir { try? FileManager.default.createDirectory(at: recDir, withIntermediateDirectories: true) }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        received += 1
        let now = CFAbsoluteTimeGetCurrent()
        lastFrameAt = now
        reportIfDue()
        guard now - lastWrite >= minInterval, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastWrite = now

        let image = CIImage(cvPixelBuffer: pixels)
        size = image.extent.size
        guard let cg = context.createCGImage(image, from: image.extent) else { return }

        let tmp = outDir.appendingPathComponent(".latest.tmp.jpg")
        let latest = outDir.appendingPathComponent("latest.jpg")
        if writeJPEG(cg, to: tmp) {
            _ = try? FileManager.default.replaceItemAt(latest, withItemAt: tmp)
            if !FileManager.default.fileExists(atPath: latest.path) {
                try? FileManager.default.moveItem(at: tmp, to: latest)
            }
        }
        if let recDir {
            let name = String(format: "frame_%06d_%.3f.jpg", written, now)
            _ = writeJPEG(cg, to: recDir.appendingPathComponent(name))
        }
        written += 1
    }

    func writeJPEG(_ image: CGImage, to url: URL) -> Bool {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            return false
        }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        return CGImageDestinationFinalize(dest)
    }

    func reportIfDue() {
        let elapsed = Date().timeIntervalSince(lastReport)
        guard elapsed >= 5 else { return }
        print(String(format: "[capture] %dx%d  device fps %.1f  written %d",
                     Int(size.width), Int(size.height), Double(received) / elapsed, written))
        received = 0
        lastReport = Date()
    }
}

// MARK: - Main

let opts = parseOptions()
allowScreenCaptureDevices()

print("Looking for a USB-connected iPad (unlock it and tap \"Trust\" if asked)...")
let devices = waitForDevices(timeout: opts.timeout)

if opts.list {
    if devices.isEmpty {
        print("No iPad/iPhone screen devices found.")
        exit(1)
    }
    for d in devices { print("- \(d.localizedName)  [\(d.modelID)]  id=\(d.uniqueID)") }
    exit(0)
}

guard devices.first != nil else {
    fputs("No iPad found. Plug it in via USB, unlock it, and trust this Mac.\n", stderr)
    exit(1)
}
guard requestCameraAccess() else {
    fputs("Camera access denied. Enable it for your terminal in System Settings > Privacy & Security > Camera.\n", stderr)
    exit(1)
}

// Use the sharpest mode the iPad offers: handwriting reads better at full resolution.
func pixels(_ f: AVCaptureDevice.Format) -> Int32 {
    let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
    return d.width * d.height
}

let outDir = URL(fileURLWithPath: opts.out, isDirectory: true)
let writer = FrameWriter(outDir: outDir, fps: opts.fps, record: opts.record)
let frameQueue = DispatchQueue(label: "frames")

/// Build and start a capture session for the iPad (again, after a stall).
func startSession(_ device: AVCaptureDevice) -> AVCaptureSession? {
    let session = AVCaptureSession()
    do {
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { fputs("[capture] cannot add iPad input\n", stderr); return nil }
        session.addInput(input)
    } catch {
        fputs("[capture] failed to open \(device.localizedName): \(error)\n", stderr)
        return nil
    }
    if let best = device.formats.max(by: { pixels($0) < pixels($1) }), pixels(best) > pixels(device.activeFormat) {
        if (try? device.lockForConfiguration()) != nil {
            device.activeFormat = best
            device.unlockForConfiguration()
        }
    }
    let output = AVCaptureVideoDataOutput()
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    output.alwaysDiscardsLateVideoFrames = true
    output.setSampleBufferDelegate(writer, queue: frameQueue)
    guard session.canAddOutput(output) else { fputs("[capture] cannot add video output\n", stderr); return nil }
    session.addOutput(output)

    // Say why, if the system interrupts or breaks the stream.
    let nc = NotificationCenter.default
    nc.addObserver(forName: AVCaptureSession.runtimeErrorNotification, object: session, queue: .main) { note in
        print("[capture] stream error: \(note.userInfo?[AVCaptureSessionErrorKey] ?? "unknown")")
    }
    nc.addObserver(forName: AVCaptureSession.wasInterruptedNotification, object: session, queue: .main) { _ in
        print("[capture] stream interrupted by the system")
    }
    nc.addObserver(forName: AVCaptureSession.interruptionEndedNotification, object: session, queue: .main) { _ in
        print("[capture] interruption ended")
    }
    session.startRunning()
    return session
}

NotificationCenter.default.addObserver(forName: AVCaptureDevice.wasDisconnectedNotification, object: nil, queue: .main) { note in
    if let d = note.object as? AVCaptureDevice, d.hasMediaType(.muxed) {
        print("[capture] \(d.localizedName) disconnected (cable? locked?)")
    }
}

var device = devices.first!
guard var session = startSession(device) else { exit(1) }
let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
print(dims.width > 0 ? "Format: \(dims.width)x\(dims.height)" : "Format: chosen by the device (size shown in the first [capture] report)")
print("Capturing \(device.localizedName) → \(outDir.path)/latest.jpg at \(opts.fps) fps\(opts.record ? " (recording)" : ""). Ctrl-C to stop.")

// Watchdog: the iPad streams continuously, even on a still screen, so silence means the stream stalled.
// Restart it; if that doesn't help, reconnect to the iPad from scratch.
var stage = 0  // 0 healthy, 1 restarted, 2+ rebuilt
var stageAt = CFAbsoluteTimeGetCurrent()
var warned = false
let watchdog = Timer(timeInterval: 1.0, repeats: true) { _ in
    let now = CFAbsoluteTimeGetCurrent()
    let silent = now - writer.lastFrameAt
    if silent < 1.5 {
        if stage > 0 { print("[capture] frames are back") }
        stage = 0
        warned = false
        return
    }
    guard silent > 3, now - stageAt > 3 else { return }
    stageAt = now
    if stage == 0 {
        print(String(format: "[capture] no frames for %.0fs, restarting the stream...", silent))
        session.stopRunning()
        session.startRunning()
        stage = 1
    } else {
        print("[capture] still nothing; reconnecting to the iPad...")
        session.stopRunning()
        if let again = screenDevices().first, let fresh = startSession(again) {
            device = again
            session = fresh
        }
        stage += 1
        if stage >= 3 && !warned {
            warned = true
            print("[capture] the iPad isn't sending its screen. Is it unlocked and awake? Try unplugging and replugging it.")
        }
    }
}
RunLoop.main.add(watchdog, forMode: .common)

signal(SIGINT) { _ in exit(0) }
if opts.seconds > 0 {
    RunLoop.main.run(until: Date().addingTimeInterval(opts.seconds))
    session.stopRunning()
    print("Done: wrote \(writer.written) frames.")
} else {
    RunLoop.main.run()
}
