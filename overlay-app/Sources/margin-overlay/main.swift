// margin-overlay: Margin's on-screen overlay as a small floating window.
//
//   margin-overlay [http://127.0.0.1:8765/]
//
// Always on top, on every Space, never steals focus from GoodNotes / your browser. Drag it by its top edge;
// it remembers where you put it. Quit with Cmd-Q while it's focused, or close the window.

import AppKit
import WebKit

final class Delegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, NSWindowDelegate {
    let url: URL
    var panel: NSPanel!
    var web: WKWebView!

    init(url: URL) { self.url = url }

    func applicationDidFinishLaunching(_ note: Notification) {
        let size = NSSize(width: 380, height: 560)
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let origin = NSPoint(x: screen.maxX - size.width - 20, y: screen.maxY - size.height - 20)

        panel = NSPanel(
            contentRect: NSRect(origin: origin, size: size),
            styleMask: [.titled, .closable, .resizable, .fullSizeContentView, .nonactivatingPanel, .utilityWindow],
            backing: .buffered, defer: false
        )
        panel.title = "Margin"
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.isMovableByWindowBackground = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.minSize = NSSize(width: 300, height: 320)
        panel.delegate = self
        panel.setFrameAutosaveName("MarginOverlay")

        let config = WKWebViewConfiguration()
        web = WKWebView(frame: panel.contentView!.bounds, configuration: config)
        web.autoresizingMask = [.width, .height]
        web.navigationDelegate = self
        web.setValue(false, forKey: "drawsBackground")  // let the page's translucent background show through
        panel.contentView?.wantsLayer = true
        panel.contentView?.layer?.cornerRadius = 14
        panel.contentView?.layer?.masksToBounds = true
        panel.contentView?.addSubview(web)
        web.load(URLRequest(url: url))
        panel.orderFrontRegardless()
    }

    // Margin may not be up yet (or restarted): keep retrying.
    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        retry()
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        retry()
    }

    func retry() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self else { return }
            self.web.load(URLRequest(url: self.url))
        }
    }

    func windowWillClose(_ notification: Notification) { NSApp.terminate(nil) }
}

let arg = CommandLine.arguments.dropFirst().first ?? "http://127.0.0.1:8765/"
guard let url = URL(string: arg) else { fputs("bad URL: \(arg)\n", stderr); exit(2) }
let app = NSApplication.shared
app.setActivationPolicy(.accessory)  // no Dock icon, no menu bar takeover
let delegate = Delegate(url: url)
app.delegate = delegate
app.run()
