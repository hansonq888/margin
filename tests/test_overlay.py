import json
import urllib.request

from PIL import Image

from margin.overlay import Overlay, page_snapshot, referenced_line
from margin.session import Session
from margin.transcribe import Line, Transcription

LINES = ["F_parallel = mg sin theta", "F_parallel = 5(9.8) cos 37°", "F_parallel = 39.1 N"]


def test_referenced_line():
    assert referenced_line("Check line 2 in your work.", LINES) == 2
    assert referenced_line("Look at your second line.", LINES) == 2
    assert referenced_line("Your last line looks right.", LINES) == 3
    assert referenced_line("You wrote cos 37°, but along the ramp it should be sine.", LINES) == 2
    assert referenced_line("Nice work, keep going!", LINES) is None
    assert referenced_line("Check line 9.", LINES) is None


def test_page_snapshot_marks_flags_circles_and_crossed_out():
    s = Session()
    lines = [Line("Solve x^2 = 9", "x^2=9", None, "math", True, 0.9, True), Line("junk", None, None, "crossed_out", True, 0.5),
             Line("x = 3", "x=3", None, "math", True, 0.9)]
    block = s.apply((0, 0, 10, 10), Transcription(lines, True, 2, raw=None))
    block.flags.append({"line": "x = 3", "reason": "missing -3", "hint": "h"})
    snap = page_snapshot(s)
    p = snap[0]["problems"][0]
    assert [l["text"] for l in p["lines"]] == ["Solve x^2 = 9", "x = 3"]
    assert p["lines"][1]["circled"] and p["lines"][1]["flag"]["reason"] == "missing -3"
    assert snap[0]["crossed_out"] == ["junk"] and p["checked"]


def test_server_streams_state_and_serves_crops():
    o = Overlay(port=0)
    o.port = 0
    url = o.start()
    port = o._server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        o.heard("can you help me?")
        o.margin_says("Look at your second line.")
        o.margin_says("", done=True)
        o.set_crop(7, Image.new("RGB", (100, 50), "white"))
        state = json.loads(urllib.request.urlopen(base + "/state").read())
        assert [t["who"] for t in state["turns"]] == ["you", "margin"]
        assert state["turns"][1]["text"] == "Look at your second line." and not state["turns"][1]["streaming"]
        assert urllib.request.urlopen(base + "/crop/7?v=1").headers["Content-Type"] == "image/jpeg"
        assert b"Margin" in urllib.request.urlopen(base + "/").read()
        stream = urllib.request.urlopen(base + "/events", timeout=5)
        first = stream.readline().decode()
        assert first.startswith("data: ") and json.loads(first[6:])["turns"][0]["text"] == "can you help me?"
    finally:
        o.stop()
