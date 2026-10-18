"""The on-screen overlay: what Margin heard and said, which line it means, and what it read.

A tiny local web server (127.0.0.1 only) serves one page and pushes the live state to it with Server-Sent
Events. The page is shown in a small always-on-top window (overlay-app/, Swift) or any browser tab.
Nothing here leaves the Mac.
"""

import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue

from PIL import Image

UI = Path(__file__).parent / "overlay_ui" / "index.html"


class Overlay:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host, self.port = host, port
        self.state: dict = {
            "status": "watching",  # watching | listening | thinking | speaking | looking
            "detail": "",
            "course": None,
            "mode": "ask",
            "assignment": None,
            "turns": [],  # [{"who": "you"|"margin", "text": ..., "streaming": bool}]
            "page": [],  # blocks -> problems -> lines (see page_snapshot)
            "highlight": None,  # {"block": id, "line": text, "kind": "mistake"|"help"|"talk"}
            "privacy": {"voice": "on this Mac", "ipad": False},
        }
        self._lock = threading.Lock()
        self._clients: list[Queue] = []
        self._crops: dict[int, bytes] = {}
        self._server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ server

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        overlay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal clean
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", UI.read_bytes())
                elif path == "/state":
                    self._send(200, "application/json", json.dumps(overlay.snapshot()).encode())
                elif path.startswith("/crop/"):
                    data = overlay._crops.get(int(re.sub(r"\D", "", path) or -1))
                    self._send(200, "image/jpeg", data) if data else self._send(404, "text/plain", b"no crop")
                elif path == "/events":
                    self._events()
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code, kind, body):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                q: Queue = Queue()
                with overlay._lock:
                    overlay._clients.append(q)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(f"data: {json.dumps(overlay.snapshot())}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                            self.wfile.write(f"data: {msg}\n\n".encode())
                        except Empty:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with overlay._lock:
                        if q in overlay._clients:
                            overlay._clients.remove(q)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.url

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    # ------------------------------------------------------------------ state

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def update(self, **fields) -> None:
        with self._lock:
            changed = any(self.state.get(k) != v for k, v in fields.items())
            self.state.update(fields)
            if not changed:
                return
            msg = json.dumps(self.state)
            for q in self._clients:
                q.put(msg)

    def set_status(self, status: str, detail: str = "") -> None:
        self.update(status=status, detail=detail)

    def heard(self, text: str) -> None:
        turns = self.state["turns"][-7:] + [{"who": "you", "text": text, "streaming": False}]
        self.update(turns=turns, highlight=None, status="thinking", detail="")

    def margin_says(self, sentence: str, done: bool = False) -> None:
        """Append a streamed sentence to Margin's current turn (starting one if needed)."""
        turns = [dict(t) for t in self.state["turns"]]
        if not turns or turns[-1]["who"] != "margin" or not turns[-1]["streaming"]:
            turns = turns[-7:] + [{"who": "margin", "text": "", "streaming": True}]
        if sentence:
            turns[-1]["text"] = (turns[-1]["text"] + " " + sentence).strip()
        turns[-1]["streaming"] = not done
        self.update(turns=turns)

    def set_crop(self, block_id: int, crop: Image.Image) -> None:
        img = crop.convert("RGB")
        img.thumbnail((640, 640))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        self._crops[block_id] = buf.getvalue()

    def highlight(self, block_id: int | None, line_text: str | None, kind: str) -> None:
        if block_id is None or not line_text:
            return
        self.update(highlight={"block": block_id, "line": line_text, "kind": kind, "at": time.time()})


def page_snapshot(session, coach=None, matched=None) -> list[dict]:
    """The page as the overlay shows it: blocks of problems, each line with what Margin knows about it."""
    focus = coach.focus[1] if coach and coach.focus else None
    out = []
    for block in sorted(session.blocks.values(), key=lambda b: (b.bbox[1], b.bbox[0])):
        judged = block.judgeable_problems()
        problems = []
        for problem in block.problems():
            q = matched([l.text for l in problem]) if matched else None
            problems.append({
                "question": {"label": q.label, "text": q.text} if q else None,
                "focus": bool(focus and problem[0].text == focus[0].text),
                "checked": any(problem[0] is j[0] for j in judged),
                "lines": [
                    {
                        "n": i,
                        "text": line.text,
                        "latex": line.latex,
                        "kind": line.kind,
                        "confidence": line.confidence,
                        "circled": line is block.circled,
                        "flag": ({"reason": f["reason"], "hint": f.get("hint", "")}
                                 if (f := session.flag_for(block, problem, line)) else None),
                    }
                    for i, line in enumerate(problem, 1)
                ],
            })
        crossed = [l.text for l in block.lines if l.kind == "crossed_out"]
        out.append({
            "id": block.id,
            "version": block.version,
            "clipped": sorted(block.clipped),
            "problems": problems,
            "crossed_out": crossed,
            "crop": f"/crop/{block.id}?v={block.version}",
        })
    return out


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}


def referenced_line(reply: str, lines: list[str]) -> int | None:
    """Which line (1-based) Margin's reply is about: "line 3", "your second line", "the last line", or the line
    whose distinctive bits it quotes ("you wrote cos 37°")."""
    text = reply.lower()
    m = re.search(r"\bline (\d+)\b", text)
    if m and 0 < int(m.group(1)) <= len(lines):
        return int(m.group(1))
    m = re.search(r"\b(first|second|third|fourth|fifth|sixth|last) line\b", text)
    if m:
        n = len(lines) if m.group(1) == "last" else _ORDINALS[m.group(1)]
        return n if 0 < n <= len(lines) else None
    tokens = lambda s: set(re.findall(r"[a-z]+\d*|\d+(?:\.\d+)?°?", s.lower())) - {"the", "and", "you", "your", "is", "it"}
    said = tokens(text)
    scores = [len(tokens(line) & said) for line in lines]
    best = max(scores, default=0)
    if best >= 2 and scores.count(best) == 1:
        return scores.index(best) + 1
    return None
