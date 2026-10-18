import numpy as np
from PIL import Image, ImageDraw

from margin.ink import InkRegion, InkTracker, PageChange, Scroll, estimate_shift

W, H = 1200, 900


def notebook_page(height: int = 3000, seed: int = 0) -> Image.Image:
    """A tall synthetic page with ruled lines and scribbles, so scrolling has texture to lock onto."""
    rng = np.random.default_rng(seed)
    page = Image.new("L", (W, height), 245)
    d = ImageDraw.Draw(page)
    for y in range(60, height, 48):
        d.line([(0, y), (W, y)], fill=210, width=1)
    for _ in range(120):
        x, y = int(rng.integers(40, W - 200)), int(rng.integers(40, height - 40))
        pts = [(x + i * 12, y + int(rng.integers(-10, 10))) for i in range(12)]
        d.line(pts, fill=30, width=3)
    return page


def view(page: Image.Image, top: int) -> Image.Image:
    return page.crop((0, top, W, top + H)).convert("RGB")


def write(img: Image.Image, x: int, y: int) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.line([(x, y), (x + 60, y - 20), (x + 120, y + 10), (x + 180, y - 15)], fill=(0, 0, 0), width=4)
    return img


def test_new_ink_is_emitted_only_after_idle():
    page = view(notebook_page(), 0)
    tracker = InkTracker(idle_seconds=1.5)
    assert isinstance(tracker.update(page, 0.0)[0], PageChange)

    inked = write(page, 400, 500)
    assert tracker.update(inked, 1.0) == []  # still writing
    assert tracker.update(inked, 2.0) == []  # only 1.0s idle
    events = tracker.update(inked, 2.6)
    regions = [e for e in events if isinstance(e, InkRegion)]
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0].bbox
    assert x0 <= 400 and x1 >= 580 and y0 <= 480 and y1 >= 510
    assert regions[0].kind == "added"

    assert tracker.update(inked, 5.0) == []  # committed; no repeat


def test_erasing_is_reported_as_erased():
    page = view(notebook_page(), 0)
    inked = write(page, 400, 500)
    tracker = InkTracker(idle_seconds=1.0)
    tracker.update(inked, 0.0)
    tracker.update(page, 0.5)
    events = tracker.update(page, 2.0)
    assert [e.kind for e in events if isinstance(e, InkRegion)] == ["erased"]


def test_scroll_is_not_new_ink():
    page = notebook_page()
    tracker = InkTracker()
    tracker.update(view(page, 400), 0.0)
    events = tracker.update(view(page, 600), 0.2)  # content moves up by 200px
    assert len(events) == 1 and isinstance(events[0], Scroll)
    assert abs(events[0].dy - (-200)) <= 8
    assert abs(events[0].dx) <= 8
    # After the scroll, nothing should be reported as ink.
    assert not [e for e in tracker.update(view(page, 600), 3.0) if isinstance(e, InkRegion)]


def test_page_turn_announces_once_then_rereads_page():
    tracker = InkTracker(idle_seconds=1.0)
    tracker.update(view(notebook_page(seed=1), 0), 0.0)
    # A zoom/page-turn animation: several big frame-to-frame jumps.
    events = []
    for i, seed in enumerate([2, 3, 4], start=1):
        events += tracker.update(view(notebook_page(seed=seed), 0), i * 0.1)
    assert [e.reason for e in events if isinstance(e, PageChange)] == ["changing"]
    final = view(notebook_page(seed=4), 0)
    assert tracker.update(final, 0.8) == []  # not settled yet
    settled = tracker.update(final, 2.0)
    assert isinstance(settled[0], PageChange) and settled[0].reason == "settled"
    assert settled[1:] and all(r.kind == "page" for r in settled[1:])
    assert tracker.update(final, 5.0) == []


def test_ignore_regions_mask_toolbars():
    page = view(notebook_page(), 0)
    tracker = InkTracker(idle_seconds=0.5, ignore=[(0.0, 0.0, 1.0, 0.1)])
    tracker.update(page, 0.0)
    toolbar_change = write(page, 400, 40)  # inside the top 10%
    tracker.update(toolbar_change, 0.1)
    assert tracker.update(toolbar_change, 2.0) == []


def test_estimate_shift():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 255, (200, 300)).astype(np.int16)
    b = np.roll(a, (-17, 9), axis=(0, 1))
    assert estimate_shift(a, b) == (-17, 9)


def test_slow_scroll_on_sparse_page_is_not_ink():
    sparse = Image.new("L", (W, 3000), 245)
    d = ImageDraw.Draw(sparse)
    for y in (500, 900, 1300):
        d.line([(300, y), (700, y + 15), (900, y - 10)], fill=20, width=4)
    tracker = InkTracker(idle_seconds=1.0)
    tracker.update(view(sparse, 300), 0.0)
    events = []
    for i, top in enumerate(range(304, 344, 4), start=1):  # 4px per frame: under the sudden threshold
        events += tracker.update(view(sparse, top), i * 0.1)
    events += tracker.update(view(sparse, 340), 3.0)
    assert not [e for e in events if isinstance(e, InkRegion)]


def blank_page() -> Image.Image:
    return Image.new("RGB", (W, H), (245, 245, 240))


def test_line_finished_after_a_pause_is_reread_in_context():
    tracker = InkTracker(idle_seconds=1.0)
    page = blank_page()
    tracker.update(page, 0.0)
    first = write(page, 300, 400)  # "2x + 3 = 1"
    tracker.update(first, 0.5)
    tracker.update(first, 2.0)  # committed
    finished = write(first, 500, 400)  # the trailing "1", written after a pause
    tracker.update(finished, 3.0)
    regions = [e for e in tracker.update(finished, 4.5) if isinstance(e, InkRegion)]
    assert len(regions) == 1
    x0, _, x1, _ = regions[0].bbox
    assert x0 <= 300 and x1 >= 680  # whole line, not just the new stroke


def test_separate_blocks_stay_separate():
    tracker = InkTracker(idle_seconds=1.0)
    page = write(blank_page(), 100, 150)  # existing work far away
    tracker.update(page, 0.0)
    inked = write(page, 700, 700)
    tracker.update(inked, 0.5)
    regions = [e for e in tracker.update(inked, 2.0) if isinstance(e, InkRegion)]
    assert len(regions) == 1 and regions[0].bbox[0] > 500


def test_solid_ui_palette_is_not_ink():
    tracker = InkTracker(idle_seconds=1.0)
    page = blank_page()
    tracker.update(page, 0.0)
    palette = page.copy()
    ImageDraw.Draw(palette).rounded_rectangle([1080, 300, 1160, 800], radius=20, fill=(40, 40, 45))
    tracker.update(palette, 0.5)
    assert tracker.update(palette, 2.0) == []


def test_crops_stay_out_of_top_toolbar_band():
    tracker = InkTracker(idle_seconds=1.0, ignore=[(0.0, 0.0, 1.0, 0.1)])
    page = blank_page()
    tracker.update(page, 0.0)
    inked = write(page, 400, 110)  # just below the band; padding would reach into it
    tracker.update(inked, 0.5)
    regions = [e for e in tracker.update(inked, 2.0) if isinstance(e, InkRegion)]
    assert regions and regions[0].bbox[1] >= 90


def test_page_reread_splits_into_blocks():
    page = write(write(blank_page(), 100, 200), 700, 700)  # two separate problems
    events = InkTracker().update(page, 0.0)
    regions = [e for e in events if isinstance(e, InkRegion)]
    assert [r.kind for r in regions] == ["page", "page"]
    assert regions[0].bbox[2] < regions[1].bbox[0]


def test_writing_cut_off_by_the_screen_edge_is_marked_clipped():
    tracker = InkTracker(idle_seconds=1.0, ignore=[(0.0, 0.0, 1.0, 0.1)])
    page = blank_page()
    tracker.update(page, 0.0)
    cut = page.copy()
    ImageDraw.Draw(cut).line([(400, 70), (450, 130), (520, 100), (600, 140)], fill=(0, 0, 0), width=4)  # runs under the toolbar
    whole = write(page, 400, 500)
    tracker.update(cut, 0.5)
    regions = [e for e in tracker.update(cut, 2.0) if isinstance(e, InkRegion)]
    assert regions and regions[0].clipped
    tracker.update(whole, 2.5)
    regions = [e for e in tracker.update(whole, 4.0) if isinstance(e, InkRegion) and e.bbox[1] > 300]
    assert regions and not regions[0].clipped


def scribble_line(d: ImageDraw.ImageDraw, x: int, y: int, size: float) -> None:
    """A handwriting-like line whose letters are about 30*size px tall."""
    for i in range(6):
        cx = x + int(i * 40 * size)
        d.line([(cx, y), (cx + int(15 * size), y - int(30 * size)), (cx + int(30 * size), y)], fill=(0, 0, 0), width=max(3, int(4 * size)))


def test_zoomed_in_problem_stays_one_block():
    for size in (1.0, 2.5):
        page = blank_page()
        d = ImageDraw.Draw(page)
        for row in range(3):  # three lines of one problem, spaced like normal writing at this zoom
            scribble_line(d, 150, int(250 + row * 70 * size), size)
        regions = [e for e in InkTracker().update(page, 0.0) if isinstance(e, InkRegion)]
        assert len(regions) == 1, f"split into {len(regions)} blocks at zoom {size}"


def test_clipped_reports_which_side():
    page = blank_page()
    d = ImageDraw.Draw(page)
    scribble_line(d, 150, 70, 1.0)  # letters reach up under the top band (y < 45)
    scribble_line(d, 150, 600, 1.0)
    tracker = InkTracker(ignore=[(0.0, 0.0, 1.0, 0.05)])
    regions = [e for e in tracker.update(page, 0.0) if isinstance(e, InkRegion)]
    assert [sorted(r.clipped) for r in regions] == [["top"], []]
