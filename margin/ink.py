"""Ink tracker: turn a stream of iPad screen frames into small crops of *new* handwriting.

Runs locally with no AI. For each frame it decides between:
  - nothing happened,
  - the page scrolled/panned (content moved, not new ink),
  - the page is changing (turned, zoomed, switched notebook) -> one PageChange("changing"), then once it
    settles, PageChange("settled") plus a "page" region with all visible ink so the whole page is re-read,
  - ink was added or erased, and the pen has been idle long enough -> emit a region. Regions are expanded
    to the whole block of nearby writing, so a line finished after a pause is re-read in context.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class InkRegion:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in full-resolution frame pixels
    crop: Image.Image
    kind: str  # "added" | "erased" | "page" (whole visible page, after a page change)
    changed_px: int
    # Sides ("top", "bottom", "left", "right") where writing runs off the visible area (e.g. zoomed in);
    # the reading may be missing pieces there. Empty = fully visible.
    clipped: frozenset = frozenset()


@dataclass
class Scroll:
    dx: int  # full-resolution pixels the content moved
    dy: int


@dataclass
class PageChange:
    reason: str


@dataclass
class InkTracker:
    idle_seconds: float = 1.5
    scale: float = 0.25  # work on a downscaled copy for speed
    diff_threshold: int = 40  # grayscale delta that counts as a change
    min_changed_px: int = 12  # at working scale; smaller blobs are noise
    merge_px: int = 6  # dilate by this many working-scale pixels to merge strokes into lines
    pad_px: int = 24  # padding around emitted crops, full resolution
    sudden_fraction: float = 0.02  # frame-to-frame change above this fraction -> scroll or page change
    scroll_residual: float = 0.15  # after aligning, at most this share of the change may remain to call it a scroll
    ink_dark: int = 110  # full-res grayscale below this is ink (grid/ruling lines are lighter)
    context_px: int = 12  # working-scale gap that still counts as the same block (if writing size is unknown)
    context_per_height: float = 1.5  # otherwise: the gap scales with handwriting size (zooming changes it)
    ui_dark_fraction: float = 0.3  # crops this solid-dark are app UI (palettes, toolbars), not handwriting
    ignore: list[tuple[float, float, float, float]] = field(default_factory=list)  # normalized x0,y0,x1,y1

    _baseline: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_activity: float = field(default=0.0, init=False)
    _mask: np.ndarray | None = field(default=None, init=False, repr=False)
    _unsettled: bool = field(default=False, init=False)
    _bounds: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), init=False)  # visible area, working scale

    def update(self, frame: Image.Image, t: float) -> list[InkRegion | Scroll | PageChange]:
        small = self._prepare(frame)
        if self._baseline is None or self._baseline.shape != small.shape:
            self._rebase(small, t)
            return [PageChange("start"), *self._page_region(frame)]

        # Handwriting adds ink a little at a time; scrolls, zooms and page turns change a lot at once.
        jump = int(self._changed(small, self._prev).sum())
        self._prev = small
        if jump >= self.min_changed_px:
            self._last_activity = t
        if jump > self.sudden_fraction * small.size:
            return self._explain_sudden_change(small, t)

        if self._unsettled:
            if t - self._last_activity < self.idle_seconds:
                return []
            self._unsettled = False
            self._baseline = small
            return [PageChange("settled"), *self._page_region(frame)]

        changed = self._changed(small, self._baseline)
        if changed.sum() < self.min_changed_px or t - self._last_activity < self.idle_seconds:
            return []

        # A slow scroll on a sparse page can sneak under the sudden-change test; check before committing.
        shift = estimate_shift(self._baseline, small)
        if shift is not None and self._is_scroll(small, shift):
            self._rebase(small, t)
            return [self._scroll_event(shift)]

        regions = self._regions(changed, small, frame)
        self._baseline = small
        return regions

    # -- internals -------------------------------------------------------------

    def _prepare(self, frame: Image.Image) -> np.ndarray:
        w, h = frame.size
        small = frame.convert("L").resize((max(1, int(w * self.scale)), max(1, int(h * self.scale))), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.int16)
        if self._mask is None or self._mask.shape != arr.shape:
            self._mask = np.ones(arr.shape, dtype=bool)
            sh, sw = arr.shape
            for x0, y0, x1, y1 in self.ignore:
                self._mask[int(y0 * sh) : int(y1 * sh), int(x0 * sw) : int(x1 * sw)] = False
            ys, xs = np.nonzero(self._mask)
            self._bounds = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else (0, 0, 0, 0)
        return arr

    def _changed(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Strip solid UI before masking, so a palette half-inside an ignore band is still seen as solid.
        return strip_solid(np.abs(a - b) > self.diff_threshold) & self._mask

    def _explain_sudden_change(self, small: np.ndarray, t: float) -> list[Scroll | PageChange]:
        shift = estimate_shift(self._baseline, small)
        scrolled = shift is not None and self._is_scroll(small, shift)
        self._rebase(small, t)
        if scrolled:
            return [self._scroll_event(shift)]
        # Zooms and page turns animate over several frames: announce once, re-read when settled.
        already = self._unsettled
        self._unsettled = True
        return [] if already else [PageChange("changing")]

    def _scroll_event(self, shift: tuple[int, int]) -> Scroll:
        dy, dx = shift
        return Scroll(dx=round(dx / self.scale), dy=round(dy / self.scale))

    def _rebase(self, small: np.ndarray, t: float) -> None:
        self._baseline = small
        self._prev = small
        self._last_activity = t

    def _is_scroll(self, small: np.ndarray, shift: tuple[int, int]) -> bool:
        dy, dx = shift
        if dy == 0 and dx == 0:
            return False
        unaligned = int(self._changed(small, self._baseline).sum())
        aligned = np.roll(self._baseline, (dy, dx), axis=(0, 1))
        diff = self._changed(small, aligned)
        # Ignore the band that scrolled in from off-screen (and the wrapped-around band).
        h, w = diff.shape
        core = diff[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)]
        return unaligned > 0 and core.size > 0 and core.sum() <= self.scroll_residual * unaligned

    def _ink(self, frame: Image.Image, shape: tuple[int, int]) -> np.ndarray:
        """Ink mask at working scale: any full-res ink pixel marks its cell (thin strokes survive downscaling)."""
        f = round(1 / self.scale)
        h, w = shape
        gray = np.asarray(frame.convert("L"))[: h * f, : w * f]
        if gray.shape != (h * f, w * f):
            gray = np.pad(gray, ((0, h * f - gray.shape[0]), (0, w * f - gray.shape[1])), constant_values=255)
        return strip_solid((gray < self.ink_dark).reshape(h, f, w, f).any(axis=(1, 3))) & self._mask

    def _regions(self, changed: np.ndarray, small: np.ndarray, frame: Image.Image) -> list[InkRegion]:
        # Group the change into blocks of nearby writing, so the model always sees whole lines in context.
        ink = self._ink(frame, small.shape)
        blocks, _ = ndimage.label(ndimage.binary_dilation(ink | changed, iterations=self._context(ink)))
        block_slices = ndimage.find_objects(blocks)
        blobs, _ = ndimage.label(ndimage.binary_dilation(changed, iterations=self.merge_px))

        groups: dict[frozenset, dict] = {}
        for i, slc in enumerate(ndimage.find_objects(blobs), start=1):
            if slc is None:
                continue
            blob = changed & (blobs == i)
            count = int(blob.sum())
            if count < self.min_changed_px:
                continue
            ids = frozenset(int(b) for b in np.unique(blocks[blob]) if b)
            g = groups.setdefault(ids, {"slices": [slc], "count": 0, "darker": 0})
            g["slices"] += [block_slices[b - 1] for b in ids if block_slices[b - 1] is not None]
            g["count"] += count
            g["darker"] += int((small < self._baseline)[blob].sum())

        regions = []
        for g in groups.values():
            y0 = min(sl[0].start for sl in g["slices"])
            y1 = max(sl[0].stop for sl in g["slices"])
            x0 = min(sl[1].start for sl in g["slices"])
            x1 = max(sl[1].stop for sl in g["slices"])
            kind = "added" if g["darker"] >= g["count"] / 2 else "erased"
            region = self._make_region(frame, (x0, y0, x1, y1), kind, g["count"], self._clipped(ink, (x0, y0, x1, y1)))
            if region:
                regions.append(region)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _page_region(self, frame: Image.Image) -> list[InkRegion]:
        """One "page" region per block of writing on the visible page (after start / zoom / page turn)."""
        small_shape = (max(1, int(frame.size[1] * self.scale)), max(1, int(frame.size[0] * self.scale)))
        if self._mask is None or self._mask.shape != small_shape:
            self._prepare(frame)
        ink = self._ink(frame, small_shape)
        blocks, _ = ndimage.label(ndimage.binary_dilation(ink, iterations=self._context(ink)))
        regions = []
        for i, slc in enumerate(ndimage.find_objects(blocks), start=1):
            if slc is None:
                continue
            count = int((ink[slc] & (blocks[slc] == i)).sum())
            if count < self.min_changed_px:
                continue
            ys, xs = slc
            box = (xs.start, ys.start, xs.stop, ys.stop)
            region = self._make_region(frame, box, "page", count, self._clipped(ink & (blocks == i), box))
            if region:
                regions.append(region)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        return regions

    def _context(self, ink: np.ndarray) -> int:
        """How far apart strokes can be and still belong to one block: proportional to the writing's size,
        so a zoomed-in problem (bigger letters, bigger line gaps) isn't split into pieces."""
        labels, n = ndimage.label(ink, structure=np.ones((3, 3)))
        if n == 0:
            return self.context_px
        heights = [sl[0].stop - sl[0].start for sl in ndimage.find_objects(labels) if sl is not None]
        heights = [h for h in heights if h >= 3]  # ignore dots, "=" bars, noise
        if not heights:
            return self.context_px
        return int(np.clip(round(self.context_per_height * float(np.median(heights))), 6, 40))

    def _clipped(self, ink: np.ndarray, box: tuple[int, int, int, int]) -> frozenset:
        """Which sides of the visible area the ink inside `box` touches."""
        x0, y0, x1, y1 = box
        ys, xs = np.nonzero(ink[y0:y1, x0:x1])
        if len(xs) == 0:
            return frozenset()
        ys, xs = ys + y0, xs + x0
        bx0, by0, bx1, by1 = self._bounds
        sides = {
            "top": ys.min() <= by0 + 1,
            "bottom": ys.max() >= by1 - 1,
            "left": xs.min() <= bx0 + 1,
            "right": xs.max() >= bx1 - 1,
        }
        return frozenset(side for side, hit in sides.items() if hit)

    def _make_region(
        self, frame: Image.Image, small_box: tuple[int, int, int, int], kind: str, count: int, clipped: frozenset = frozenset()
    ) -> InkRegion | None:
        fw, fh = frame.size
        x0, y0, x1, y1 = small_box
        box = [
            max(0, int(x0 / self.scale) - self.pad_px),
            max(0, int(y0 / self.scale) - self.pad_px),
            min(fw, int(x1 / self.scale) + self.pad_px),
            min(fh, int(y1 / self.scale) + self.pad_px),
        ]
        # Keep crops out of edge-anchored ignore bands (toolbars, gutters) that the padding could pull in.
        for ix0, iy0, ix1, iy1 in self.ignore:
            if ix0 <= 0.001 and ix1 >= 0.999:
                if iy0 <= 0.001:
                    box[1] = max(box[1], int(iy1 * fh))
                elif iy1 >= 0.999:
                    box[3] = min(box[3], int(iy0 * fh))
            if iy0 <= 0.001 and iy1 >= 0.999:
                if ix0 <= 0.001:
                    box[0] = max(box[0], int(ix1 * fw))
                elif ix1 >= 0.999:
                    box[2] = min(box[2], int(ix0 * fw))
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        crop = frame.crop(tuple(box))
        # Handwriting is thin strokes; a crop that is largely solid dark is a palette or toolbar.
        if kind != "erased" and (np.asarray(crop.convert("L")) < 80).mean() > self.ui_dark_fraction:
            return None
        return InkRegion(bbox=tuple(box), crop=crop, kind=kind, changed_px=count, clipped=clipped)


def strip_solid(mask: np.ndarray, size: int = 2) -> np.ndarray:
    """Remove solid blobs (palettes, toolbars, popovers) but keep thin pen strokes.

    An opening erases anything thinner than ~2*size+1 working-scale pixels, so what survives is solid UI.
    """
    solid = ndimage.binary_opening(mask, iterations=size)
    if not solid.any():
        return mask
    return mask & ~ndimage.binary_dilation(solid, iterations=size)


def crop_to_ink(img: Image.Image, dark: int = 110, pad: int = 40) -> Image.Image:
    """Crop a whole page down to its handwriting (pixels darker than faint grid/ruling lines)."""
    gray = np.asarray(img.convert("L"))
    ys, xs = np.nonzero(gray < dark)
    if len(xs) == 0:
        return img
    w, h = img.size
    return img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + pad), min(h, ys.max() + pad)))


def estimate_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int] | None:
    """Phase correlation: return (dy, dx) such that rolling `a` by it best matches `b`."""
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    if not mag.any():
        return None
    corr = np.fft.ifft2(cross / (mag + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)
