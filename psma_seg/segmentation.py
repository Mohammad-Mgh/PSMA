"""Pure numpy/scipy segmentation helpers (no Qt), shared by the tools and the exporter."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

CONN8 = np.ones((3, 3), bool)


# ------------------------------------------------------------------ thresholding
# Automatic threshold methods of 3D Slicer's Segment Editor "Threshold" effect
# (ITK / ImageJ AutoThresholder algorithms), computed on a 256-bin histogram.
THRESHOLD_METHODS = ("Manual", "Otsu", "Huang", "IsoData", "Li", "Maximum Entropy", "Moments",
                     "Kittler-Illingworth", "Renyi Entropy", "Shanbhag", "Triangle", "Yen")
N_BINS = 256
_EPS = 2.220446049250313e-16


def _hist(values: np.ndarray):
    v = np.asarray(values, np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None, None
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return None, None
    h, edges = np.histogram(v, bins=N_BINS, range=(lo, hi))
    return h.astype(np.float64), edges


def _otsu(h):
    i = np.arange(h.size)
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * i)
    m1 = m0[-1] - m0
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0, mu1 = m0 / w0, m1 / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var[~np.isfinite(var)] = -1
    return int(np.argmax(var))


def _huang(h):
    nz = np.flatnonzero(h)
    first, last = int(nz[0]), int(nz[-1])
    if first == last:
        return first
    i = np.arange(h.size, dtype=np.float64)
    s, w = np.cumsum(h), np.cumsum(h * i)
    ts = np.arange(first, last)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = w[ts] / s[ts]
        mu1 = (w[last] - w[ts]) / (s[last] - s[ts])
    c = float(last - first)
    below = i[None, :] <= ts[:, None]                       # (t, i)
    mu = np.where(below, mu0[:, None], mu1[:, None])
    x = 1.0 / (1.0 + np.abs(i[None, :] - mu) / c)
    with np.errstate(invalid="ignore", divide="ignore"):
        ent = -x * np.log(x) - (1 - x) * np.log(1 - x)
    ent = np.where((x > _EPS) & (x < 1 - _EPS), ent, 0.0)
    return int(ts[np.argmin((h[None, :] * ent).sum(1))])


def _isodata(h):
    i = np.arange(h.size, dtype=np.float64)
    t = int(round((h * i).sum() / h.sum()))
    for _ in range(1000):
        lo, hi = h[: t + 1], h[t + 1:]
        if lo.sum() == 0 or hi.sum() == 0:
            break
        m_lo = (lo * i[: t + 1]).sum() / lo.sum()
        m_hi = (hi * i[t + 1:]).sum() / hi.sum()
        new = int((m_lo + m_hi) / 2.0)
        if new == t:
            break
        t = new
    return t


def _li(h):
    i = np.arange(h.size, dtype=np.float64) + 1.0     # strictly positive bin values
    total = h.sum()
    t = (h * i).sum() / total
    for _ in range(1000):
        below = i <= t
        wb, wo = h[below].sum(), h[~below].sum()
        if wb == 0 or wo == 0:
            break
        mb = (h[below] * i[below]).sum() / wb
        mo = (h[~below] * i[~below]).sum() / wo
        new = (mb - mo) / (np.log(mb) - np.log(mo))
        if abs(new - t) < 0.5:
            t = new
            break
        t = new
    return max(0, min(h.size - 1, int(round(t)) - 1))


def _entropy_parts(h):
    norm = h / h.sum()
    p1 = np.cumsum(norm)
    p2 = 1.0 - p1
    first = int(np.argmax(np.abs(p1) >= _EPS))
    last = h.size - 1 - int(np.argmax(np.abs(p2[::-1]) >= _EPS))
    return norm, p1, p2, first, last


def _renyi_term(norm, p1, p2, first, last, alpha):
    """argmax_t of the Renyi entropy sum for alpha in {0.5, 1, 2} (ImageJ)."""
    ts = np.arange(first, last + 1)
    i = np.arange(norm.size)
    below = i[None, :] <= ts[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = np.where(below, norm[None, :] / p1[ts][:, None], 0.0)
        po = np.where(~below, norm[None, :] / p2[ts][:, None], 0.0)
        if alpha == 1:
            eb = -(np.where(pb > 0, pb * np.log(np.where(pb > 0, pb, 1.0)), 0.0)).sum(1)
            eo = -(np.where(po > 0, po * np.log(np.where(po > 0, po, 1.0)), 0.0)).sum(1)
            tot = eb + eo
        else:
            eb, eo = (pb ** alpha).sum(1), (po ** alpha).sum(1)
            prod = eb * eo
            tot = np.where(prod > 0, np.log(np.where(prod > 0, prod, 1.0)), 0.0) / (1.0 - alpha)
    tot[~np.isfinite(tot)] = -np.inf
    return int(ts[np.argmax(tot)])


def _max_entropy(h):
    return _renyi_term(*_entropy_parts(h), alpha=1)


def _renyi(h):
    norm, p1, p2, first, last = _entropy_parts(h)
    t1, t2, t3 = sorted(_renyi_term(norm, p1, p2, first, last, a) for a in (0.5, 1, 2))
    if abs(t1 - t2) <= 5:
        beta = (1, 2, 1) if abs(t2 - t3) <= 5 else (0, 1, 3)
    else:
        beta = (3, 1, 0) if abs(t2 - t3) <= 5 else (1, 2, 1)
    omega = p1[t3] - p1[t1]
    opt = (t1 * (p1[t1] + 0.25 * omega * beta[0]) + 0.25 * t2 * omega * beta[1]
           + t3 * (p2[t3] + 0.25 * omega * beta[2]))
    return int(round(opt))


def _moments(h):
    norm = h / h.sum()
    i = np.arange(h.size, dtype=np.float64)
    m1, m2, m3 = (norm * i).sum(), (norm * i ** 2).sum(), (norm * i ** 3).sum()
    cd = m2 - m1 * m1
    if cd == 0:
        return _otsu(h)
    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (-m3 + m2 * m1) / cd
    disc = c1 * c1 - 4.0 * c0
    if disc < 0:
        return _otsu(h)
    z0, z1 = 0.5 * (-c1 - np.sqrt(disc)), 0.5 * (-c1 + np.sqrt(disc))
    if z1 == z0:
        return _otsu(h)
    p0 = (z1 - m1) / (z1 - z0)
    above = np.flatnonzero(np.cumsum(norm) > p0)
    return int(above[0]) if above.size else h.size - 1


def _kittler(h):
    """Minimum error thresholding (Kittler & Illingworth), ImageJ MinError(I)."""
    i = np.arange(h.size, dtype=np.float64)
    a, b, c = np.cumsum(h), np.cumsum(h * i), np.cumsum(h * i * i)
    fallback = _isodata(h)
    t = fallback
    for _ in range(1000):
        prev = t
        if a[t] == 0 or a[-1] - a[t] == 0:
            return fallback
        with np.errstate(invalid="ignore", divide="ignore"):
            mu, nu = b[t] / a[t], (b[-1] - b[t]) / (a[-1] - a[t])
            p, q = a[t] / a[-1], (a[-1] - a[t]) / a[-1]
            s2 = c[t] / a[t] - mu * mu
            t2 = (c[-1] - c[t]) / (a[-1] - a[t]) - nu * nu
            if s2 <= 0 or t2 <= 0:
                return fallback
            w0 = 1.0 / s2 - 1.0 / t2
            w1 = mu / s2 - nu / t2
            w2 = mu * mu / s2 - nu * nu / t2 + np.log10((s2 * q * q) / (t2 * p * p))
            sq = w1 * w1 - w0 * w2
            if not np.isfinite(sq) or sq < 0 or w0 == 0:
                return fallback
            new = (w1 + np.sqrt(sq)) / w0
        if not np.isfinite(new):
            return fallback
        t = int(np.clip(np.floor(new), 0, h.size - 1))
        if t == prev:
            break
    return t


def _shanbhag(h):
    norm, p1, p2, first, last = _entropy_parts(h)
    ts = np.arange(first, last + 1)
    i = np.arange(norm.size)
    p1_prev = np.concatenate(([0.0], p1[:-1]))
    below = i[None, :] <= ts[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        tb = 0.5 / p1[ts][:, None]
        lb = np.log(np.clip(1.0 - tb * p1_prev[None, :], _EPS, None))
        eb = -(np.where(below, norm[None, :] * lb, 0.0)).sum(1) * tb[:, 0]
        to = 0.5 / p2[ts][:, None]
        lo = np.log(np.clip(1.0 - to * p2[None, :], _EPS, None))
        eo = -(np.where(~below, norm[None, :] * lo, 0.0)).sum(1) * to[:, 0]
        tot = np.abs(eb - eo)
    tot[~np.isfinite(tot)] = np.inf
    return int(ts[np.argmin(tot)])


def _triangle(h):
    nz = np.flatnonzero(h)
    lo, hi = int(nz[0]), int(nz[-1])
    lo = max(0, lo - 1)
    hi = min(h.size - 1, hi + 1)
    peak = int(np.argmax(h))
    flipped = (hi - peak) > (peak - lo)      # search the longer tail: mirror if it is on the right
    if flipped:
        h = h[::-1]
        lo, hi = h.size - 1 - hi, h.size - 1 - lo
        peak = h.size - 1 - peak
    if peak == lo:
        t = lo
    else:
        i = np.arange(lo, peak + 1)
        nx, ny = h[peak] - h[lo], -(peak - lo)
        d = nx * i + ny * h[lo:peak + 1]
        t = int(i[np.argmax(d)])
    t = max(t - 1, 0)
    return h.size - 1 - t if flipped else t


def _yen(h):
    norm = h / h.sum()
    p1 = np.cumsum(norm)
    sq = np.cumsum(norm ** 2)
    p2_sq = sq[-1] - sq
    with np.errstate(invalid="ignore", divide="ignore"):
        crit = -np.log(sq * p2_sq + _EPS) + 2.0 * np.log(p1 * (1.0 - p1) + _EPS)
    crit[~np.isfinite(crit)] = -np.inf
    return int(np.argmax(crit))


_METHOD_FUNCS = {"Otsu": _otsu, "Huang": _huang, "IsoData": _isodata, "Li": _li,
                 "Maximum Entropy": _max_entropy, "Moments": _moments,
                 "Kittler-Illingworth": _kittler, "Renyi Entropy": _renyi,
                 "Shanbhag": _shanbhag, "Triangle": _triangle, "Yen": _yen}


def auto_threshold(values: np.ndarray, method: str) -> float | None:
    """Lower threshold for `method` computed from `values` (e.g. the pixels inside the
    user's rectangle). Pixels >= the returned value are foreground, as in 3D Slicer.
    None for "Manual", for a constant image, or when the method cannot be computed."""
    if method not in _METHOD_FUNCS:
        return None
    h, edges = _hist(values)
    if h is None:
        return None
    k = _METHOD_FUNCS[method](h)
    k = int(np.clip(k, 0, N_BINS - 1))
    return float(edges[k + 1])       # every bin above the split bin is foreground


def clip_box(shape: tuple[int, int], r0: int, c0: int, r1: int, c1: int):
    """Normalised, image-clipped (r0, c0, r1, c1) with exclusive ends, or None if empty."""
    h, w = shape
    rr0, rr1 = sorted((int(r0), int(r1)))
    cc0, cc1 = sorted((int(c0), int(c1)))
    rr0, cc0 = max(0, rr0), max(0, cc0)
    rr1, cc1 = min(h, rr1 + 1), min(w, cc1 + 1)
    if rr0 >= rr1 or cc0 >= cc1:
        return None
    return rr0, cc0, rr1, cc1


def threshold_in_box(img: np.ndarray, box, lower: float, upper: float,
                     min_size: int = 1, largest_only: bool = False) -> np.ndarray:
    """Pixels inside `box` with lower <= value <= upper, as a full-size bool mask.

    Holes are filled and 8-connected blobs smaller than `min_size` pixels are dropped
    (noise specks); with `largest_only` just the biggest blob is kept.
    """
    out = np.zeros(img.shape, bool)
    if box is None or upper < lower:
        return out
    r0, c0, r1, c1 = box
    sub = img[r0:r1, c0:c1]
    m = (sub >= lower) & (sub <= upper)
    if m.any():
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m, structure=CONN8)
        if n > 1 or min_size > 1:
            sizes = ndimage.sum(m, lab, np.arange(1, n + 1))
            keep = sizes >= min_size
            if largest_only:
                keep &= sizes == sizes.max()
            m = np.isin(lab, np.flatnonzero(keep) + 1)
    out[r0:r1, c0:c1] = m
    return out


def polygon_mask(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    """Filled polygon from (row, col) vertices in raw pixel coordinates -> bool mask."""
    h, w = shape
    img = Image.new("1", (w, h), 0)
    if len(points) >= 3:
        # PIL uses pixel-corner coordinates; shift to pixel centres
        ImageDraw.Draw(img).polygon([(c - 0.5, r - 0.5) for r, c in points], fill=1, outline=1)
    return np.array(img, dtype=bool)


def label_lesions(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask > 0, structure=CONN8)


def component_at(mask: np.ndarray, row: int, col: int, search: int = 3) -> np.ndarray | None:
    """Bool mask of the lesion (connected component) at / nearest to (row, col)."""
    lab, n = label_lesions(mask)
    if n == 0:
        return None
    h, w = mask.shape
    best, best_d = 0, None
    for dr in range(-search, search + 1):
        for dc in range(-search, search + 1):
            r, c = row + dr, col + dc
            if 0 <= r < h and 0 <= c < w and lab[r, c]:
                d = dr * dr + dc * dc
                if best_d is None or d < best_d:
                    best, best_d = lab[r, c], d
    return (lab == best) if best else None


def lesion_stats(mask: np.ndarray, image: np.ndarray | None,
                 spacing: tuple[float, float], layers: np.ndarray | None = None,
                 layer_names: list[str] | None = None) -> list[dict]:
    """Per-lesion statistics: area, counts, centroid, bounding box.

    With `layers` (n_windows, H, W) and `layer_names`, each lesion also gets
    "windows": the names of the energy-window masks that overlap it.
    """
    lab, n = label_lesions(mask)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    areas = ndimage.sum(np.ones_like(lab), lab, idx)
    cents = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
    boxes = ndimage.find_objects(lab)
    if image is not None:
        totals = ndimage.sum(image, lab, idx)
        maxima = ndimage.maximum(image, lab, idx)
    px_cm2 = spacing[0] * spacing[1] / 100.0
    present = None
    if layers is not None:
        present = [np.asarray(ndimage.maximum(layer, lab, idx)) > 0 for layer in layers]
    out = []
    for i in range(n):
        sl = boxes[i]
        d = {"id": i + 1,
             "area_px": int(areas[i]),
             "area_cm2": round(float(areas[i]) * px_cm2, 3),
             "centroid_rc": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
             "bbox_rc": [sl[0].start, sl[1].start, sl[0].stop, sl[1].stop]}
        if image is not None:
            d["total_counts"] = float(totals[i])
            d["max_counts"] = float(maxima[i])
            d["mean_counts"] = round(float(totals[i]) / float(areas[i]), 2)
        if present is not None:
            names = layer_names or [str(k) for k in range(len(present))]
            d["windows"] = [names[k] for k in range(len(present)) if present[k][i]]
        out.append(d)
    # sort top-to-bottom (head to feet) for a stable, readable order
    out.sort(key=lambda d: (d["centroid_rc"][0], d["centroid_rc"][1]))
    for i, d in enumerate(out):
        d["id"] = i + 1
    return out


# ---------------------------------------------------------------- 3D Slicer style tools
def level_trace(img: np.ndarray, row: int, col: int,
                max_fraction: float = 0.25) -> np.ndarray | None:
    """Level tracing: region enclosed by the iso-count line through (row, col).

    The 8-connected pixels with value >= the value under the cursor that contain the
    cursor pixel, holes filled. None outside the image, on zero counts, or when the
    region would cover more than `max_fraction` of the image (cursor on background).
    """
    h, w = img.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    level = float(img[row, col])
    if level <= 0:
        return None
    lab, _ = ndimage.label(img >= level, structure=CONN8)
    region = lab == lab[row, col]
    if region.sum() > max_fraction * img.size:
        return None
    return ndimage.binary_fill_holes(region)


def grow_from_seeds(img: np.ndarray, lesion_seeds: np.ndarray, background_seeds: np.ndarray,
                    margin: int = 20, sigma: float = 1.0) -> np.ndarray | None:
    """Grow from seeds (Fast GrowCut, as in 3D Slicer's Segment Editor).

    Every pixel gets the label of the seed it can reach most cheaply, where a step
    between neighbouring pixels costs their count difference, so the lesion /
    background border falls on the count edge. Works in the bounding box of all
    seeds plus `margin` pixels; the box border counts as background (unless painted
    as lesion), so a single lesion stroke is already enough.
    Returns a full-size bool lesion mask, or None without lesion seeds.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    fg = np.asarray(lesion_seeds, bool)
    bg = np.asarray(background_seeds, bool) & ~fg
    if not fg.any():
        return None
    h, w = img.shape
    rows, cols = np.nonzero(fg | bg)
    r0, r1 = max(0, rows.min() - margin), min(h, rows.max() + margin + 1)
    c0, c1 = max(0, cols.min() - margin), min(w, cols.max() + margin + 1)
    sub = np.asarray(img[r0:r1, c0:c1], np.float64)
    if sigma:
        sub = ndimage.gaussian_filter(sub, sigma)
    span = float(sub.max() - sub.min()) or 1.0
    sub = sub / span
    sh, sw = sub.shape
    idx = np.arange(sh * sw).reshape(sh, sw)
    a_list, b_list, w_list = [], [], []
    for (dr, dc), f in (((0, 1), 1.0), ((1, 0), 1.0), ((1, 1), 2 ** 0.5), ((1, -1), 2 ** 0.5)):
        ra = slice(0, sh - dr)
        rb = slice(dr, sh)
        ca = slice(max(0, -dc), sw - max(0, dc))
        cb = slice(max(0, dc), sw - max(0, -dc))
        a_list.append(idx[ra, ca].ravel())
        b_list.append(idx[rb, cb].ravel())
        # small constant step keeps zero-difference edges in the sparse graph
        w_list.append((np.abs(sub[ra, ca] - sub[rb, cb]) + 1e-4).ravel() * f)
    graph = coo_matrix((np.concatenate(w_list), (np.concatenate(a_list), np.concatenate(b_list))),
                       shape=(sh * sw, sh * sw)).tocsr()

    markers = np.zeros((sh, sw), np.int8)
    markers[[0, -1], :] = 2
    markers[:, [0, -1]] = 2
    markers[bg[r0:r1, c0:c1]] = 2
    markers[fg[r0:r1, c0:c1]] = 1
    seeds = np.flatnonzero(markers)
    _, _, src = dijkstra(graph, directed=False, indices=seeds, min_only=True,
                         return_predecessors=True)
    labels = np.where(src >= 0, markers.ravel()[np.clip(src, 0, None)], 0).reshape(sh, sw)
    out = np.zeros((h, w), bool)
    out[r0:r1, c0:c1] = ndimage.binary_fill_holes(labels == 1)
    return out
