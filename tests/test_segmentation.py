import numpy as np

from psma_seg import segmentation as seg


def blob_image():
    img = np.zeros((200, 100), np.float32)
    yy, xx = np.mgrid[:200, :100]
    img += 100 * np.exp(-((yy - 50) ** 2 + (xx - 40) ** 2) / (2 * 4 ** 2))   # lesion A
    img += 50 * np.exp(-((yy - 150) ** 2 + (xx - 60) ** 2) / (2 * 3 ** 2))   # lesion B
    img += 20 * ((yy > 40) & (yy < 60) & (xx > 70))                           # "organ"
    return img


def test_clip_box():
    assert seg.clip_box((200, 100), 60, 50, 30, 20) == (30, 20, 61, 51)
    assert seg.clip_box((200, 100), -5, -5, 300, 300) == (0, 0, 200, 100)
    assert seg.clip_box((200, 100), -5, 10, -2, 20) is None
    assert seg.clip_box((200, 100), 7, 7, 7, 7) == (7, 7, 8, 8)      # a click = 1 pixel


def test_threshold_in_box():
    img = blob_image()
    box = seg.clip_box(img.shape, 30, 20, 70, 95)      # lesion A and part of the "organ"
    m40 = seg.threshold_in_box(img, box, 40, 1e9)
    m80 = seg.threshold_in_box(img, box, 80, 1e9)
    assert m40[50, 40] and m80[50, 40]
    assert m40.sum() > m80.sum() > 0
    assert not m40[150, 60]                            # outside the rectangle
    assert not m40[:30].any() and not m40[70:].any()
    # a lower limit of 15 also takes the organ (20 counts) -> two blobs; largest_only keeps one
    two = seg.threshold_in_box(img, box, 15, 1e9)
    assert two[50, 80] and two[50, 40]
    one = seg.threshold_in_box(img, box, 15, 1e9, largest_only=True)
    assert one.any() and one.sum() < two.sum()
    assert seg.threshold_in_box(img, box, 40, 30).sum() == 0     # upper < lower
    assert seg.threshold_in_box(img, None, 0, 1e9).sum() == 0
    # single-pixel specks are dropped with min_size
    speck = np.zeros((20, 20)); speck[5, 5] = 1; speck[10:14, 10:14] = 1
    kept = seg.threshold_in_box(speck, (0, 0, 20, 20), 0.5, 2, min_size=4)
    assert not kept[5, 5] and kept[12, 12]


def test_auto_threshold_methods_split_lesion_from_background():
    rng = np.random.default_rng(0)
    img = blob_image() + rng.poisson(2.0, (200, 100))
    box = seg.clip_box(img.shape, 30, 20, 70, 60)
    vals = img[box[0]:box[2], box[1]:box[3]]
    assert seg.auto_threshold(vals, "Manual") is None
    assert seg.auto_threshold(np.full(50, 3.0), "Otsu") is None       # constant image
    for method in seg.THRESHOLD_METHODS[1:]:
        thr = seg.auto_threshold(vals, method)
        assert thr is not None and np.isfinite(thr), method
        assert vals.min() < thr <= vals.max(), method
        m = seg.threshold_in_box(img, box, thr, 1e9)
        assert m[50, 40], method                      # the peak is always foreground
        assert m.sum() < vals.size, method            # and not the whole rectangle


def test_polygon_mask_square():
    m = seg.polygon_mask((50, 50), [(10, 10), (10, 20), (20, 20), (20, 10)])
    assert m[15, 15] and not m[5, 5]
    assert 90 <= m.sum() <= 130


def test_component_at_and_stats():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:20, 10:20] = 1
    mask[60:65, 70:80] = 1
    comp = seg.component_at(mask, 21, 15, search=2)    # just outside lesion 1
    assert comp is not None and comp.sum() == 100
    assert seg.component_at(mask, 40, 40) is None
    img = np.ones((100, 100), np.float32) * 2
    st = seg.lesion_stats(mask, img, (2.0, 2.0))
    assert [d["area_px"] for d in st] == [100, 50]     # sorted head -> feet
    assert st[0]["area_cm2"] == 4.0 and st[0]["total_counts"] == 200
    assert st[1]["bbox_rc"] == [60, 70, 65, 80]


def _blobs():
    yy, xx = np.mgrid[:100, :100]
    small = 100 * np.exp(-((yy - 50) ** 2 + (xx - 40) ** 2) / 50)
    big = 300 * np.exp(-((yy - 50) ** 2 + (xx - 62) ** 2) / 80)
    return small + big


def test_level_trace_follows_count_level():
    from psma_seg.segmentation import level_trace
    yy, xx = np.mgrid[:80, :80]
    img = 100 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 100)
    inner = level_trace(img, 40, 45)          # closer to the centre -> smaller region
    outer = level_trace(img, 40, 50)
    assert inner is not None and outer is not None
    assert inner.sum() < outer.sum() and inner[40, 40] and outer[40, 40]
    assert (inner & ~outer).sum() == 0
    assert level_trace(np.zeros((10, 10)), 5, 5) is None          # no counts
    assert level_trace(img, 0, 0) is None                          # background: too large


def test_grow_from_seeds_separates_touching_hot_spots():
    from psma_seg.segmentation import grow_from_seeds
    img = _blobs()
    fg = np.zeros(img.shape, bool)
    fg[50, 40] = True
    bg = np.zeros(img.shape, bool)
    bg[50, 62] = True
    region = grow_from_seeds(img, fg, bg)
    assert region[50, 40] and not region[50, 62]
    assert 50 < region.sum() < 1500
    assert grow_from_seeds(img, np.zeros_like(fg), bg) is None    # no lesion seed
