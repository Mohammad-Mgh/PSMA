import numpy as np

from psma_seg import segmentation as seg


def blob_image():
    img = np.zeros((200, 100), np.float32)
    yy, xx = np.mgrid[:200, :100]
    img += 100 * np.exp(-((yy - 50) ** 2 + (xx - 40) ** 2) / (2 * 4 ** 2))   # lesion A
    img += 50 * np.exp(-((yy - 150) ** 2 + (xx - 60) ** 2) / (2 * 3 ** 2))   # lesion B
    img += 20 * ((yy > 40) & (yy < 60) & (xx > 70))                           # "organ"
    return img


def test_find_peak():
    img = blob_image()
    assert seg.find_peak(img, 52, 42, search=3) == (50, 40)


def test_region_grow_threshold_and_radius():
    img = blob_image()
    r40 = seg.region_grow(img, 51, 41, 40, max_radius=30)
    r80 = seg.region_grow(img, 51, 41, 80, max_radius=30)
    assert r40[50, 40] and r80[50, 40]
    assert r40.sum() > r80.sum() > 0
    assert not r40[150, 60]                     # other lesion not included
    small = seg.region_grow(img, 50, 40, 5, max_radius=3)
    assert small.sum() <= np.pi * 3 ** 2 + 10   # limited by radius
    assert seg.region_grow(img, -5, 40, 40, 10) is None
    assert seg.region_grow(np.zeros((10, 10)), 5, 5, 40, 3) is None


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
