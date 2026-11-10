"""PDF report of one whole-body scan: images with lesion outlines + lesion tables.

Pages
    1 ...   one page per energy-window choice (Sum of all windows, then each window):
            anterior + posterior side by side (display orientation), lesion outlines
            with the lesion numbers of the tables
    then    summary totals, lesion list (Sum), one lesion table per energy window,
            notes, status and signature lines

The report is a *clinical* document: its header holds the patient name, ID and study
date. It is written next to the DICOM file, never into the anonymous dataset folder.

Drawn with Qt's own PDF writer (QPdfWriter + QPainter), no extra dependency.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QMarginsF, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QFontMetricsF, QImage, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen
from scipy import ndimage

from . import __version__, config
from .canvas import render_gray
from .dicom_io import SUM, WholeBodyStudy
from .segmentation import lesion_stats

DPI = 300
UPSCALE = 3                       # image pixels are enlarged before drawing the outline
RED = config.MASK_COLOR
BLUE = config.PARTIAL_MASK_COLOR
VIEW_NAMES = {"ant": "Anterior", "post": "Posterior"}


# ---------------------------------------------------------------- display settings
@dataclass
class Display:
    """Levels relative to the auto level of each image set, so every energy window
    looks as the current one does on screen."""
    rel_min: float = 0.0
    rel_max: float = 1.0
    gamma: float = 1.0
    inverted: bool = True


def auto_level(images: list[np.ndarray]) -> float:
    """Upper auto-contrast level of a set of images (as the app's Auto contrast)."""
    px = np.concatenate([np.asarray(a, float).ravel() for a in images])
    body = px[px > 0]
    v = float(np.percentile(body, config.AUTO_MAX_PERCENTILE)) if body.size else float(px.max())
    return max(v, 1.0)


# ---------------------------------------------------------------- small helpers
def person_name(dicom_name: str) -> str:
    """'DOE^JOHN' -> 'DOE, JOHN'."""
    parts = [p.strip() for p in str(dicom_name).split("^") if p.strip()]
    return ", ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")


def nice_date(d: str) -> str:
    try:
        return dt.datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return d or ""


def hours_post_injection(series_description: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr|hrs|hour|hours)", series_description or "", re.I)
    return float(m.group(1)) if m else None


def default_report_path(study: WholeBodyStudy) -> Path:
    """<DICOM folder>/<NAME>_<date>_<hours>h_PSMA_report.pdf"""
    m = study.meta
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(m.get("PatientName", "")).replace("^", "_")).strip("_")
    h = hours_post_injection(m.get("SeriesDescription", ""))
    parts = [name or "patient", m.get("StudyDate", "") or "", f"{h:g}h" if h is not None else ""]
    return Path(study.path).parent / ("_".join(p for p in parts if p) + "_PSMA_report.pdf")


def _fmt(x, nd=0) -> str:
    if x is None:
        return ""
    return f"{x:,.{nd}f}"


# ---------------------------------------------------------------- image rendering
def render_view(img: np.ndarray, mask: np.ndarray, partial: np.ndarray | None,
                levels: tuple[float, float], disp: Display, mirror: bool) -> QImage:
    """Gray image (display orientation) with red outlines (blue = partial), enlarged
    UPSCALE times so the outline is thin."""
    g = render_gray(img.astype(np.float32), levels[0], levels[1], disp.gamma, disp.inverted)
    k = UPSCALE
    g = np.repeat(np.repeat(g, k, 0), k, 1)
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    m = np.repeat(np.repeat(mask > 0, k, 0), k, 1)
    if m.any():
        edge = m & ~ndimage.binary_erosion(m, iterations=2, border_value=0)
        rgb[edge] = RED
        if partial is not None and partial.any():
            p = np.repeat(np.repeat(partial > 0, k, 0), k, 1)
            rgb[edge & p] = BLUE
    if mirror:
        rgb = rgb[:, ::-1]
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ---------------------------------------------------------------- report data
@dataclass
class Table:
    title: str
    header: list[str]
    rows: list[list[str]]
    widths: list[float]           # relative column widths
    note: str = ""


class Report:
    def __init__(self, study: WholeBodyStudy, layers: dict[str, np.ndarray], disp: Display,
                 annotator: str = "", status: str = "", notes: str = "", unsaved: bool = False):
        self.study, self.disp = study, disp
        self.annotator, self.status, self.notes, self.unsaved = annotator, status, notes, unsaved
        self.views = study.views
        self.names = list(study.energy_windows)
        n_ew = len(self.names)
        self.layers = {v: (np.repeat(np.asarray(layers[v])[None], n_ew, 0)
                           if np.asarray(layers[v]).ndim == 2 else np.asarray(layers[v])) > 0
                       for v in self.views}
        sp = study.pixel_spacing
        # lesion statistics, numbered as in the app's lesion table
        self.union = {v: self.layers[v].any(0) for v in self.views}
        self.partial = {v: self.union[v] & ~self.layers[v].all(0) for v in self.views}
        self.stats = {SUM: {v: lesion_stats(self.union[v].astype(np.uint8), study.image(v, SUM),
                                            sp, self.layers[v].astype(np.uint8), self.names)
                            for v in self.views}}
        for e in range(n_ew):
            self.stats[e] = {v: lesion_stats(self.layers[v][e].astype(np.uint8), study.image(v, e), sp)
                             for v in self.views}

    # ------------------------------------------------------------ pages of images
    def image_pages(self) -> list[dict]:
        pages = []
        for ew in [SUM, *range(len(self.names))]:
            imgs = {v: self.study.image(v, ew) for v in self.views}
            base = auto_level(list(imgs.values()))
            levels = (self.disp.rel_min * base, max(self.disp.rel_max * base, self.disp.rel_min * base + 1e-3))
            views = []
            for v in self.views:
                mask = self.union[v] if ew == SUM else self.layers[v][ew]
                partial = self.partial[v] if ew == SUM else None
                views.append(dict(view=v, qimage=render_view(imgs[v], mask, partial, levels, self.disp,
                                                             self.study.mirror.get(v, False)),
                                  stats=self.stats[ew][v], mirror=self.study.mirror.get(v, False)))
            title = "Sum of all energy windows" if ew == SUM else f"Energy window: {self.names[ew]}"
            pages.append(dict(title=title, views=views, sum=(ew == SUM), levels=levels))
        return pages

    # ------------------------------------------------------------ tables
    def tables(self) -> list[Table]:
        out = []
        rows = []
        for ew in [SUM, *range(len(self.names))]:
            for v in self.views:
                st = self.stats[ew][v]
                rows.append(["Sum" if ew == SUM else self.names[ew], VIEW_NAMES.get(v, v), str(len(st)),
                             _fmt(sum(s["area_cm2"] for s in st), 2),
                             _fmt(sum(s.get("total_counts", 0) for s in st)),
                             _fmt(max((s.get("max_counts", 0) for s in st), default=0))])
        out.append(Table("Summary (whole-body tumour burden)",
                         ["Energy window", "View", "Lesions", "Total area cm²", "Total counts",
                          "Max counts"], rows, [2.2, 1.3, 1, 1.5, 1.6, 1.3]))
        cols = ["View", "#", "Area cm²", "Area px", "Max counts", "Total counts", "Mean counts",
                "Centre (row, col)"]
        widths = [1.2, 0.5, 1.0, 0.9, 1.1, 1.3, 1.1, 1.5]

        def lesion_rows(ew, with_windows):
            r = []
            for v in self.views:
                for s in self.stats[ew][v]:
                    row = [VIEW_NAMES.get(v, v), str(s["id"]), _fmt(s["area_cm2"], 2), str(s["area_px"]),
                           _fmt(s.get("max_counts")), _fmt(s.get("total_counts")),
                           _fmt(s.get("mean_counts"), 1),
                           f"{s['centroid_rc'][0]:.0f}, {s['centroid_rc'][1]:.0f}"]
                    if with_windows:
                        w = s.get("windows", self.names)
                        row.append("all" if len(w) == len(self.names) else ", ".join(w))
                    r.append(row)
            return r

        multi = len(self.names) > 1
        out.append(Table("Lesions — sum of all energy windows",
                         cols + (["Windows"] if multi else []), lesion_rows(SUM, multi),
                         widths + ([1.6] if multi else []),
                         note=("Counts of the summed image. Windows: energy windows in which the "
                               "lesion is segmented (blue outline = only some windows)." if multi else "")))
        if multi:
            for e, name in enumerate(self.names):
                out.append(Table(f"Lesions — energy window {name}", cols, lesion_rows(e, False), widths,
                                 note=f"Counts of the {name} image."))
        return out

    # ------------------------------------------------------------ writing
    def write(self, path: str | Path) -> Path:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        pdf = QPdfWriter(str(tmp))
        pdf.setResolution(DPI)
        pdf.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait,
                                      QMarginsF(12, 12, 12, 12), QPageLayout.Millimeter))
        m = self.study.meta
        pdf.setTitle(f"PSMA whole-body report {person_name(m.get('PatientName', ''))}")
        pdf.setCreator(f"PSMA-Seg {__version__}")
        p = QPainter()
        if not p.begin(pdf):
            raise OSError(f"Cannot write {path}")
        try:
            _Writer(p, pdf, self).run()
        finally:
            p.end()
        tmp.replace(path)
        return path


def write_report(path, study, layers, disp: Display, **kw) -> Path:
    return Report(study, layers, disp, **kw).write(path)


# ---------------------------------------------------------------- page drawing
class _Writer:
    def __init__(self, p: QPainter, pdf: QPdfWriter, rep: Report):
        self.p, self.pdf, self.rep = p, pdf, rep
        r = pdf.pageLayout().paintRectPixels(DPI)
        self.W, self.H = r.width(), r.height()
        self.mm = DPI / 25.4
        self.generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # fonts ---------------------------------------------------------
    def font(self, pt, bold=False):
        f = QFont("Segoe UI")
        f.setPointSizeF(pt)
        f.setBold(bold)
        self.p.setFont(f)
        return QFontMetricsF(f, self.pdf)

    def text(self, x, y, w, h, s, pt=9, bold=False, align=Qt.AlignLeft | Qt.AlignVCenter,
             color=QColor(0, 0, 0)):
        self.font(pt, bold)
        self.p.setPen(color)
        self.p.drawText(QRectF(x, y, w, h), int(align), s)

    # header / footer -----------------------------------------------
    def header(self, first: bool) -> float:
        """Draw the page header; returns the y below it."""
        rep, m, mm = self.rep, self.rep.study.meta, self.mm
        y = 0.0
        if first:
            self.text(0, y, self.W, 9 * mm, "PSMA whole-body scan — lesion report", 15, True)
            y += 10 * mm
            h = hours_post_injection(m.get("SeriesDescription", ""))
            age = re.sub(r"^0+", "", m.get("PatientAge", "")).replace("Y", " y")
            left = [("Patient", person_name(m.get("PatientName", ""))),
                    ("Patient ID", m.get("PatientID", "")),
                    ("Sex / age", " / ".join(x for x in (m.get("PatientSex", ""), age) if x)),
                    ("Study date", nice_date(m.get("StudyDate", "")))]
            right = [("Series", " ".join(m.get("SeriesDescription", "").split())),
                     ("Time after injection", f"{h:g} h" if h is not None else "—"),
                     ("Scanner", f"{m.get('Manufacturer', '')} {m.get('ManufacturerModelName', '')}".strip()),
                     ("Energy windows", ", ".join(rep.names))]
            row_h = 5.2 * mm
            for i, ((k1, v1), (k2, v2)) in enumerate(zip(left, right)):
                yy = y + i * row_h
                self.text(0, yy, 28 * mm, row_h, k1, 9, True)
                self.text(28 * mm, yy, self.W / 2 - 30 * mm, row_h, v1, 9)
                self.text(self.W / 2, yy, 38 * mm, row_h, k2, 9, True)
                self.text(self.W / 2 + 38 * mm, yy, self.W / 2 - 38 * mm, row_h, v2, 9)
            y += 4 * row_h + 1.5 * mm
            line = f"Annotator: {rep.annotator or '—'}    Status: {rep.status or '—'}    Case ID: {rep.study.case_id}"
            self.text(0, y, self.W, row_h, line, 8.5, color=QColor(60, 60, 60))
            y += row_h
            if rep.unsaved:
                self.text(0, y, self.W, row_h, "Segmentation NOT SAVED — this report shows unsaved changes.",
                          9, True, color=QColor(200, 0, 0))
                y += row_h
        else:
            s = (f"{person_name(m.get('PatientName', ''))}   ·   ID {m.get('PatientID', '')}   ·   "
                 f"{nice_date(m.get('StudyDate', ''))}   ·   {' '.join(m.get('SeriesDescription', '').split())}")
            self.text(0, y, self.W, 6 * mm, s, 9, True)
            y += 6.5 * mm
        self.p.setPen(QPen(QColor(150, 150, 150), 0.3 * mm))
        self.p.drawLine(QPointF(0, y), QPointF(self.W, y))
        return y + 2.5 * mm

    def footer(self, page, total):
        mm = self.mm
        y = self.H - 5 * mm
        self.text(0, y, self.W, 5 * mm,
                  f"PSMA-Seg {__version__} · generated {self.generated} · not a substitute for the "
                  "physician's reading", 7, color=QColor(110, 110, 110))
        self.text(0, y, self.W, 5 * mm, f"Page {page} / {total}", 7,
                  align=Qt.AlignRight | Qt.AlignVCenter, color=QColor(110, 110, 110))

    # image page ----------------------------------------------------
    def image_page(self, pg: dict, top: float):
        mm, p = self.mm, self.p
        self.text(0, top, self.W, 7 * mm, pg["title"], 12, True)
        top += 8 * mm
        legend_h = 6 * mm
        avail_h = self.H - top - legend_h - 9 * mm - 6 * mm
        n = len(pg["views"])
        gap = 8 * mm
        qi = pg["views"][0]["qimage"]
        aspect = qi.width() / qi.height()
        img_h = avail_h
        img_w = img_h * aspect
        if n * img_w + (n - 1) * gap > self.W:
            img_w = (self.W - (n - 1) * gap) / n
            img_h = img_w / aspect
        x0 = (self.W - (n * img_w + (n - 1) * gap)) / 2
        for i, v in enumerate(pg["views"]):
            x = x0 + i * (img_w + gap)
            self.text(x, top, img_w, 6 * mm, VIEW_NAMES.get(v["view"], v["view"]) +
                      ("  (mirrored)" if v["mirror"] else ""), 10, True, align=Qt.AlignCenter)
            rect = QRectF(x, top + 6 * mm, img_w, img_h)
            p.drawImage(rect, v["qimage"])
            p.setPen(QPen(QColor(120, 120, 120), 0.2 * mm))
            p.drawRect(rect)
            self.labels(v, rect)
            self.text(x, rect.bottom(), img_w / 2, 5 * mm, "R" if not v["mirror"] else "L", 8,
                      True, color=QColor(90, 90, 90))
            self.text(x + img_w / 2, rect.bottom(), img_w / 2, 5 * mm, "L" if not v["mirror"] else "R",
                      8, True, align=Qt.AlignRight | Qt.AlignVCenter, color=QColor(90, 90, 90))
        y = top + 6 * mm + img_h + 6 * mm
        lv = pg["levels"]
        legend = (f"Red outline: lesion" + ("; blue: lesion in only some energy windows" if pg["sum"]
                                             and len(self.rep.names) > 1 else "") +
                  f".  Numbers = lesion # in the tables.  Display {lv[0]:.0f}–{lv[1]:.0f} counts, "
                  f"gamma {self.rep.disp.gamma:.2f}{', inverted' if self.rep.disp.inverted else ''}.")
        self.text(0, y, self.W, legend_h, legend, 8, align=Qt.AlignCenter, color=QColor(60, 60, 60))

    def labels(self, v: dict, rect: QRectF):
        """Lesion numbers next to each lesion (to the right, display orientation)."""
        p, mm = self.p, self.mm
        h, w = self.rep.study.shape
        sx, sy = rect.width() / w, rect.height() / h
        fm = self.font(7, True)
        for s in v["stats"]:
            r0, c0, r1, c1 = s["bbox_rc"]
            if v["mirror"]:
                c0, c1 = w - c1, w - c0
            label = str(s["id"])
            tw, th = fm.horizontalAdvance(label) + 1.2 * mm, fm.height()
            x = rect.left() + c1 * sx + 0.8 * mm
            if x + tw > rect.right():
                x = rect.left() + c0 * sx - 0.8 * mm - tw
            y = rect.top() + (r0 + r1) / 2 * sy - th / 2
            box = QRectF(x, y, tw, th)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawRoundedRect(box, 0.6 * mm, 0.6 * mm)
            p.setBrush(Qt.NoBrush)
            p.setPen(QColor(*RED))
            p.drawText(box, int(Qt.AlignCenter), label)

    # table pages ---------------------------------------------------
    def flow_tables(self, top_first: float, top_other: float):
        """Split tables + notes into pages -> list of lists of (kind, item, rows slice)."""
        mm = self.mm
        bottom = self.H - 9 * mm
        self.row_h = 5.0 * mm
        self.title_h = 8 * mm
        pages, cur, y = [], [], top_first
        for t in self.rep.tables():
            rows = t.rows or [None]               # None -> "no lesions" line
            i = 0
            while i < len(rows):
                need = self.title_h + 2 * self.row_h
                if y + need > bottom and cur:
                    pages.append(cur)
                    cur, y = [], top_other
                fit = int((bottom - y - self.title_h - self.row_h - (4 * mm if t.note else 0)) // self.row_h)
                fit = max(fit, 1)
                chunk = rows[i:i + fit]
                cur.append(("table", t, chunk, i > 0))
                y += self.title_h + self.row_h * (1 + len(chunk)) + (4 * mm if t.note else 0) + 4 * mm
                i += fit
        sign_h = 60 * mm
        if y + sign_h > bottom and cur:
            pages.append(cur)
            cur = []
        cur.append(("sign", None, None, False))
        pages.append(cur)
        return pages

    def draw_table(self, t: Table, rows, cont: bool, y: float) -> float:
        p, mm = self.p, self.mm
        self.text(0, y, self.W, self.title_h, t.title + (" (continued)" if cont else ""), 11, True)
        y += self.title_h
        tot = sum(t.widths)
        xs = [0.0]
        for wdt in t.widths:
            xs.append(xs[-1] + wdt / tot * self.W)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(230, 233, 240))
        p.drawRect(QRectF(0, y, self.W, self.row_h))
        p.setBrush(Qt.NoBrush)
        aligns = [Qt.AlignLeft if j == 0 or hd in ("View", "Windows", "Energy window")
                  else Qt.AlignRight for j, hd in enumerate(t.header)]
        for j, hd in enumerate(t.header):
            self.text(xs[j] + 1 * mm, y, xs[j + 1] - xs[j] - 2 * mm, self.row_h, hd, 8, True,
                      align=aligns[j] | Qt.AlignVCenter)
        y += self.row_h
        for k, row in enumerate(rows):
            if row is None:
                self.text(1 * mm, y, self.W, self.row_h, "No lesions segmented.", 8.5,
                          color=QColor(80, 80, 80))
            else:
                if k % 2:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(246, 247, 250))
                    p.drawRect(QRectF(0, y, self.W, self.row_h))
                    p.setBrush(Qt.NoBrush)
                for j, cell in enumerate(row):
                    self.text(xs[j] + 1 * mm, y, xs[j + 1] - xs[j] - 2 * mm, self.row_h, cell, 8.5,
                              align=aligns[j] | Qt.AlignVCenter)
            y += self.row_h
        p.setPen(QPen(QColor(170, 170, 170), 0.2 * mm))
        p.drawLine(QPointF(0, y), QPointF(self.W, y))
        if t.note:
            self.text(0, y, self.W, 4 * mm, t.note, 7.5, color=QColor(90, 90, 90))
            y += 4 * mm
        return y + 4 * mm

    def draw_sign(self, y: float):
        mm, rep = self.mm, self.rep
        self.text(0, y, self.W, self.title_h, "Notes", 11, True)
        y += self.title_h
        self.font(9)
        self.p.setPen(QColor(0, 0, 0))
        notes_rect = QRectF(0, y, self.W, 22 * mm)
        self.p.drawText(notes_rect, int(Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap),
                        rep.notes or "—")
        y += 24 * mm
        self.text(0, y, self.W, 6 * mm,
                  f"Status: {rep.status or '—'}      Segmented by: {rep.annotator or '—'}", 9)
        y += 14 * mm
        self.p.setPen(QPen(QColor(0, 0, 0), 0.25 * mm))
        third = self.W / 3
        for i, lab in enumerate(("Reviewed by", "Signature", "Date")):
            x = i * third
            self.p.drawLine(QPointF(x + 2 * mm, y), QPointF(x + third - 6 * mm, y))
            self.text(x + 2 * mm, y + 1 * mm, third, 5 * mm, lab, 8, color=QColor(80, 80, 80))

    # run -----------------------------------------------------------
    def run(self):
        img_pages = self.rep.image_pages()
        top = self._header_height()            # table pages use the short header
        table_pages = self.flow_tables(top, top)
        total = len(img_pages) + len(table_pages)
        n = 0
        for i, pg in enumerate(img_pages):
            if n:
                self.pdf.newPage()
            n += 1
            top = self.header(first=(i == 0))
            self.image_page(pg, top)
            self.footer(n, total)
        for items in table_pages:
            self.pdf.newPage()
            n += 1
            y = self.header(first=False)
            for kind, t, rows, cont in items:
                y = self.draw_table(t, rows, cont, y) if kind == "table" else self.draw_sign(y)
            self.footer(n, total)

    def _header_height(self) -> float:
        """Height of the short header of the following pages."""
        return 6.5 * self.mm + 2.5 * self.mm
