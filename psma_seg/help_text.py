"""Help shown by F1 (Help → Keyboard shortcuts & help)."""

HELP_HTML = """
<h2>PSMA whole-body lesion segmentation</h2>

<h3>Workflow</h3>
<ol>
<li><b>File → Open folder</b> (Ctrl+Shift+O): every NM whole-body file below the folder is
listed. ○ not started · … in progress · ✔ complete · ✖ excluded.
The first unfinished file opens automatically.</li>
<li>Adjust contrast if needed (right-drag, or <i>Auto contrast</i> = A).</li>
<li>Segment every lesion on the <b>anterior</b> and the <b>posterior</b> view
(usually <b>Region grow</b>, then refine with brush / eraser).</li>
<li>Do <b>not</b> segment physiological uptake (salivary glands, kidneys, liver, spleen,
bowel, bladder) unless it is a lesion.</li>
<li>Set the <b>Status</b>: <i>Complete</i> when the case is fully reviewed. A complete case
with no lesion is a valid negative case. Use <i>Excluded</i> for unusable scans.</li>
<li><b>Save &amp; next</b> (Ctrl+Enter), or Save (Ctrl+S).</li>
</ol>

<h3>Tools</h3>
<table cellpadding="3">
<tr><td><b>G</b></td><td>Region grow: click a hot lesion. Pixels ≥ threshold % of the local
peak, connected to it and within the max radius, are added. Moving the sliders re-tunes
the last lesion.</td></tr>
<tr><td><b>B</b></td><td>Brush: paint by dragging.</td></tr>
<tr><td><b>E</b></td><td>Eraser.</td></tr>
<tr><td><b>L</b></td><td>Lasso: drag a closed outline, <i>or</i> click points and
double-click / Enter to close. Esc cancels, Backspace removes the last point.</td></tr>
<tr><td><b>D</b></td><td>Delete lesion: click a lesion to remove it.</td></tr>
<tr><td><b>H</b></td><td>Pan.</td></tr>
<tr><td><b>Shift</b></td><td>with any tool: remove instead of add.</td></tr>
</table>

<h3>Energy windows</h3>
<p>Each energy window (e.g. 113 keV and 208 keV) has its own lesion mask. Choose the window
at the top of the <b>Segment</b> tab, or press <b>W</b> to cycle.</p>
<ul>
<li><b>Sum of all windows</b>: drawing, erasing, region grow, lasso, delete and clear change
<b>every</b> window at once. The overlay shows all lesions:
<span style="color:#ff1e1e">red</span> = in all windows,
<span style="color:#1e8cff">blue</span> = in some windows only.</li>
<li><b>One window</b>: edits change only that window's mask, and only its lesions are
shown.</li>
</ul>
<p>Undo / redo work across windows.</p>

<h3>Keys</h3>
<table cellpadding="3">
<tr><td>Ctrl+Z / Ctrl+Y</td><td>Undo / redo</td></tr>
<tr><td>[ &nbsp; ]</td><td>Smaller / bigger brush</td></tr>
<tr><td>X</td><td>Show / hide mask</td></tr>
<tr><td>W</td><td>Next energy window (Sum → 113 keV → 208 keV …)</td></tr>
<tr><td>A</td><td>Auto contrast</td></tr>
<tr><td>I</td><td>Invert grey scale</td></tr>
<tr><td>F</td><td>Fit image to window</td></tr>
<tr><td>Ctrl+S</td><td>Save case</td></tr>
<tr><td>Ctrl+Enter</td><td>Save and open the next file</td></tr>
<tr><td>Ctrl+O / Ctrl+Shift+O</td><td>Open file / folder</td></tr>
</table>

<h3>Mouse</h3>
<p>Wheel: zoom (both views stay in sync) · Middle-drag or Ctrl+left-drag: pan ·
Right-drag: contrast (up/down = upper level, left/right = lower level).</p>

<h3>Orientation</h3>
<p>Both views are shown with the patient's right on the viewer's left. The posterior view
is mirrored for display only; masks are stored in the original DICOM orientation.</p>

<h3>Lesion table</h3>
<p>Lists the lesions of the selected energy window; in <i>Sum</i> the <b>Windows</b> column
says which windows each lesion is in. Click a row to jump to that lesion. Counts are from
the energy window currently shown.</p>
"""
