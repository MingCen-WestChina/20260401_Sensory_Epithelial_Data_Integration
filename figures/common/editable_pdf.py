"""Minimal vector-PDF writer used by the manuscript figure scripts.

The writer keeps scatter points and text as independent vector objects, which is
useful when a final panel must be adjusted in Adobe Illustrator.  Font files are
resolved through Matplotlib so the code works on Windows, macOS, and Linux.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas


FONT_FAMILY = os.environ.get("COCHLEA_FONT_FAMILY", "Arial")
_REGISTERED_FONTS: dict[tuple[bool, bool], str] = {}


def _register_fonts() -> None:
    variants = {
        (False, False): ("normal", "normal", "Cochlea-Regular"),
        (True, False): ("normal", "bold", "Cochlea-Bold"),
        (False, True): ("italic", "normal", "Cochlea-Italic"),
        (True, True): ("italic", "bold", "Cochlea-BoldItalic"),
    }
    for key, (style, weight, pdf_name) in variants.items():
        properties = font_manager.FontProperties(
            family=FONT_FAMILY,
            style=style,
            weight=weight,
        )
        font_file = Path(font_manager.findfont(properties, fallback_to_default=True))
        if not font_file.is_file():
            raise FileNotFoundError(f"Matplotlib could not resolve a TrueType font: {font_file}")
        if pdf_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(pdf_name, str(font_file)))
        _REGISTERED_FONTS[key] = pdf_name


def _font_name(text: Text) -> str:
    italic = text.get_fontstyle() in {"italic", "oblique"}
    weight = text.get_fontweight()
    bold = str(weight).lower() in {"bold", "heavy", "semibold", "demibold"}
    if isinstance(weight, (int, float)):
        bold = weight >= 600
    return _REGISTERED_FONTS[(bold, italic)]


def _set_fill(pdf, color, alpha=None) -> None:
    red, green, blue, color_alpha = to_rgba(color)
    pdf.setFillColorRGB(red, green, blue)
    pdf.setFillAlpha(color_alpha if alpha is None else color_alpha * alpha)


def _set_stroke(pdf, color, alpha=None) -> None:
    red, green, blue, color_alpha = to_rgba(color)
    pdf.setStrokeColorRGB(red, green, blue)
    pdf.setStrokeAlpha(color_alpha if alpha is None else color_alpha * alpha)


def _page_point(display_xy, fig, crop_bbox):
    factor = 72.0 / fig.dpi
    return (
        display_xy[0] * factor - crop_bbox.x0 * 72.0,
        display_xy[1] * factor - crop_bbox.y0 * 72.0,
    )


def _draw_rectangle(pdf, patch, fig, crop_bbox) -> None:
    if not patch.get_visible() or patch.get_fill() is False:
        return
    facecolor = patch.get_facecolor()
    if facecolor[3] <= 0:
        return
    vertices = patch.get_transform().transform(patch.get_path().vertices[:4])
    page_vertices = [_page_point(vertex, fig, crop_bbox) for vertex in vertices]
    path = pdf.beginPath()
    path.moveTo(*page_vertices[0])
    for vertex in page_vertices[1:]:
        path.lineTo(*vertex)
    path.close()
    _set_fill(pdf, facecolor, patch.get_alpha())
    pdf.drawPath(path, fill=1, stroke=0)


def _draw_collection(pdf, collection, fig, crop_bbox) -> None:
    if not collection.get_visible() or not isinstance(collection, PathCollection):
        return
    offsets = np.asarray(collection.get_offsets())
    if offsets.size == 0:
        return
    display_offsets = collection.get_offset_transform().transform(offsets)
    sizes = np.asarray(collection.get_sizes(), dtype=float)
    if sizes.size == 0:
        return
    if sizes.size == 1:
        sizes = np.repeat(sizes, len(offsets))
    facecolors = np.asarray(collection.get_facecolors())
    if facecolors.shape[0] == 0:
        return
    if facecolors.shape[0] == 1:
        facecolors = np.repeat(facecolors, len(offsets), axis=0)
    collection_alpha = collection.get_alpha()
    for display_xy, size, facecolor in zip(display_offsets, sizes, facecolors):
        if not np.isfinite(display_xy).all() or size <= 0 or facecolor[3] <= 0:
            continue
        x, y = _page_point(display_xy, fig, crop_bbox)
        radius = np.sqrt(size) / 2.0
        _set_fill(pdf, facecolor, collection_alpha)
        pdf.circle(x, y, radius, stroke=0, fill=1)


def _draw_line(pdf, line, fig, crop_bbox) -> None:
    if not line.get_visible() or not isinstance(line, Line2D):
        return
    x_values = np.asarray(line.get_xdata(orig=False), dtype=float)
    y_values = np.asarray(line.get_ydata(orig=False), dtype=float)
    if x_values.size == 0 or y_values.size == 0:
        return
    display_points = line.get_transform().transform(np.column_stack([x_values, y_values]))
    linestyle = line.get_linestyle()
    if linestyle not in {"None", "none", "", " "} and len(display_points) > 1:
        path = pdf.beginPath()
        drawing = False
        for display_xy in display_points:
            if not np.isfinite(display_xy).all():
                drawing = False
                continue
            point = _page_point(display_xy, fig, crop_bbox)
            if drawing:
                path.lineTo(*point)
            else:
                path.moveTo(*point)
                drawing = True
        _set_stroke(pdf, line.get_color(), line.get_alpha())
        pdf.setLineWidth(float(line.get_linewidth()))
        pdf.drawPath(path, fill=0, stroke=1)

    if line.get_marker() == "o":
        marker_indices = np.arange(len(display_points))
        markevery = line.get_markevery()
        if isinstance(markevery, (list, tuple, np.ndarray)):
            marker_indices = np.asarray(markevery, dtype=int)
        radius = float(line.get_markersize()) / 2.0
        marker_color = line.get_markerfacecolor()
        for index in marker_indices:
            if index < 0 or index >= len(display_points):
                continue
            display_xy = display_points[index]
            if not np.isfinite(display_xy).all():
                continue
            x, y = _page_point(display_xy, fig, crop_bbox)
            _set_fill(pdf, marker_color, line.get_alpha())
            pdf.circle(x, y, radius, stroke=0, fill=1)


def _draw_spines_and_ticks(pdf, ax, fig, crop_bbox) -> None:
    if not ax.axison:
        return
    for spine in ax.spines.values():
        if not spine.get_visible() or spine.get_linewidth() <= 0:
            continue
        vertices = spine.get_transform().transform(spine.get_path().vertices)
        page_vertices = [_page_point(vertex, fig, crop_bbox) for vertex in vertices]
        path = pdf.beginPath()
        path.moveTo(*page_vertices[0])
        for vertex in page_vertices[1:]:
            path.lineTo(*vertex)
        _set_stroke(pdf, spine.get_edgecolor(), spine.get_alpha())
        pdf.setLineWidth(float(spine.get_linewidth()))
        pdf.drawPath(path, fill=0, stroke=1)

    for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        for tick in axis.get_major_ticks():
            tick_line = tick.tick1line
            length = float(tick_line.get_markersize())
            if not tick_line.get_visible() or length <= 0:
                continue
            location = tick.get_loc()
            display_xy = tick_line.get_transform().transform([[location, 0]])[0]
            x, y = _page_point(display_xy, fig, crop_bbox)
            end = (x, y - length) if axis_name == "x" else (x - length, y)
            _set_stroke(pdf, tick_line.get_color(), tick_line.get_alpha())
            pdf.setLineWidth(float(tick_line.get_markeredgewidth()))
            pdf.line(x, y, *end)


def _draw_text(pdf, text, fig, crop_bbox) -> None:
    if not isinstance(text, Text) or not text.get_visible():
        return
    value = text.get_text()
    if not value:
        return
    display_xy = text.get_transform().transform(text.get_position())
    x, y = _page_point(display_xy, fig, crop_bbox)
    font_name = _font_name(text)
    font_size = float(text.get_fontsize())
    width = pdfmetrics.stringWidth(value, font_name, font_size)
    ascent, descent = pdfmetrics.getAscentDescent(font_name, font_size)
    x_offset = {"left": 0.0, "center": -width / 2.0, "right": -width}.get(
        text.get_horizontalalignment(), 0.0
    )
    y_offset = {
        "baseline": 0.0,
        "bottom": -descent,
        "top": -ascent,
        "center": -(ascent + descent) / 2.0,
        "center_baseline": -(ascent + descent) / 2.0,
    }.get(text.get_verticalalignment(), 0.0)
    _set_fill(pdf, text.get_color(), text.get_alpha())
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(float(text.get_rotation()))
    pdf.setFont(font_name, font_size)
    pdf.drawString(x_offset, y_offset, value)
    pdf.restoreState()


def save_ai_editable_pdf(fig, output_file, bbox_inches=None, title=None) -> None:
    """Save a figure as a PDF with independent vector paths and editable text."""
    _register_fonts()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if bbox_inches is None:
        crop_bbox = fig.bbox_inches
    elif bbox_inches == "tight":
        crop_bbox = fig.get_tightbbox(renderer).padded(0.1)
    else:
        crop_bbox = bbox_inches

    page_size = (crop_bbox.width * 72.0, crop_bbox.height * 72.0)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdf_canvas.Canvas(str(output_file), pagesize=page_size, pageCompression=1)
    pdf.setCreator("Cochlear sensory epithelium figure workflow")
    if title:
        pdf.setTitle(title)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)

    for ax in fig.axes:
        for patch in ax.patches:
            if isinstance(patch, Rectangle):
                _draw_rectangle(pdf, patch, fig, crop_bbox)
    for collection in fig.findobj(PathCollection):
        _draw_collection(pdf, collection, fig, crop_bbox)
    for line in fig.findobj(Line2D):
        _draw_line(pdf, line, fig, crop_bbox)
    for ax in fig.axes:
        _draw_spines_and_ticks(pdf, ax, fig, crop_bbox)
    for text in fig.findobj(Text):
        _draw_text(pdf, text, fig, crop_bbox)

    pdf.showPage()
    pdf.save()
