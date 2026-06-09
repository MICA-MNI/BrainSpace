"""Tests for display-independent HTML surface reports."""

import json
import re
import subprocess
import sys

import numpy as np

from brainspace.mesh.mesh_creation import build_polydata
from brainspace.plotting import write_surface_report


def _surfaces():
    points_lh = np.array([
        [-1.0, 0.0, 0.0],
        [-0.2, 0.0, 0.0],
        [-0.6, 0.7, 0.0],
    ])
    points_rh = np.array([
        [0.2, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.6, 0.7, 0.0],
    ])
    cells = np.array([[0, 1, 2]])
    return build_polydata(points_lh, cells), build_polydata(points_rh, cells)


def _payload(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="brainspace-data" type="application/json">(.*?)</script>',
        text,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_write_surface_report_embeds_maps(tmp_path):
    surf_lh, surf_rh = _surfaces()
    values = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0])
    out = write_surface_report(
        {"lh": surf_lh, "rh": surf_rh},
        {"thickness": values},
        tmp_path / "report.html",
        color_range="sym",
        nan_color=(0.5, 0.5, 0.5, 1),
    )

    payload = _payload(out)
    assert payload["title"] == "BrainSpace Surface Report"
    assert payload["maps"][0]["name"] == "thickness"
    assert payload["maps"][0]["range"] == [-4.0, 4.0]
    assert payload["maps"][0]["values"]["lh"] == [0.0, 1.0, None]
    assert payload["nanColor"] == [128, 128, 128, 255]
    assert payload["surfaces"]["lh"]["cells"] == [[0, 1, 2]]


def test_write_surface_report_embeds_map_metadata(tmp_path):
    surf_lh, surf_rh = _surfaces()
    out = write_surface_report(
        {"lh": surf_lh, "rh": surf_rh},
        {
            "Cortical thickness": {
                "values": np.arange(6, dtype=float),
                "unit": "mm",
                "description": "Real-valued marker",
                "cmap": "viridis",
                "color_range": (0, 5),
            }
        },
        tmp_path / "report.html",
    )

    payload = _payload(out)
    assert payload["maps"][0]["name"] == "Cortical thickness"
    assert payload["maps"][0]["unit"] == "mm"
    assert payload["maps"][0]["description"] == "Real-valued marker"
    assert payload["maps"][0]["range"] == [0.0, 5.0]


def test_plot_hemispheres_html_backend(tmp_path):
    from brainspace.plotting import plot_hemispheres

    surf_lh, surf_rh = _surfaces()
    out = tmp_path / "plot.html"
    result = plot_hemispheres(
        surf_lh,
        surf_rh,
        np.arange(6, dtype=float),
        backend="html",
        filename=out,
        title="Demo",
    )

    assert result == out
    payload = _payload(out)
    assert payload["title"] == "Demo"
    assert payload["maps"][0]["values"]["rh"] == [3.0, 4.0, 5.0]


def test_report_import_does_not_load_vtk_renderer():
    code = (
        "import sys; "
        "from brainspace.plotting import write_surface_report; "
        "print('vtk' in sys.modules); "
        "print('brainspace.plotting.base' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines() == ["False", "False"]


def test_plain_dict_report_generation_does_not_load_vtk_renderer(tmp_path):
    out = tmp_path / "plain.html"
    code = """
import sys
from pathlib import Path

import numpy as np

from brainspace.plotting import write_surface_report

lh = {
    "points": np.array([[-1.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.6, 0.7, 0.0]]),
    "cells": np.array([[0, 1, 2]]),
}
rh = {
    "points": np.array([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0], [0.6, 0.7, 0.0]]),
    "cells": np.array([[0, 1, 2]]),
}
write_surface_report({"lh": lh, "rh": rh}, np.arange(6, dtype=float), Path(%r))
print(Path(%r).exists())
print("vtk" in sys.modules)
print("brainspace.plotting.base" in sys.modules)
""" % (str(out), str(out))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines() == ["True", "False", "False"]
