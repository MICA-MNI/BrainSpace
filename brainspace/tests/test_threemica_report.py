"""Tests for the optional threemica bridge."""

import subprocess
import sys
import types

import numpy as np

import brainspace.plotting.threemica_report as threemica_report
from brainspace.plotting.threemica_report import (
    available_threemica_surface_maps,
    write_brainspace_threemica_report,
    write_threemica_report,
)


def test_available_threemica_surface_maps_lists_builtin_choices():
    maps = available_threemica_surface_maps()
    assert "thickness" in maps
    assert "fc_gradient1" in maps
    assert "mpc_gradient2" in maps
    assert maps["thickness"]["group"] == "markers"
    assert maps["fc_gradient1"]["group"] == "gradients"


def test_write_threemica_report_delegates(monkeypatch, tmp_path):
    calls = {}

    def run(**kwargs):
        calls.update(kwargs)
        return [tmp_path / "report.html"]

    fake = types.SimpleNamespace(run=run)
    monkeypatch.setitem(sys.modules, "threemica", fake)

    out = write_threemica_report(
        bids_root=tmp_path / "bids",
        subjects=["sub-001"],
        sessions=["ses-01"],
        maps=["thickness"],
        resolution="fsLR-32k",
        output_root=tmp_path / "out",
        smooth_mm=5,
        scope={"surface": {}},
    )

    assert out == [tmp_path / "report.html"]
    assert calls["subjects"] == ["sub-001"]
    assert calls["sessions"] == ["ses-01"]
    assert calls["maps"] == ["thickness"]
    assert calls["resolution"] == "fsLR-32k"
    assert calls["output_root"] == tmp_path / "out"
    assert calls["smooth_mm"] == 5
    assert calls["interactive"] is False
    assert calls["scope"] == {"surface": {}}


def test_write_brainspace_threemica_report_selects_builtin_maps(monkeypatch, tmp_path):
    calls = {}

    def fake_write(**kwargs):
        calls.update(kwargs)
        return [kwargs["output_root"] / "derivatives" / "threemica" / "report.html"]

    monkeypatch.setattr(threemica_report, "write_threemica_report", fake_write)

    out = write_brainspace_threemica_report(
        tmp_path / "out",
        maps=["thickness", "fc_gradient1"],
        clean_input=False,
    )

    assert out == [tmp_path / "out" / "derivatives" / "threemica" / "report.html"]
    assert calls["subjects"] == ["sub-brainspaceref"]
    assert calls["maps"] == ["thickness", "fc_gradient1"]
    assert calls["resolution"] == "fsLR-32k"
    assert calls["interactive"] is False
    assert [m["tag"] for m in calls["scope"]["brainspace"]["maps"]] == [
        "thickness",
        "fc_gradient1",
    ]
    assert (calls["bids_root"] / "derivatives" / "brainspace" /
            "sub-brainspaceref" / "surf" /
            "sub-brainspaceref_hemi-L_surf-fsLR-32k_label-midthickness.surf.gii").exists()
    assert (calls["bids_root"] / "derivatives" / "brainspace" /
            "sub-brainspaceref" / "maps" /
            "sub-brainspaceref_hemi-R_surf-fsLR-32k_label-fc_gradient1.func.gii").exists()


def test_write_brainspace_threemica_report_expands_custom_components(monkeypatch, tmp_path):
    calls = {}

    def fake_write(**kwargs):
        calls.update(kwargs)
        return [kwargs["output_root"] / "report.html"]

    monkeypatch.setattr(threemica_report, "write_threemica_report", fake_write)
    values = np.column_stack([
        np.linspace(-1, 1, 64984),
        np.linspace(1, -1, 64984),
    ])

    write_brainspace_threemica_report(
        tmp_path / "out",
        maps=[],
        custom_maps={
            "aligned_gradient": {
                "values": values,
                "label": "Aligned Gradient",
                "unit": "score",
                "cmap": "diverging",
            }
        },
        clean_input=False,
    )

    assert calls["maps"] == ["aligned_gradient_1", "aligned_gradient_2"]
    scope_maps = calls["scope"]["brainspace"]["maps"]
    assert [m["label"] for m in scope_maps] == ["Aligned Gradient 1", "Aligned Gradient 2"]
    assert all(m["unit"] == "score" for m in scope_maps)


def test_threemica_report_import_does_not_load_vtk_renderer():
    code = (
        "import sys; "
        "from brainspace.plotting import write_brainspace_threemica_report, write_threemica_report; "
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
