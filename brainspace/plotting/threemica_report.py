"""Optional bridge to Zhengchen Cai's threemica HTML viewer."""

# Author: The BrainSpace developers
# License: BSD 3 clause

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_BRAINSPACE_MAPS = {
    "curvature": {
        "filename": "conte69_32k_curvature.csv",
        "label": "Cortical Curvature",
        "unit": "1/mm",
        "cmap": "diverging",
        "group": "markers",
    },
    "thickness": {
        "filename": "conte69_32k_thickness.csv",
        "label": "Cortical Thickness",
        "unit": "mm",
        "cmap": "pos-only",
        "group": "markers",
    },
    "t1wt2w": {
        "filename": "conte69_32k_t1wt2w.csv",
        "label": "T1w/T2w Myelin Proxy",
        "unit": "ratio",
        "cmap": "pos-only",
        "group": "markers",
    },
    "fc_gradient1": {
        "filename": "conte69_32k_fc_gradient0.csv",
        "label": "Functional Connectivity Gradient 1",
        "unit": "score",
        "cmap": "diverging",
        "group": "gradients",
    },
    "fc_gradient2": {
        "filename": "conte69_32k_fc_gradient1.csv",
        "label": "Functional Connectivity Gradient 2",
        "unit": "score",
        "cmap": "diverging",
        "group": "gradients",
    },
    "mpc_gradient1": {
        "filename": "conte69_32k_mpc_gradient0.csv",
        "label": "MPC Gradient 1",
        "unit": "score",
        "cmap": "diverging",
        "group": "gradients",
    },
    "mpc_gradient2": {
        "filename": "conte69_32k_mpc_gradient1.csv",
        "label": "MPC Gradient 2",
        "unit": "score",
        "cmap": "diverging",
        "group": "gradients",
    },
}


@dataclass(frozen=True)
class ThreemicaSurfaceMap:
    """Surface map specification for threemica export."""

    tag: str
    label: str
    values: np.ndarray
    unit: str = ""
    cmap: str = "auto"
    scale: float = 1.0


def available_threemica_surface_maps():
    """Return BrainSpace bundled surface maps available for threemica export.

    Returns
    -------
    dict
        Mapping from map tag to metadata. Tags can be passed to
        :func:`write_brainspace_threemica_report` via ``maps=[...]``.
    """
    return {
        tag: {
            "label": spec["label"],
            "unit": spec["unit"],
            "cmap": spec["cmap"],
            "group": spec["group"],
        }
        for tag, spec in _BRAINSPACE_MAPS.items()
    }


def _datasets_root():
    return Path(__file__).resolve().parents[1] / "datasets"


def _select_builtin_maps(selection):
    if selection is None or (isinstance(selection, str) and selection == "all"):
        return list(_BRAINSPACE_MAPS)
    if isinstance(selection, str):
        if selection in {"markers", "gradients"}:
            return [
                tag for tag, spec in _BRAINSPACE_MAPS.items()
                if spec["group"] == selection
            ]
        selection = [selection]

    unknown = [tag for tag in selection if tag not in _BRAINSPACE_MAPS]
    if unknown:
        raise ValueError(
            "Unknown BrainSpace threemica map(s): {}. Available maps are: {}.".format(
                ", ".join(unknown),
                ", ".join(sorted(_BRAINSPACE_MAPS)),
            )
        )
    return list(selection)


def _builtin_surface_map(tag):
    spec = _BRAINSPACE_MAPS[tag]
    path = _datasets_root() / "matrices" / "main_group" / spec["filename"]
    values = np.loadtxt(path, dtype=np.float32)
    return ThreemicaSurfaceMap(
        tag=tag,
        label=spec["label"],
        values=values,
        unit=spec["unit"],
        cmap=spec["cmap"],
    )


def _safe_tag(value):
    tag = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    tag = "_".join(part for part in tag.split("_") if part)
    return tag or "map"


def _custom_surface_maps(custom_maps):
    if not custom_maps:
        return []

    out = []
    for name, spec in custom_maps.items():
        if isinstance(spec, dict):
            values = spec["values"]
            label = spec.get("label", name)
            unit = spec.get("unit", "")
            cmap = spec.get("cmap", "auto")
            scale = float(spec.get("scale", 1.0))
        else:
            values = spec
            label = name
            unit = ""
            cmap = "auto"
            scale = 1.0

        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            out.append(ThreemicaSurfaceMap(_safe_tag(name), str(label), arr, unit, cmap, scale))
            continue
        if arr.ndim == 2:
            if arr.shape[0] < arr.shape[1]:
                arr = arr.T
            for i in range(arr.shape[1]):
                out.append(ThreemicaSurfaceMap(
                    "{}_{}".format(_safe_tag(name), i + 1),
                    "{} {}".format(label, i + 1),
                    arr[:, i],
                    unit,
                    cmap,
                    scale,
                ))
            continue
        raise ValueError("Custom map {!r} must be one- or two-dimensional.".format(name))
    return out


def _write_func_gii(path, values):
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    img = GiftiImage(darrays=[
        GiftiDataArray(np.asarray(values, dtype=np.float32), intent="NIFTI_INTENT_SHAPE")
    ])
    nib.save(img, str(path))


def _surface_arrays(surf):
    if isinstance(surf, dict):
        return (
            np.asarray(surf["points"], dtype=np.float32),
            np.asarray(surf["cells"], dtype=np.int32),
        )
    return (
        np.asarray(surf.Points, dtype=np.float32),
        np.asarray(surf.GetCells2D(), dtype=np.int32),
    )


def _write_surface_gii(path, surf):
    import nibabel as nib
    from nibabel.gifti import GiftiDataArray, GiftiImage

    points, cells = _surface_arrays(surf)
    img = GiftiImage(darrays=[
        GiftiDataArray(points, intent="NIFTI_INTENT_POINTSET"),
        GiftiDataArray(cells, intent="NIFTI_INTENT_TRIANGLE"),
    ])
    nib.save(img, str(path))


def _prepare_surface_files(surfaces, surf_dir, subject, resolution):
    if resolution != "fsLR-32k":
        raise ValueError("BrainSpace bundled threemica export currently supports resolution='fsLR-32k'.")

    if surfaces is None:
        root = _datasets_root() / "surfaces"
        sources = {
            "L": root / "conte69_32k_lh.gii",
            "R": root / "conte69_32k_rh.gii",
        }
        n_lh = n_rh = 32492
        for hemi, src in sources.items():
            dst = surf_dir / "{}_hemi-{}_surf-{}_label-midthickness.surf.gii".format(
                subject, hemi, resolution
            )
            shutil.copy2(src, dst)
        return n_lh, n_rh

    if isinstance(surfaces, dict):
        surf_lh, surf_rh = surfaces["lh"], surfaces["rh"]
    else:
        surf_lh, surf_rh = surfaces

    for hemi, surf in [("L", surf_lh), ("R", surf_rh)]:
        dst = surf_dir / "{}_hemi-{}_surf-{}_label-midthickness.surf.gii".format(
            subject, hemi, resolution
        )
        _write_surface_gii(dst, surf)
    return _surface_arrays(surf_lh)[0].shape[0], _surface_arrays(surf_rh)[0].shape[0]


def write_brainspace_threemica_report(output_root, *, maps="all",
                                      custom_maps=None, surfaces=None,
                                      subject="sub-brainspaceref",
                                      resolution="fsLR-32k",
                                      smooth_mm=None,
                                      clean_input=True):
    """Export BrainSpace surface maps through threemica's original viewer.

    Parameters
    ----------
    output_root : str or path-like
        Directory where threemica writes ``derivatives/threemica``.
    maps : {'all', 'markers', 'gradients'} or sequence of str, optional
        BrainSpace bundled maps to include. Available tags are returned by
        :func:`available_threemica_surface_maps`.
    custom_maps : dict, optional
        Additional surface maps. Values can be arrays, or dictionaries with
        ``values``, ``label``, ``unit``, ``cmap`` and ``scale`` keys. Two-
        dimensional arrays are exported column-wise.
    surfaces : pair or dict, optional
        Left/right surfaces. If omitted, BrainSpace's bundled Conte69 32k
        surfaces are used.
    subject : str, optional
        Subject ID used in the temporary BIDS derivative. The default is a
        synthetic reference label for BrainSpace's bundled Conte69 maps; pass a
        real ``sub-*`` ID when exporting participant-specific maps.
    resolution : {'fsLR-32k'}, optional
        threemica resolution label.
    smooth_mm : int, optional
        Smoothing FWHM passed through to threemica.
    clean_input : bool, optional
        Remove the temporary input BIDS tree after the self-contained HTML is
        written.

    Returns
    -------
    list[pathlib.Path]
        Paths returned by threemica.
    """
    output_root = Path(output_root).resolve()
    if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject):
        raise ValueError("subject must be a BIDS label: sub- followed by letters or digits.")
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(tempfile.mkdtemp(prefix="_brainspace_threemica_input-",
                                       dir=output_root))
    try:
        derivative = "brainspace"
        base = input_root / "derivatives" / derivative / subject
        surf_dir = base / "surf"
        map_dir = base / "maps"

        surf_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        n_lh, n_rh = _prepare_surface_files(surfaces, surf_dir, subject, resolution)
        selected = [_builtin_surface_map(tag) for tag in _select_builtin_maps(maps)]
        selected.extend(_custom_surface_maps(custom_maps))
        if not selected:
            raise ValueError("No maps selected for threemica export.")

        tags = [item.tag for item in selected]
        if len(tags) != len(set(tags)):
            raise ValueError("Surface map tags must be unique after normalization.")
        scope_maps = []
        for item in selected:
            values = np.asarray(item.values, dtype=np.float32).ravel()
            if values.size != n_lh + n_rh:
                raise ValueError(
                    "Map {!r} has {} values; expected {} ({} left + {} right).".format(
                        item.tag, values.size, n_lh + n_rh, n_lh, n_rh
                    )
                )
            for hemi, hemi_values in [("L", values[:n_lh]), ("R", values[n_lh:])]:
                path = map_dir / "{}_hemi-{}_surf-{}_label-{}.func.gii".format(
                    subject, hemi, resolution, item.tag
                )
                _write_func_gii(path, hemi_values)
            scope_maps.append({
                "tag": item.tag,
                "label": item.label,
                "unit": item.unit,
                "cmap": item.cmap,
                "scale": item.scale,
            })

        scope = {
            "surface": {
                "derivative": derivative,
                "subdir": "surf",
                "label": "midthickness",
            },
            derivative: {"maps": scope_maps},
        }

        return write_threemica_report(
            bids_root=input_root,
            subjects=[subject],
            maps=[m.tag for m in selected],
            resolution=resolution,
            output_root=output_root,
            smooth_mm=smooth_mm,
            interactive=False,
            scope=scope,
        )
    finally:
        if clean_input:
            shutil.rmtree(input_root, ignore_errors=True)


def write_threemica_report(bids_root=None, *, subjects=None, sessions=None,
                           maps=None, resolution=None, output_root=None,
                           smooth_mm=None, interactive=False, scope=None):
    """Write threemica reports without modifying the threemica viewer.

    This is a thin optional-dependency adapter. BrainSpace passes arguments to
    :func:`threemica.run` and leaves threemica's HTML template, JavaScript
    runtime, atlas/query payload, controls, and output layout unchanged.

    Parameters
    ----------
    bids_root : str or path-like, optional
        BIDS root containing ``derivatives/``. If omitted, threemica resolves
        from the current working directory.
    subjects : sequence of str, optional
        Subject IDs, for example ``["sub-001"]``. Required when
        ``interactive=False``.
    sessions : sequence of str, optional
        Session IDs to include.
    maps : sequence of str, optional
        threemica scope tags to render. Required when ``interactive=False``.
    resolution : str or sequence of str, optional
        threemica-supported resolution, for example ``"fsLR-32k"``. Required
        when ``interactive=False``.
    output_root : str or path-like, optional
        Root under which threemica writes ``derivatives/threemica``.
    smooth_mm : int, optional
        Optional smoothing FWHM passed through to threemica.
    interactive : bool, optional
        Whether threemica should prompt for missing choices.
    scope : dict, optional
        Normalized threemica scope dictionary. If omitted, threemica loads or
        creates ``derivatives/threemica/threemica_scope.json``.

    Returns
    -------
    list[pathlib.Path]
        Paths returned by :func:`threemica.run`.
    """
    try:
        from threemica import run
    except ImportError as exc:
        raise ImportError(
            "write_threemica_report requires the optional 'threemica' package. "
            "Install threemica to use Zhengchen Cai's original viewer."
        ) from exc

    out = run(
        bids_root=bids_root,
        subjects=list(subjects) if subjects is not None else None,
        sessions=list(sessions) if sessions is not None else None,
        maps=list(maps) if maps is not None else None,
        resolution=resolution,
        output_root=Path(output_root) if output_root is not None else None,
        smooth_mm=smooth_mm,
        interactive=interactive,
        scope=scope,
    )
    return [Path(p) for p in out]
