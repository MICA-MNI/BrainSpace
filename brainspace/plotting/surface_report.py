"""Display-independent HTML reports for surface maps."""

# Author: The BrainSpace developers
# License: BSD 3 clause

import json
from html import escape
from pathlib import Path

import numpy as np

from .colormaps import colormaps


def _rgba255(color):
    arr = np.asarray(color, dtype=float)
    if arr.size == 3:
        arr = np.append(arr, 1.0)
    if arr.size != 4:
        raise ValueError("Colors must have 3 or 4 channels.")
    if np.nanmax(arr) <= 1.0:
        arr = arr * 255
    return np.clip(np.round(arr), 0, 255).astype(int).tolist()


def _json_list(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(object)
        out[~np.isfinite(arr)] = None
        return out.tolist()
    return arr.tolist()


def _n_points(surf):
    if isinstance(surf, dict):
        return np.asarray(surf["points"]).shape[0]
    return surf.n_points


def _surface_payload(surf, name):
    if isinstance(surf, dict):
        points = np.asarray(surf["points"], dtype=np.float32)
        cells = np.asarray(surf["cells"], dtype=np.int64)
    else:
        points = np.asarray(surf.Points, dtype=np.float32)
        cells = np.asarray(surf.GetCells2D(), dtype=np.int64)

    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(
            "HTML surface reports currently require triangular surfaces; "
            "{} has cells with shape {}.".format(name, cells.shape)
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "HTML surface reports require point coordinates with shape "
            "(n_points, 3); {} has points with shape {}.".format(name, points.shape)
        )

    return {
        "points": points.round(6).tolist(),
        "cells": cells.tolist(),
    }


def _as_surface_dict(surfaces):
    if isinstance(surfaces, dict):
        if "lh" not in surfaces or "rh" not in surfaces:
            raise ValueError("Surface dictionaries must contain 'lh' and 'rh'.")
        return {"lh": surfaces["lh"], "rh": surfaces["rh"]}
    if isinstance(surfaces, (list, tuple)) and len(surfaces) == 2:
        return {"lh": surfaces[0], "rh": surfaces[1]}
    raise TypeError("Surfaces must be a {'lh': surf_lh, 'rh': surf_rh} dict or a pair.")


def _point_data_values(surfaces, name):
    try:
        return {
            "lh": np.asarray(surfaces["lh"].PointData[name], dtype=float),
            "rh": np.asarray(surfaces["rh"].PointData[name], dtype=float),
        }
    except KeyError as exc:
        raise KeyError("Point-data array {!r} is missing from a surface.".format(name)) from exc


def _split_values(surfaces, values):
    if isinstance(values, dict):
        return {
            "lh": np.asarray(values["lh"], dtype=float).ravel(),
            "rh": np.asarray(values["rh"], dtype=float).ravel(),
        }

    arr = np.asarray(values, dtype=float).ravel()
    n_lh = _n_points(surfaces["lh"])
    n_rh = _n_points(surfaces["rh"])
    if arr.size != n_lh + n_rh:
        raise ValueError(
            "Surface map has {} values; expected {} ({} left + {} right).".format(
                arr.size, n_lh + n_rh, n_lh, n_rh
            )
        )
    return {"lh": arr[:n_lh], "rh": arr[n_lh:]}


def _map_spec(values):
    if isinstance(values, dict) and "values" in values:
        opts = {k: v for k, v in values.items() if k != "values"}
        return values["values"], opts
    return values, {}


def _iter_maps(surfaces, arrays):
    n_lh = _n_points(surfaces["lh"])
    n_rh = _n_points(surfaces["rh"])
    n_total = n_lh + n_rh
    if arrays is None:
        yield "Surface", {"lh": np.zeros(n_lh), "rh": np.zeros(n_rh)}, {}
        return

    if isinstance(arrays, str):
        yield arrays, _point_data_values(surfaces, arrays), {}
        return

    if isinstance(arrays, dict):
        for name, values in arrays.items():
            values, opts = _map_spec(values)
            if isinstance(values, str):
                yield str(name), _point_data_values(surfaces, values), opts
            else:
                yield str(name), _split_values(surfaces, values), opts
        return

    arr = np.asarray(arrays, dtype=object)
    if arr.ndim == 1 and arr.size == n_total:
        yield "Map 1", _split_values(surfaces, arrays), {}
        return
    if arr.ndim == 2 and arr.shape[1] == n_total:
        for i, values in enumerate(arr):
            yield "Map {}".format(i + 1), _split_values(surfaces, values), {}
        return

    if isinstance(arrays, (list, tuple)):
        for i, values in enumerate(arrays):
            name = values if isinstance(values, str) else "Map {}".format(i + 1)
            if isinstance(values, str):
                yield name, _point_data_values(surfaces, values), {}
            else:
                yield name, _split_values(surfaces, values), {}
        return

    yield "Map 1", _split_values(surfaces, arrays), {}


def _resolve_cmap(cmap):
    if isinstance(cmap, str) and cmap in colormaps:
        lut = np.asarray(colormaps[cmap], dtype=np.uint8)
    else:
        if not callable(cmap):
            import matplotlib as mpl
            cmap = mpl.colormaps[cmap]
        lut = np.asarray(cmap(np.linspace(0, 1, 256)) * 255, dtype=np.uint8)
    if lut.shape[1] == 3:
        lut = np.column_stack([lut, np.full(lut.shape[0], 255, dtype=np.uint8)])
    return lut[:, :4].astype(int).tolist()


def _map_range(values, color_range):
    finite = np.concatenate([
        np.asarray(values["lh"], dtype=float),
        np.asarray(values["rh"], dtype=float),
    ])
    finite = finite[np.isfinite(finite)]
    if color_range == "sym":
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        return [-vmax, vmax]
    if color_range is not None:
        lo, hi = color_range
        return [float(lo), float(hi)]
    if not finite.size:
        return [0.0, 1.0]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.01
        lo, hi = lo - pad, hi + pad
    return [lo, hi]


def _metadata_html(metadata):
    if not metadata:
        return ""
    rows = "\n".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(escape(str(k)), escape(str(v)))
        for k, v in metadata.items()
    )
    return "<details><summary>Report metadata</summary><table>{}</table></details>".format(rows)


def _html_template(payload):
    title = escape(payload["title"])
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    metadata = _metadata_html(payload.get("metadata"))
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #f6f7f9; color: #15171a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  body {{ display: flex; flex-direction: column; }}
  header {{ display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding: 10px 14px;
    border-bottom: 1px solid #d8dde5; background: #fff; }}
  h1 {{ font-size: 15px; margin: 0; font-weight: 650; }}
  details {{ border-bottom: 1px solid #d8dde5; background: #fff; padding: 8px 14px; }}
  summary {{ cursor: pointer; font-size: 13px; color: #3f4752; }}
  table {{ border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
  th, td {{ text-align: left; padding: 3px 14px 3px 0; vertical-align: top; }}
  th {{ color: #555d69; font-weight: 600; }}
  select, button {{ font: inherit; border: 1px solid #c8ced8; background: #fff;
    border-radius: 6px; padding: 5px 8px; }}
  main {{ flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center;
    padding: 12px 12px 44px; position: relative; }}
  #viewer {{ width: min(100%, calc((100vh - 124px) * 16 / 9), 1440px);
    aspect-ratio: 16 / 9; height: auto; display: block; border: 1px solid #d8dde5;
    background: #f6f7f9; }}
  #mapInfo, #range {{ font-size: 13px; color: #555d69; }}
  #range {{ margin-left: auto; }}
  #colorbarWrap {{ display: flex; align-items: center; gap: 7px; min-width: 260px; }}
  #colorbar {{ width: 150px; height: 12px; border-radius: 999px; border: 1px solid #c8ced8; }}
  #helpBtn {{ width: 32px; padding-left: 0; padding-right: 0; }}
  #help {{ display: none; position: fixed; inset: 0; z-index: 30; align-items: center;
    justify-content: center; background: rgba(21, 23, 26, 0.32); }}
  #help.visible {{ display: flex; }}
  #helpPanel {{ width: min(560px, calc(100vw - 32px)); background: #fff; color: #15171a;
    border: 1px solid #d8dde5; border-radius: 8px; box-shadow: 0 18px 60px rgba(0,0,0,.22);
    padding: 22px; }}
  #helpPanel h2 {{ margin: 0 0 14px; font-size: 18px; }}
  #helpPanel dl {{ display: grid; grid-template-columns: 72px 1fr; gap: 9px 14px; margin: 0; }}
  #helpPanel dt {{ font: 600 13px ui-monospace, SFMono-Regular, Menlo, monospace; color: #15171a; }}
  #helpPanel dd {{ margin: 0; font-size: 13px; color: #555d69; }}
  #helpClose {{ margin-top: 18px; }}
  #helpBar {{ position: absolute; left: 50%; bottom: 12px; transform: translateX(-50%);
    color: #6b7280; font-size: 12px; white-space: nowrap; pointer-events: none; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <select id="mapSelect"></select>
  <span id="mapInfo"></span>
  <span id="colorbarWrap"><span id="cbMin"></span><canvas id="colorbar" width="150" height="12"></canvas><span id="cbMax"></span></span>
  <button id="resetBtn" type="button">Reset</button>
  <button id="helpBtn" type="button" title="Viewer controls">?</button>
  <span id="range"></span>
</header>
{metadata}
<main>
  <canvas id="viewer"></canvas>
  <div id="helpBar">H help · drag rotate · wheel zoom · [ ] switch map · R reset</div>
</main>
<div id="help" aria-hidden="true">
  <div id="helpPanel" role="dialog" aria-modal="true" aria-labelledby="helpTitle">
    <h2 id="helpTitle">Viewer Controls</h2>
    <dl>
      <dt>DRAG</dt><dd>Rotate the surface in 3D.</dd>
      <dt>WHEEL</dt><dd>Zoom in or out.</dd>
      <dt>[ ]</dt><dd>Switch between maps.</dd>
      <dt>R</dt><dd>Reset camera and zoom.</dd>
      <dt>H</dt><dd>Show or hide this help panel.</dd>
    </dl>
    <button id="helpClose" type="button">Close</button>
  </div>
</div>
<script id="brainspace-data" type="application/json">{data}</script>
<script>
const payload = JSON.parse(document.getElementById("brainspace-data").textContent);
const canvas = document.getElementById("viewer");
const gl = canvas.getContext("webgl", {{antialias: true}}) || canvas.getContext("experimental-webgl");
if (!gl) throw new Error("WebGL is not available in this browser.");

const vertexSrc = `
attribute vec3 aPosition;
attribute vec4 aColor;
uniform mat4 uMatrix;
varying vec4 vColor;
void main() {{
  gl_Position = uMatrix * vec4(aPosition, 1.0);
  vColor = aColor;
}}`;
const fragmentSrc = `
precision mediump float;
varying vec4 vColor;
void main() {{
  gl_FragColor = vColor;
}}`;

function shader(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSrc));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSrc));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);

const positions = [];
const sources = [];
const hemiStats = {};
for (const hemi of ["lh", "rh"]) {{
  const pts = payload.surfaces[hemi].points;
  let ymin = Infinity, ymax = -Infinity, zminH = Infinity, zmaxH = -Infinity;
  for (const p of pts) {{
    ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
    zminH = Math.min(zminH, p[2]); zmaxH = Math.max(zmaxH, p[2]);
  }}
  hemiStats[hemi] = {{
    yCenter: (ymin + ymax) / 2,
    zCenter: (zminH + zmaxH) / 2,
    span: Math.max(ymax - ymin, zmaxH - zminH, 1)
  }};
}}
const layoutSpan = Math.max(hemiStats.lh.span, hemiStats.rh.span, 1);
for (const hemi of ["lh", "rh"]) {{
  const stats = hemiStats[hemi];
  const sideOffset = (hemi === "lh" ? -0.58 : 0.58) * layoutSpan;
  const surf = payload.surfaces[hemi];
  for (const cell of surf.cells) {{
    for (const idx of cell) {{
      const p = surf.points[idx];
      const lateralDepth = hemi === "lh" ? -p[0] : p[0];
      positions.push(p[1] - stats.yCenter + sideOffset, p[2] - stats.zCenter, lateralDepth);
      sources.push([hemi, idx]);
    }}
  }}
}}

let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
for (let i = 0; i < positions.length; i += 3) {{
  xmin = Math.min(xmin, positions[i]); xmax = Math.max(xmax, positions[i]);
  ymin = Math.min(ymin, positions[i + 1]); ymax = Math.max(ymax, positions[i + 1]);
  zmin = Math.min(zmin, positions[i + 2]); zmax = Math.max(zmax, positions[i + 2]);
}}
const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2, cz = (zmin + zmax) / 2;
const scale0 = 2.4 / Math.max(xmax - xmin, ymax - ymin, zmax - zmin || 1);
for (let i = 0; i < positions.length; i += 3) {{
  positions[i] = (positions[i] - cx) * scale0;
  positions[i + 1] = (positions[i + 1] - cy) * scale0;
  positions[i + 2] = (positions[i + 2] - cz) * scale0;
}}

const shades = [];
const light = [0.28, -0.45, 0.84];
const llen = Math.hypot(light[0], light[1], light[2]) || 1;
light[0] /= llen; light[1] /= llen; light[2] /= llen;
for (let i = 0; i < positions.length; i += 9) {{
  const ax = positions[i], ay = positions[i + 1], az = positions[i + 2];
  const bx = positions[i + 3], by = positions[i + 4], bz = positions[i + 5];
  const cx2 = positions[i + 6], cy2 = positions[i + 7], cz2 = positions[i + 8];
  const ux = bx - ax, uy = by - ay, uz = bz - az;
  const vx = cx2 - ax, vy = cy2 - ay, vz = cz2 - az;
  let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const nlen = Math.hypot(nx, ny, nz) || 1;
  nx /= nlen; ny /= nlen; nz /= nlen;
  const diffuse = Math.abs(nx * light[0] + ny * light[1] + nz * light[2]);
  const shade = 0.62 + 0.38 * diffuse;
  shades.push(shade, shade, shade);
}}

const posBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
const aPosition = gl.getAttribLocation(program, "aPosition");
gl.enableVertexAttribArray(aPosition);
gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, 0, 0);

const colorBuffer = gl.createBuffer();
const aColor = gl.getAttribLocation(program, "aColor");
gl.enableVertexAttribArray(aColor);
const uMatrix = gl.getUniformLocation(program, "uMatrix");

function colorFor(value, map, shade) {{
  if (value === null || Number.isNaN(value)) return payload.nanColor.map(x => x / 255);
  const lo = map.range[0], hi = map.range[1];
  const t = Math.max(0, Math.min(1, (value - lo) / (hi - lo || 1)));
  const c = map.cmap[Math.round(t * (map.cmap.length - 1))];
  return [c[0] * shade / 255, c[1] * shade / 255, c[2] * shade / 255, c[3] / 255];
}}

function fmt(value) {{
  if (!Number.isFinite(value)) return "NA";
  const a = Math.abs(value);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return value.toExponential(3);
  return value.toPrecision(4);
}}

function drawColorbar(map) {{
  const c = document.getElementById("colorbar");
  const ctx = c.getContext("2d");
  const image = ctx.createImageData(c.width, c.height);
  for (let x = 0; x < c.width; x++) {{
    const lut = map.cmap[Math.round((x / Math.max(c.width - 1, 1)) * (map.cmap.length - 1))];
    for (let y = 0; y < c.height; y++) {{
      const o = (y * c.width + x) * 4;
      image.data[o] = lut[0]; image.data[o + 1] = lut[1];
      image.data[o + 2] = lut[2]; image.data[o + 3] = 255;
    }}
  }}
  ctx.putImageData(image, 0, 0);
  document.getElementById("cbMin").textContent = fmt(map.range[0]);
  document.getElementById("cbMax").textContent = fmt(map.range[1]);
}}

function updateColors(mapIndex) {{
  const map = payload.maps[mapIndex];
  const colors = [];
  for (let i = 0; i < sources.length; i++) {{
    const [hemi, idx] = sources[i];
    colors.push(...colorFor(map.values[hemi][idx], map, shades[i]));
  }}
  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(aColor, 4, gl.FLOAT, false, 0, 0);
  const unit = map.unit ? ` ${{map.unit}}` : "";
  document.getElementById("mapInfo").textContent = map.description || "";
  document.getElementById("range").textContent = `${{fmt(map.range[0])}} to ${{fmt(map.range[1])}}${{unit}}`;
  drawColorbar(map);
  select.value = String(mapIndex);
  draw();
}}

function matrix(rx, ry, zoom) {{
  const sx = Math.sin(rx), cx = Math.cos(rx);
  const sy = Math.sin(ry), cy = Math.cos(ry);
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const xScale = aspect > 1 ? zoom / aspect : zoom;
  const yScale = aspect > 1 ? zoom : zoom * aspect;
  return new Float32Array([
    cy * xScale, sx * sy * xScale, cx * sy * xScale, 0,
    0, cx * yScale, -sx * yScale, 0,
    -sy * xScale, sx * cy * xScale, cx * cy * xScale, 0,
    0, 0, 0, 1
  ]);
}}

let rx = 0.0, ry = 0.0, zoom = 0.95;
function resize() {{
  const ratio = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * ratio);
  const h = Math.floor(canvas.clientHeight * ratio);
  if (canvas.width !== w || canvas.height !== h) {{
    canvas.width = w; canvas.height = h;
    gl.viewport(0, 0, w, h);
  }}
}}
function draw() {{
  resize();
  gl.clearColor(0.965, 0.970, 0.978, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  gl.uniformMatrix4fv(uMatrix, false, matrix(rx, ry, zoom));
  gl.drawArrays(gl.TRIANGLES, 0, positions.length / 3);
}}

let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener("pointerdown", e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); }});
canvas.addEventListener("pointerup", () => dragging = false);
canvas.addEventListener("pointermove", e => {{
  if (!dragging) return;
  ry += (e.clientX - lastX) * 0.01;
  rx += (e.clientY - lastY) * 0.01;
  lastX = e.clientX; lastY = e.clientY;
  draw();
}});
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  zoom = Math.max(0.25, Math.min(3.0, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
  draw();
}}, {{passive: false}});
window.addEventListener("resize", draw);

const select = document.getElementById("mapSelect");
payload.maps.forEach((m, i) => {{
  const option = document.createElement("option");
  option.value = i;
  option.textContent = m.name;
  select.appendChild(option);
}});
select.addEventListener("change", e => updateColors(Number(e.target.value)));
document.getElementById("resetBtn").addEventListener("click", () => {{ rx = 0.0; ry = 0.0; zoom = 0.95; draw(); }});
function toggleHelp(show) {{
  const help = document.getElementById("help");
  const visible = show === undefined ? !help.classList.contains("visible") : show;
  help.classList.toggle("visible", visible);
  help.setAttribute("aria-hidden", visible ? "false" : "true");
}}
document.getElementById("helpBtn").addEventListener("click", () => toggleHelp());
document.getElementById("helpClose").addEventListener("click", () => toggleHelp(false));
document.getElementById("help").addEventListener("click", e => {{ if (e.target.id === "help") toggleHelp(false); }});
window.addEventListener("keydown", e => {{
  if (e.key === "h" || e.key === "H") toggleHelp();
  if (e.key === "r" || e.key === "R") {{ rx = 0.0; ry = 0.0; zoom = 0.95; draw(); }}
  if (e.key === "[") updateColors((Number(select.value) - 1 + payload.maps.length) % payload.maps.length);
  if (e.key === "]") updateColors((Number(select.value) + 1) % payload.maps.length);
  if (e.key === "Escape") toggleHelp(false);
}});
updateColors(0);
</script>
</body>
</html>
"""
    html = html.replace("{{", "{").replace("}}", "}")
    return html.replace("{title}", title).replace("{data}", data).replace("{metadata}", metadata)


def write_surface_report(surfaces, arrays=None, filename="surface_report.html",
                         title="BrainSpace Surface Report", cmap="viridis",
                         color_range=None, nan_color=(0.7, 0.7, 0.7, 1),
                         metadata=None):
    """Write a display-independent HTML report for left/right surface maps.

    Parameters
    ----------
    surfaces : dict or sequence
        Either ``{'lh': surf_lh, 'rh': surf_rh}`` or a two-item sequence of
        BrainSpace surfaces.
    arrays : str, ndarray, sequence or dict, optional
        Surface maps to include. One-dimensional arrays must contain left and
        right hemisphere values concatenated. Two-dimensional arrays are
        interpreted row-wise. Strings refer to point-data arrays already
        present on both surfaces. Dictionaries map display names to arrays.
    filename : str or path-like
        HTML file to write.
    title : str, optional
        Report title.
    cmap : str, optional
        Matplotlib or BrainSpace colormap name.
    color_range : {'sym'}, tuple or None, optional
        Color range. If None, each map uses its finite data range.
    nan_color : tuple, optional
        RGBA color for missing values.
    metadata : dict, optional
        Key/value metadata rendered above the viewer.

    Returns
    -------
    pathlib.Path
        Path to the written report.
    """
    surfaces = _as_surface_dict(surfaces)
    maps = []
    for name, values, opts in _iter_maps(surfaces, arrays):
        map_cmap = opts.get("cmap", cmap)
        map_color_range = opts.get("color_range", color_range)
        maps.append({
            "name": str(name),
            "values": {
                "lh": _json_list(values["lh"]),
                "rh": _json_list(values["rh"]),
            },
            "range": _map_range(values, map_color_range),
            "cmap": _resolve_cmap(map_cmap),
            "unit": str(opts.get("unit", "")),
            "description": str(opts.get("description", "")),
        })

    payload = {
        "title": str(title),
        "nanColor": _rgba255(nan_color),
        "surfaces": {
            "lh": _surface_payload(surfaces["lh"], "lh"),
            "rh": _surface_payload(surfaces["rh"], "rh"),
        },
        "maps": maps,
        "metadata": metadata or {},
    }

    filename = Path(filename)
    filename.write_text(_html_template(payload), encoding="utf-8")
    return filename
