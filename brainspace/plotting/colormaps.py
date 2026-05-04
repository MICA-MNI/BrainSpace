import numpy as np

yeo7_colors = np.array([[0, 0, 0, 255],
                        [0, 118, 14, 255],
                        [230, 148, 34, 255],
                        [205, 62, 78, 255],
                        [120, 18, 134, 255],
                        [220, 248, 164, 255],
                        [70, 130, 180, 255],
                        [196, 58, 250, 255]], dtype=np.uint8)

eco_kos_colors = np.array([[0, 0, 0, 255],
                           [126, 40, 127, 255],
                           [51, 104, 156, 255],
                           [167, 210, 140, 255],
                           [254, 205, 8, 255],
                           [255, 253, 25, 255]], dtype=np.uint8)

# 5-step Spectral palette (control points from ColorBrewer 2.0, Apache 2.0).
# Reference: https://colorbrewer2.org
spec_5_colors = np.array([[0, 0, 0, 255],
                          [50, 136, 189, 255],
                          [171, 221, 164, 255],
                          [235, 235, 181, 255],
                          [253, 174, 97, 255],
                          [213, 62, 79, 255]], dtype=np.uint8)


def _interp_lut(stops, n=256):
    """Build a 256-row RGBA uint8 lookup by linearly interpolating ``stops``.

    ``stops`` is an (k, 3) or (k, 4) array of RGB(A) values in [0, 255].
    Output always has alpha = 255 unless explicit alphas are provided.
    """
    stops = np.asarray(stops, dtype=np.float64)
    k = stops.shape[0]
    xs = np.linspace(0.0, 1.0, k)
    grid = np.linspace(0.0, 1.0, n)
    cols = stops.shape[1]
    out = np.empty((n, 4), dtype=np.float64)
    for c in range(min(cols, 4)):
        out[:, c] = np.interp(grid, xs, stops[:, c])
    if cols < 4:
        out[:, 3] = 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# Diverging blue->grey->red colormap. Built programmatically from 5 control
# points to keep the file small and reviewable instead of hand-listing 256
# RGB tuples.
_BuGyRd_stops = np.array([
    [33, 113, 181],   # blue
    [120, 170, 210],  # light blue
    [200, 200, 200],  # grey midpoint
    [220, 90, 60],    # warm red
    [103, 0, 13],     # dark red
])
BuGyRd = _interp_lut(_BuGyRd_stops, n=256)

colormaps = {
    'yeo7': yeo7_colors,
    'eco_kos': eco_kos_colors,
    'spec_5': spec_5_colors,
    'BuGyRd': BuGyRd,
}
