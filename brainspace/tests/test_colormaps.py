"""Tests for plotting.colormaps registrations and BuGyRd interpolation (#91)."""

import numpy as np

from brainspace.plotting.colormaps import (
    colormaps, yeo7_colors, eco_kos_colors, spec_5_colors, BuGyRd,
)


def test_registered_names():
    for name in ('yeo7', 'eco_kos', 'spec_5', 'BuGyRd', 'cat35'):
        assert name in colormaps
        cm = colormaps[name]
        assert cm.dtype == np.uint8
        assert cm.shape[1] == 4


def test_cat35_shape_and_first_slot():
    cm = colormaps['cat35']
    assert cm.shape == (35, 4)
    # First slot reserved for unknown/medial-wall: pure black, opaque.
    np.testing.assert_array_equal(cm[0], [0, 0, 0, 255])
    # All entries opaque.
    assert np.all(cm[:, 3] == 255)
    # All 35 colors should be distinct.
    unique_rows = {tuple(row) for row in cm}
    assert len(unique_rows) == 35


def test_BuGyRd_is_256_smooth_lut():
    assert BuGyRd.shape == (256, 4)
    assert np.all(BuGyRd[:, 3] == 255)
    # Endpoints should match the configured stops (within rounding).
    np.testing.assert_array_equal(BuGyRd[0, :3], [33, 113, 181])
    np.testing.assert_array_equal(BuGyRd[-1, :3], [103, 0, 13])
    # Smoothness: no large per-row jump in any channel.
    diffs = np.abs(np.diff(BuGyRd[:, :3].astype(int), axis=0))
    assert diffs.max() < 8


def test_categorical_palettes_have_alpha_255():
    for cm in (yeo7_colors, eco_kos_colors, spec_5_colors):
        assert np.all(cm[:, 3] == 255)
