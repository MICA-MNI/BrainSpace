"""Tests for shared input validation across embedding entry points (#147)."""

import numpy as np
import pytest

from brainspace.gradient.embedding import (
    diffusion_mapping, laplacian_eigenmaps, DiffusionMaps, LaplacianEigenmaps,
)


def _psd(n, seed):
    rs = np.random.RandomState(seed)
    a = rs.randn(n, n + 5)
    return a @ a.T


def test_diffusion_mapping_rejects_nan():
    a = _psd(8, seed=0)
    a[0, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        diffusion_mapping(a, n_components=2)


def test_diffusion_mapping_rejects_inf():
    a = _psd(8, seed=1)
    a[2, 3] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        diffusion_mapping(a, n_components=2)


def test_laplacian_eigenmaps_rejects_nan():
    a = _psd(8, seed=2)
    a[1, 4] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        laplacian_eigenmaps(a, n_components=2)


def test_diffusion_mapping_rejects_too_large_n_components():
    a = _psd(6, seed=3)
    with pytest.raises(ValueError, match="too large"):
        diffusion_mapping(a, n_components=10)


def test_laplacian_eigenmaps_rejects_too_large_n_components():
    a = _psd(6, seed=4)
    with pytest.raises(ValueError, match="too large"):
        laplacian_eigenmaps(a, n_components=10)


def test_class_wrappers_propagate_validation():
    a = _psd(8, seed=5)
    a[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        DiffusionMaps(n_components=2).fit(a)
    a2 = _psd(8, seed=6)
    a2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        LaplacianEigenmaps(n_components=2).fit(a2)


def test_n_components_zero_or_negative_rejected():
    a = _psd(8, seed=7)
    with pytest.raises(ValueError, match=">= 1"):
        diffusion_mapping(a, n_components=0)
    with pytest.raises(ValueError, match=">= 1"):
        laplacian_eigenmaps(a, n_components=-1)
