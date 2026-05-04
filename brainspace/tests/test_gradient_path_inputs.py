"""Tests for path-like and vectorized inputs in GradientMaps.fit (issue #140)."""

from pathlib import Path

import numpy as np
import pytest

from brainspace.gradient import GradientMaps
from brainspace.gradient.gradient import _devectorize, _load_matrix


def _make_psd_matrix(n, seed):
    rs = np.random.RandomState(seed)
    a = rs.randn(n, n + 5)
    return a @ a.T


def test_devectorize_with_diagonal_roundtrip():
    rs = np.random.RandomState(0)
    a = rs.randn(6, 6)
    a = (a + a.T) / 2  # symmetric

    rows, cols = np.tril_indices(6, k=0)
    v = a[rows, cols]
    rec = _devectorize(v, discard_diagonal=False)
    np.testing.assert_allclose(rec, a)


def test_devectorize_without_diagonal_zeros_diag():
    rs = np.random.RandomState(1)
    a = rs.randn(5, 5)
    a = (a + a.T) / 2

    rows, cols = np.tril_indices(5, k=-1)
    v = a[rows, cols]
    rec = _devectorize(v, discard_diagonal=True)

    expected = a.copy()
    np.testing.assert_array_equal(np.diag(rec), np.zeros(5))
    expected_off = expected - np.diag(np.diag(expected))
    np.testing.assert_allclose(rec, expected_off)


def test_devectorize_invalid_length_raises():
    with pytest.raises(ValueError):
        _devectorize(np.zeros(7), discard_diagonal=False)


def test_load_matrix_from_path(tmp_path):
    a = _make_psd_matrix(8, seed=2)
    p = tmp_path / "m.npy"
    np.save(p, a)

    np.testing.assert_array_equal(_load_matrix(str(p)), a)
    np.testing.assert_array_equal(_load_matrix(p), a)  # PathLike


def test_load_matrix_from_path_vectorized(tmp_path):
    a = _make_psd_matrix(7, seed=3)
    a = (a + a.T) / 2
    rows, cols = np.tril_indices(7, k=0)
    v = a[rows, cols]
    p = tmp_path / "vec.npy"
    np.save(p, v)

    rec = _load_matrix(p, vectorized=True, discard_diagonal=False)
    np.testing.assert_allclose(rec, a)


def test_fit_with_path_single(tmp_path):
    a = _make_psd_matrix(20, seed=4)
    p = tmp_path / "single.npy"
    np.save(p, a)

    gm_arr = GradientMaps(n_components=3, random_state=0).fit(a)
    gm_path = GradientMaps(n_components=3, random_state=0).fit(str(p))

    np.testing.assert_allclose(gm_arr.lambdas_, gm_path.lambdas_)
    np.testing.assert_allclose(gm_arr.gradients_, gm_path.gradients_)


def test_fit_with_path_list(tmp_path):
    mats = [_make_psd_matrix(15, seed=s) for s in (5, 6, 7)]
    paths = []
    for i, m in enumerate(mats):
        p = tmp_path / f"m{i}.npy"
        np.save(p, m)
        paths.append(str(p))

    gm_arr = GradientMaps(n_components=3, random_state=0).fit(mats)
    gm_path = GradientMaps(n_components=3, random_state=0).fit(paths)

    for la, lp in zip(gm_arr.lambdas_, gm_path.lambdas_):
        np.testing.assert_allclose(la, lp)
    for ga, gp in zip(gm_arr.gradients_, gm_path.gradients_):
        np.testing.assert_allclose(ga, gp)


def test_fit_with_vectorized_array():
    a = _make_psd_matrix(18, seed=8)
    a = (a + a.T) / 2
    rows, cols = np.tril_indices(18, k=0)
    v = a[rows, cols]

    gm_full = GradientMaps(n_components=3, random_state=0).fit(a)
    gm_vec = GradientMaps(n_components=3, random_state=0).fit(
        v, vectorized=True)

    np.testing.assert_allclose(gm_full.lambdas_, gm_vec.lambdas_)
    np.testing.assert_allclose(gm_full.gradients_, gm_vec.gradients_)


def test_fit_with_vectorized_paths_joint(tmp_path):
    mats = []
    paths = []
    for s in (10, 11):
        m = _make_psd_matrix(12, seed=s)
        m = (m + m.T) / 2
        mats.append(m)
        rows, cols = np.tril_indices(12, k=0)
        v = m[rows, cols]
        p = tmp_path / f"vec{s}.npy"
        np.save(p, v)
        paths.append(Path(p))

    gm_arr = GradientMaps(n_components=2, approach='le',
                         alignment='joint', random_state=0).fit(mats)
    gm_path = GradientMaps(n_components=2, approach='le',
                          alignment='joint', random_state=0).fit(
        paths, vectorized=True)

    for la, lp in zip(gm_arr.lambdas_, gm_path.lambdas_):
        np.testing.assert_allclose(la, lp)
